import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pack_padded_sequence
import json
import glob
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. KONFIGURÁCIÓ (4. TERV: OPTIMALIZÁCIÓ) ---
HIDDEN_SIZE = 64
EMBED_DIM = 64
NUM_CLASSES = 5
NUM_EPOCHS = 40         # Több epoch, mert a Scheduler lassítani fog
BATCH_SIZE = 16
LEARNING_RATE = 0.001
NUM_LAYERS = 2          # Marad a Deep architektúra
DROPOUT_RATE = 0.5      # Marad az erős dropout
WEIGHT_DECAY = 1e-4     

CURRENT_DIR = Path(__file__).parent.absolute() if '__file__' in locals() else Path(".").absolute()
DATA_DIR = CURRENT_DIR.parent / "consensus"

# --- 2. ADATBETÖLTÉS ---
def load_all_data(data_dir):
    data = []
    vocab = set()
    all_labels_for_weights = [] # Gyűjtjük a címkéket a súlyozáshoz
    
    search_path = data_dir / "**" / "*.json"
    json_files = glob.glob(str(search_path), recursive=True)
    
    print(f"Fájlok keresése... Találat: {len(json_files)}")
    
    for file_path in json_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = json.load(f)
                if isinstance(content, dict): content = [content]
                for task in content:
                    try:
                        text = task.get('data', {}).get('text', "")
                        if not text: continue
                        
                        annotations = task.get('annotations', [])
                        if not annotations: continue
                        result = annotations[0].get('result', [])
                        if not result: continue
                        choice_str = result[0]['value']['choices'][0]
                        label = int(choice_str.split('-')[0])
                        
                        tokens = text.split()
                        data.append({"text": tokens, "label": label})
                        all_labels_for_weights.append(label - 1) # 0-4 skála
                        for w in tokens: vocab.add(w)
                    except: continue
        except: continue
    return data, list(vocab), all_labels_for_weights

raw_data, vocab_list, all_train_labels = load_all_data(DATA_DIR)
print(f"Összes minta: {len(raw_data)}")

# --- 3. SÚLYOK KISZÁMÍTÁSA (ÚJ!) ---
# Kiszámoljuk, melyik osztály mennyire ritka
# class_weight = n_samples / (n_classes * n_samples_j)
class_weights = compute_class_weight(
    class_weight='balanced', 
    classes=np.unique(all_train_labels), 
    y=all_train_labels
)
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float)
print(f"\nSzámított osztály súlyok (Class 1 -> Class 5):")
print(class_weights_tensor)
print("(A ritkább osztályok nagyobb súlyt kaptak)\n")

# --- 4. ELŐKÉSZÍTÉS ---
word2idx = {w: i+1 for i, w in enumerate(vocab_list)}
word2idx["<PAD>"] = 0
vocab_size = len(word2idx)

class LegalDataset(Dataset):
    def __init__(self, data, word2idx):
        self.data = data
        self.word2idx = word2idx
    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        item = self.data[idx]
        tokens = item['text']
        label = item['label'] - 1
        indices = [self.word2idx.get(w, 0) for w in tokens]
        return torch.tensor(indices, dtype=torch.long), torch.tensor(label, dtype=torch.long)

def collate_fn(batch):
    batch.sort(key=lambda x: len(x[0]), reverse=True)
    sequences, labels = zip(*batch)
    lengths = [len(s) for s in sequences]
    max_len = max(lengths)
    padded_seqs = torch.zeros(len(sequences), max_len, dtype=torch.long)
    for i, seq in enumerate(sequences): padded_seqs[i, :len(seq)] = seq
    return padded_seqs, torch.stack(labels), torch.tensor(lengths)

full_dataset = LegalDataset(raw_data, word2idx)
train_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_size
train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

# --- 5. MODELL (MARAD A PLAN 3) ---
class LegalModelPlan4(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size, num_classes, dropout_rate, num_layers):
        super(LegalModelPlan4, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_size, num_layers=num_layers, batch_first=True, bidirectional=True, dropout=dropout_rate)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x, lengths):
        embedded = self.embedding(x)
        packed_embedded = pack_padded_sequence(embedded, lengths.cpu(), batch_first=True)
        _, (hidden_state, _) = self.lstm(packed_embedded)
        hidden_forward = hidden_state[-2]
        hidden_backward = hidden_state[-1]
        cat_hidden = torch.cat((hidden_forward, hidden_backward), dim=1)
        out = self.dropout(cat_hidden)
        out = self.fc(out)
        return out

# --- 6. TANÍTÁS (SCHEDULERREL ÉS SÚLYOKKAL) ---
model = LegalModelPlan4(vocab_size, EMBED_DIM, HIDDEN_SIZE, NUM_CLASSES, DROPOUT_RATE, NUM_LAYERS)

# ÚJ: Súlyozott Loss Function
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

# ÚJ: Learning Rate Scheduler (Ha nem javul a loss, csökkenti a LR-t)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

train_losses = []
val_accuracies = []
best_val_acc_seen = 0.0

print(f"\n--- 4. Terv (Optimization) Indítása ---")
print("Cél: Ritka osztályok javítása súlyozással + LR finomhangolás")

for epoch in range(NUM_EPOCHS):
    model.train()
    total_loss = 0
    
    for X_batch, y_batch, lengths in train_loader:
        optimizer.zero_grad()
        outputs = model(X_batch, lengths)
        loss = criterion(outputs, y_batch)
        loss.backward()
        
        # Opcionális: Gradient Clipping a stabilitásért
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        total_loss += loss.item()
    
    avg_loss = total_loss / len(train_loader)
    train_losses.append(avg_loss)
    
    # Validáció
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for X_batch, y_batch, lengths in test_loader:
            outputs = model(X_batch, lengths)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == y_batch).sum().item()
            total += y_batch.size(0)
    
    val_acc = correct / total * 100
    val_accuracies.append(val_acc)
    
    # Scheduler léptetése (a validációs pontosság alapján)
    scheduler.step(val_acc)
    
    if val_acc > best_val_acc_seen:
        best_val_acc_seen = val_acc
        msg = "(Új rekord!)"
    else:
        msg = ""
    
    # Kiírjuk az aktuális Learning Rate-et is
    current_lr = optimizer.param_groups[0]['lr']
    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] | Loss: {avg_loss:.4f} | Val Acc: {val_acc:.2f}% | LR: {current_lr:.6f} {msg}")

print(f"\nTanítás kész. Rekord: {best_val_acc_seen:.2f}%")

# --- 7. VIZUALIZÁCIÓ ---
# Most különösen érdekes lesz a Confusion Matrix az 1-es és 2-es osztályoknál
model.eval()
all_preds = []
all_labels = []
with torch.no_grad():
    for X_batch, y_batch, lengths in test_loader:
        outputs = model(X_batch, lengths)
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(y_batch.cpu().numpy())

accuracy = accuracy_score(all_labels, all_preds)
f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)

print(f"Final Accuracy: {accuracy:.4f}")
print(f"Final F1 Score: {f1:.4f}")
target_names = [f"Class {i+1}" for i in sorted(list(set(all_labels)))]
print("\n" + classification_report(all_labels, all_preds, target_names=target_names, zero_division=0))

plt.figure(figsize=(8, 6))
cm = confusion_matrix(all_labels, all_preds)
sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', xticklabels=target_names, yticklabels=target_names)
plt.title('Confusion Matrix - Plan 4 (Weighted)')
plt.savefig('confusion_matrix_plan4.png')
print("Képek elmentve.")