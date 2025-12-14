import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split, WeightedRandomSampler
from torch.nn.utils.rnn import pack_padded_sequence
import json
import glob
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import zipfile

# --- 1. KONFIGURÁCIÓ ---
HIDDEN_SIZE = 64
EMBED_DIM = 64
NUM_CLASSES = 5
NUM_EPOCHS = 30         
BATCH_SIZE = 16
LEARNING_RATE = 0.001
NUM_LAYERS = 2          
DROPOUT_RATE = 0.5      
WEIGHT_DECAY = 1e-4     

# --- PATH SETUP ---
CURRENT_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = CURRENT_DIR.parent  # This is the 'project' folder

# --- ADATBETÖLTÉS ZIP-BŐL ---
def load_all_data(project_root):
    data = []
    vocab = set()
    all_labels = [] 
    
    # 1. Find the unknown ZIP file
    zip_files = list(project_root.glob("*.zip"))
    
    if not zip_files:
        raise FileNotFoundError(f"No .zip file found in {project_root}")
    
    target_zip = zip_files[0] # We take the first one found
    print(f"Processing ZIP file: {target_zip.name}")
    
    # 2. Open ZIP without extracting
    with zipfile.ZipFile(target_zip, 'r') as z:
        # Iterate over every file inside the zip
        for filename in z.namelist():
            if not filename.endswith('.json'): 
                continue
            
            # LabelStudio exports might put weird hidden files (e.g., __MACOSX), skip them
            if "__MACOSX" in filename:
                continue

            try:
                # Read the file directly from memory
                with z.open(filename) as f:
                    # We accept JSONs that might contain non-utf8 chars gracefully, 
                    # but usually LabelStudio is UTF-8. 
                    content = json.load(f)
                    
                    if isinstance(content, dict): content = [content]
                    
                    for task in content:
                        try:
                            # --- Parsing Logic (Same as before) ---
                            text = task.get('data', {}).get('text', "")
                            if not text: continue
                            
                            annotations = task.get('annotations', [])
                            if not annotations: continue
                            result = annotations[0].get('result', [])
                            if not result: continue
                            
                            # Extract label
                            choice_str = result[0]['value']['choices'][0]
                            label = int(choice_str.split('-')[0])
                            
                            tokens = text.split()
                            data.append({"text": tokens, "label": label})
                            all_labels.append(label - 1)
                            for w in tokens: vocab.add(w)
                            # -------------------------------------
                        except Exception: 
                            continue
            except Exception as e:
                print(f"Skipping bad file inside zip: {filename} ({e})")
                continue

    return data, list(vocab), all_labels

# Load the data
try:
    raw_data, vocab_list, full_labels_list = load_all_data(PROJECT_ROOT)
    print(f"Total samples loaded: {len(raw_data)}")
except FileNotFoundError as e:
    print(e)
    exit()

# --- 3. SAMPLER ELŐKÉSZÍTÉSE (A LÉNYEG) ---
# Itt trükközünk: Nem random splitet használunk, hanem manuálisan választjuk szét,
# hogy a Train sethez tudjunk súlyokat számolni.

word2idx = {w: i+1 for i, w in enumerate(vocab_list)}
word2idx["<PAD>"] = 0
vocab_size = len(word2idx)

# Adatok keverése és szétvágása (80/20) manuálisan indexekkel
indices = list(range(len(raw_data)))
np.random.shuffle(indices)
split = int(np.floor(0.8 * len(raw_data)))
train_indices, test_indices = indices[:split], indices[split:]

# Segédfüggvény az adatok kinyeréséhez indexek alapján
def get_subset(indices, data):
    return [data[i] for i in indices]

train_data_subset = get_subset(train_indices, raw_data)
test_data_subset = get_subset(test_indices, raw_data)

# Súlyok számítása CSAK a tréning halmazra ( Oversamplinghez )
train_labels = [d['label']-1 for d in train_data_subset]
class_counts = np.bincount(train_labels)
class_weights = 1. / class_counts
sample_weights = [class_weights[label] for label in train_labels]
sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

print(f"Ritka osztályok túlmintavételezése beállítva. (Train size: {len(train_data_subset)})")

# --- 4. DATASET ÉS DATALOADER ---
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

train_dataset = LegalDataset(train_data_subset, word2idx)
test_dataset = LegalDataset(test_data_subset, word2idx)

# FONTOS: shuffle=False kell, mert a sampler végzi a keverést!
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

# --- 5. MODELL (DEEP LSTM) ---
class LegalModelPlan5(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size, num_classes, dropout_rate, num_layers):
        super(LegalModelPlan5, self).__init__()
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

# --- 6. TANÍTÁS ---
model = LegalModelPlan5(vocab_size, EMBED_DIM, HIDDEN_SIZE, NUM_CLASSES, DROPOUT_RATE, NUM_LAYERS)

# VISSZAÁLLUNK SIMA CROSS ENTROPY-RA (Mert a sampler már megoldja a súlyozást)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

train_losses = []
val_accuracies = []
best_val_acc_seen = 0.0

print(f"\n--- 5. Terv (Oversampling) Indítása ---")

for epoch in range(NUM_EPOCHS):
    model.train()
    total_loss = 0
    
    for X_batch, y_batch, lengths in train_loader:
        optimizer.zero_grad()
        outputs = model(X_batch, lengths)
        loss = criterion(outputs, y_batch)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
    
    avg_loss = total_loss / len(train_loader)
    train_losses.append(avg_loss)
    
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
    scheduler.step(val_acc)
    
    msg = ""
    if val_acc > best_val_acc_seen:
        best_val_acc_seen = val_acc
        msg = "(Új rekord!)"
    
    current_lr = optimizer.param_groups[0]['lr']
    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] | Loss: {avg_loss:.4f} | Val Acc: {val_acc:.2f}% | LR: {current_lr:.6f} {msg}")

# --- 7. VIZUALIZÁCIÓ ---
model.eval()
all_preds = []
all_labels = []
with torch.no_grad():
    for X_batch, y_batch, lengths in test_loader:
        outputs = model(X_batch, lengths)
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(y_batch.cpu().numpy())

print(f"\nFinal Accuracy: {accuracy_score(all_labels, all_preds):.4f}")
print(f"Final F1 Score: {f1_score(all_labels, all_preds, average='weighted', zero_division=0):.4f}")

target_names = [f"Class {i+1}" for i in sorted(list(set(all_labels)))]
print("\n" + classification_report(all_labels, all_preds, target_names=target_names, zero_division=0))

plt.figure(figsize=(8, 6))
cm = confusion_matrix(all_labels, all_preds)
sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', xticklabels=target_names, yticklabels=target_names)
plt.title('Confusion Matrix - Plan 5 (Oversampling)')
plt.savefig('confusion_matrix_plan5.png')