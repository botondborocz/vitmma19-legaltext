import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler, random_split
from torch.nn.utils.rnn import pack_padded_sequence
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, recall_score
import matplotlib.pyplot as plt
import seaborn as sns
import io

# --- IMPORT A SAJÁT MODULBÓL ---
# Feltételezve, hogy a data_loader.py ugyanott van (src mappa)
try:
    from data_loader import load_data_from_zip, LegalDataset, collate_fn
except ImportError:
    # Ha esetleg nem importálható közvetlenül, próbáljuk meg relatívan
    import sys
    sys.path.append(str(Path(__file__).parent))
    from data_loader import load_data_from_zip, LegalDataset, collate_fn

# --- 1. KONFIGURÁCIÓ ---
# FONTOS: A fastText vektorok 300 dimenziósak!
EMBED_DIM = 300         
HIDDEN_SIZE = 128       
NUM_CLASSES = 5
NUM_EPOCHS = 20         
BATCH_SIZE = 32         
LEARNING_RATE = 0.001
NUM_LAYERS = 2          
DROPOUT_RATE = 0.5      
WEIGHT_DECAY = 1e-4     

# Elérési utak
CURRENT_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = CURRENT_DIR.parent
OUTPUT_DIR = PROJECT_ROOT / "output" # ÚJ: Kimeneti mappa
VECTOR_PATH = PROJECT_ROOT / "resources" / "cc.hu.300.vec"

# Output mappa létrehozása, ha nem létezik
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --- 2. ELŐRE TANÍTOTT VEKTOROK BETÖLTÉSE ---
def load_vectors(fname):
    print(f"Vektorok betöltése innen: {fname} (Ez eltarthat pár percig...)")
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split()) 
    data = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = np.array(tokens[1:], dtype=np.float32)
    print(f"Betöltve {len(data)} db magyar szóvektor.")
    return data

# --- 3. ADATBETÖLTÉS (KÖZÖS MODULBÓL) ---
try:
    if not VECTOR_PATH.exists():
        raise FileNotFoundError(f"HIÁNYZIK A VEKTOR FÁJL! Töltsd le a cc.hu.300.vec fájlt és tedd ide: {VECTOR_PATH}")
    
    # 1. Először a vektorokat töltsük be
    word_vectors = load_vectors(VECTOR_PATH)
    
    # 2. Aztán az adatokat a közös loaderrel
    raw_data, vocab_list = load_data_from_zip(PROJECT_ROOT)
    print(f"Összes minta: {len(raw_data)}")
    
except Exception as e:
    print(f"\nHIBA: {e}")
    exit()

# --- 4. ELŐKÉSZÍTÉS & EMBEDDING MÁTRIX ---
word2idx = {w: i+1 for i, w in enumerate(vocab_list)}
word2idx["<PAD>"] = 0
vocab_size = len(word2idx)

print("Embedding mátrix építése...")
embedding_matrix = np.zeros((vocab_size, EMBED_DIM))
found_count = 0

for word, i in word2idx.items():
    if word in word_vectors:
        embedding_matrix[i] = word_vectors[word]
        found_count += 1
    else:
        embedding_matrix[i] = np.random.normal(scale=0.6, size=(EMBED_DIM, ))

print(f"Lefedettség: {found_count/vocab_size*100:.2f}%")

# --- 5. DATASET & OVERSAMPLING ---
# Adatok szétválasztása manuálisan, hogy a sampler-t be tudjuk állítani a TRAIN-re
indices = list(range(len(raw_data)))
np.random.shuffle(indices)
split = int(np.floor(0.8 * len(raw_data)))
train_indices, test_indices = indices[:split], indices[split:]

def get_subset(indices, data): return [data[i] for i in indices]
train_data_subset = get_subset(train_indices, raw_data)
test_data_subset = get_subset(test_indices, raw_data)

# Oversampling súlyok számítása
train_labels = [d['label']-1 for d in train_data_subset]
class_counts = np.bincount(train_labels, minlength=5)
class_counts[class_counts == 0] = 1 
class_weights = 1. / class_counts
sample_weights = [class_weights[label] for label in train_labels]
sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

# Dataset objektumok
train_dataset = LegalDataset(train_data_subset, word2idx)
test_dataset = LegalDataset(test_data_subset, word2idx)

# DataLoaderek
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

# --- 6. MODELL (PRE-TRAINED EMBEDDINGS) ---
class LegalModelPlan6(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size, num_classes, dropout_rate, num_layers, embedding_matrix):
        super(LegalModelPlan6, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        # Súlyok másolása
        self.embedding.weight.data.copy_(torch.from_numpy(embedding_matrix))
        self.embedding.weight.requires_grad = True # Finomhangolás engedélyezése
        
        self.lstm = nn.LSTM(embed_dim, hidden_size, num_layers=num_layers, batch_first=True, bidirectional=True, dropout=dropout_rate)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x, lengths):
        embedded = self.embedding(x)
        packed_embedded = pack_padded_sequence(embedded, lengths.cpu(), batch_first=True)
        _, (hidden_state, _) = self.lstm(packed_embedded)
        cat_hidden = torch.cat((hidden_state[-2], hidden_state[-1]), dim=1)
        out = self.dropout(cat_hidden)
        out = self.fc(out)
        return out

# --- 7. TANÍTÁS ---
model = LegalModelPlan6(vocab_size, EMBED_DIM, HIDDEN_SIZE, NUM_CLASSES, DROPOUT_RATE, NUM_LAYERS, embedding_matrix)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

train_losses = []
val_accuracies = []
best_val_acc_seen = 0.0

print(f"\n--- 6. Terv (Pre-trained Word2Vec + Oversampling) Indítása ---")

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

# --- 8. KIÉRTÉKELÉS ÉS MENTÉS ---
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
recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)

print(f"\nFinal Accuracy: {accuracy:.4f}")
print(f"Final F1 Score: {f1:.4f}")

# Metrikák mentése
metrics_file = OUTPUT_DIR / "plan6_metrics.txt"
with open(metrics_file, "w") as f:
    f.write("Plan 6 (Pre-trained + Oversampling) Metrics\n")
    f.write("===========================================\n")
    f.write(f"Accuracy:  {accuracy:.4f}\n")
    f.write(f"Precision: {f1:.4f}\n") # Precision helyett most az F1-et írjuk a formátum miatt, vagy számolhatod külön
    f.write(f"Recall:    {recall:.4f}\n")
    f.write(f"F1 Score:  {f1:.4f}\n")

print(f"Metrikák elmentve: {metrics_file}")

# Confusion Matrix mentése
plt.figure(figsize=(8, 6))
cm = confusion_matrix(all_labels, all_preds)
target_names = [f"Class {i+1}" for i in sorted(list(set(all_labels)))]
sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', xticklabels=target_names, yticklabels=target_names)
plt.title('Confusion Matrix - Plan 6')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')

matrix_file = OUTPUT_DIR / "confusion_matrix_plan6.png"
plt.savefig(matrix_file)
print(f"Confusion Matrix elmentve: {matrix_file}")