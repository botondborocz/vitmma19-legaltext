import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score
import seaborn as sns
from matplotlib import pyplot as plt
import torch.nn.functional as F
import sys
import io

# --- IMPORT FROM SHARED MODULE ---
try:
    from data_loader import load_data_from_zip, LegalDataset, collate_fn
except ImportError:
    sys.path.append(str(Path(__file__).parent))
    from data_loader import load_data_from_zip, LegalDataset, collate_fn

# --- 1. CONFIGURATION ---
EMBED_DIM = 300         # FastText miatt fix
HIDDEN_SIZE = 64        # LSTM méret
NUM_CLASSES = 5
NUM_EPOCHS = 20         # Kevesebb is elég lesz a OneCycleLR miatt
BATCH_SIZE = 32         
LEARNING_RATE = 0.003   # Magasabb induló LR, a scheduler majd kezeli
DROPOUT_RATE = 0.5      
WEIGHT_DECAY = 1e-4     
PATIENCE = 5            

# Path Setup
CURRENT_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = CURRENT_DIR.parent
OUTPUT_DIR = PROJECT_ROOT / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
VECTOR_PATH = PROJECT_ROOT / "resources" / "cc.hu.300.vec"
CHECKPOINT_PATH = OUTPUT_DIR / "checkpoint_plan10.pt"

# --- 2. EARLY STOPPING ---
class EarlyStopping:
    def __init__(self, patience=5, verbose=False, delta=0, path='checkpoint.pt'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

# --- 3. VEKTOROK BETÖLTÉSE ---
def load_vectors(fname):
    print(f"Vektorok betöltése innen: {fname}...")
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = np.array(tokens[1:], dtype=np.float32)
    print(f"Betöltve {len(data)} db magyar szóvektor.")
    return data

# --- 4. ADATOK ELŐKÉSZÍTÉSE ---
try:
    if not VECTOR_PATH.exists():
        raise FileNotFoundError(f"HIÁNYZIK: {VECTOR_PATH}")
    word_vectors = load_vectors(VECTOR_PATH)
    raw_data, vocab_list = load_data_from_zip(PROJECT_ROOT)
    print(f"Összes minta: {len(raw_data)}")
except Exception as e:
    print(f"HIBA: {e}")
    exit()

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

full_dataset = LegalDataset(raw_data, word2idx)

if len(full_dataset) > 0:
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])
    
    # OSZTÁLY SÚLYOK (Ritka osztályok büntetése)
    train_indices = train_dataset.indices
    train_labels = [raw_data[i]['label'] - 1 for i in train_indices]
    class_counts = np.bincount(train_labels, minlength=5)
    total_samples = len(train_labels)
    class_weights = total_samples / (5 * class_counts)
    class_weights_tensor = torch.FloatTensor(class_weights)
    print(f"Osztály súlyok: {class_weights}")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
else:
    print("Error: Dataset is empty.")
    exit()

# --- 5. MODEL: Bi-LSTM + SELF-ATTENTION ---
class SelfAttention(nn.Module):
    def __init__(self, hidden_size):
        super(SelfAttention, self).__init__()
        self.hidden_size = hidden_size
        self.projection = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self, encoder_outputs):
        # encoder_outputs shape: [batch, seq_len, hidden_size]
        # Calculate energy for each word
        energy = self.projection(encoder_outputs) # [batch, seq_len, 1]
        weights = F.softmax(energy.squeeze(-1), dim=1) # [batch, seq_len]
        
        # Weighted sum of encoder outputs
        # weights.unsqueeze(1) -> [batch, 1, seq_len]
        # bmm (batch matrix multiplication) -> [batch, 1, hidden_size]
        outputs = torch.bmm(weights.unsqueeze(1), encoder_outputs).squeeze(1)
        return outputs, weights

class LegalAttnModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size, num_classes, dropout_rate, embedding_matrix):
        super(LegalAttnModel, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.embedding.weight.data.copy_(torch.from_numpy(embedding_matrix))
        self.embedding.weight.requires_grad = True # Fine-tuning enabled
        
        # Bi-LSTM
        self.lstm = nn.LSTM(embed_dim, hidden_size, batch_first=True, bidirectional=True)
        
        # Attention Layer (Input size is hidden_size * 2 because of Bidirectional)
        self.attention = SelfAttention(hidden_size * 2)
        
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, num_classes)
        )

    def forward(self, x, lengths):
        embedded = self.embedding(x)
        
        # Pack padded sequence
        packed_embedded = torch.nn.utils.rnn.pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=True)
        packed_output, (hidden, cell) = self.lstm(packed_embedded)
        
        # Unpack sequence
        output, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        
        # Attention Mechanism
        attn_output, attn_weights = self.attention(output)
        
        out = self.dropout(attn_output)
        return self.fc(out)

# --- 6. TRAINING ---
model = LegalAttnModel(vocab_size, EMBED_DIM, HIDDEN_SIZE, NUM_CLASSES, DROPOUT_RATE, embedding_matrix)

# Weighted Loss
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

# OneCycleLR Scheduler - Gyors konvergencia
scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=LEARNING_RATE, 
                                          steps_per_epoch=len(train_loader), 
                                          epochs=NUM_EPOCHS)

early_stopping = EarlyStopping(patience=PATIENCE, verbose=True, path=str(CHECKPOINT_PATH))

print(f"\n--- Plan 10: Bi-LSTM + Attention + FastText + OneCycleLR ---")

train_losses = []
val_accuracies = []

for epoch in range(NUM_EPOCHS):
    model.train()
    total_loss = 0
    
    for X_batch, y_batch, lengths in train_loader:
        optimizer.zero_grad()
        outputs = model(X_batch, lengths)
        loss = criterion(outputs, y_batch)
        loss.backward()
        
        # Gradient Clipping (LSTM-hez fontos)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        scheduler.step() # Minden batch után léptetjük!
        total_loss += loss.item()
    
    avg_loss = total_loss / len(train_loader)
    train_losses.append(avg_loss)
    
    # Validation
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for X_batch, y_batch, lengths in test_loader:
            outputs = model(X_batch, lengths)
            loss = criterion(outputs, y_batch)
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == y_batch).sum().item()
            total += y_batch.size(0)
    
    avg_val_loss = val_loss / len(test_loader)
    val_acc = correct / total * 100
    val_accuracies.append(val_acc)
    
    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] | Train Loss: {avg_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.2f}%")
    
    early_stopping(avg_val_loss, model)
    if early_stopping.early_stop:
        print("Early stopping aktiválva!")
        break

# --- 7. EVALUATION ---
print("\nLegjobb modell visszatöltése...")
model.load_state_dict(torch.load(CHECKPOINT_PATH))

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

# Save Metrics
metrics_file = OUTPUT_DIR / "plan10_metrics.txt"
with open(metrics_file, "w") as f:
    f.write("Plan 10 (Bi-LSTM + Attention) Metrics\n")
    f.write("=====================================\n")
    f.write(f"Accuracy:  {accuracy:.4f}\n")
    f.write(f"Precision: {f1:.4f}\n") 
    f.write(f"Recall:    {recall:.4f}\n")
    f.write(f"F1 Score:  {f1:.4f}\n")

# Save Matrix
plt.figure(figsize=(8, 6))
cm = confusion_matrix(all_labels, all_preds)
target_names = [f"Class {i+1}" for i in sorted(list(set(all_labels)))]
sns.heatmap(cm, annot=True, fmt='d', cmap='RdPu', xticklabels=target_names, yticklabels=target_names)
plt.title('Confusion Matrix - Plan 10')
matrix_file = OUTPUT_DIR / "confusion_matrix_plan10.png"
plt.savefig(matrix_file)
print(f"Eredmények mentve: {metrics_file}, {matrix_file}")