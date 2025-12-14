import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
import seaborn as sns
from matplotlib import pyplot as plt
import torch.nn.functional as F
import sys

# --- IMPORT FROM SHARED MODULE ---
try:
    from data_loader import load_data_from_zip, LegalDataset, collate_fn
except ImportError:
    sys.path.append(str(Path(__file__).parent))
    from data_loader import load_data_from_zip, LegalDataset, collate_fn

# --- 1. CONFIGURATION ---
EMBED_DIM = 100         
NUM_FILTERS = 100       
FILTER_SIZES = [3, 4, 5] 
NUM_CLASSES = 5
NUM_EPOCHS = 30         # Felemelhetjük bátran, az Early Stopping majd megállítja!
BATCH_SIZE = 32         
LEARNING_RATE = 0.001
DROPOUT_RATE = 0.5      
WEIGHT_DECAY = 1e-3     
PATIENCE = 5            # ÚJ: Ha 5 epochig nem javul a loss, leállunk.

# Path Setup
CURRENT_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = CURRENT_DIR.parent
OUTPUT_DIR = PROJECT_ROOT / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_PATH = OUTPUT_DIR / "checkpoint_plan8.pt" # Ide mentjük az ideiglenes legjobb modellt

# --- 2. EARLY STOPPING OSZTÁLY (BELSŐ) ---
class EarlyStopping:
    """
    Korai leállítás, ha a validációs loss nem javul.
    Elmenti a legjobb modellt (checkpoint).
    """
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
        '''Elmenti a modellt, ha csökken a validációs loss.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

# --- 3. DATA LOADING ---
raw_data, vocab_list = load_data_from_zip(PROJECT_ROOT)
print(f"Total samples loaded: {len(raw_data)}")

# --- 4. PREPROCESSING ---
word2idx = {w: i+1 for i, w in enumerate(vocab_list)}
word2idx["<PAD>"] = 0
vocab_size = len(word2idx)

full_dataset = LegalDataset(raw_data, word2idx)

if len(full_dataset) > 0:
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
else:
    print("Error: Dataset is empty.")
    exit()

# --- 5. MODEL ARCHITECTURE (CNN) ---
class LegalCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_filters, filter_sizes, num_classes, dropout_rate):
        super(LegalCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=embed_dim, 
                      out_channels=num_filters, 
                      kernel_size=fs) 
            for fs in filter_sizes
        ])
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(num_filters * len(filter_sizes), num_classes)

    def forward(self, x, lengths):
        embedded = self.embedding(x) 
        embedded = embedded.permute(0, 2, 1) 
        conved = [F.relu(conv(embedded)) for conv in self.convs]
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        cat = torch.cat(pooled, dim=1) 
        out = self.dropout(cat)
        return self.fc(out)

# --- 6. TRAINING LOOP WITH EARLY STOPPING ---
model = LegalCNN(vocab_size, EMBED_DIM, NUM_FILTERS, FILTER_SIZES, NUM_CLASSES, DROPOUT_RATE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

# Inicializáljuk az Early Stoppingot
early_stopping = EarlyStopping(patience=PATIENCE, verbose=True, path=str(CHECKPOINT_PATH))

print(f"\n--- Plan 8 Started: Multi-Channel CNN + Early Stopping ---")

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
        optimizer.step()
        total_loss += loss.item()
    
    avg_loss = total_loss / len(train_loader)
    train_losses.append(avg_loss)
    
    # Validation
    model.eval()
    val_loss = 0 # Loss-t is mérünk most már a validáción!
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
    
    # --- EARLY STOPPING CHECK ---
    early_stopping(avg_val_loss, model)
    
    if early_stopping.early_stop:
        print("Early stopping aktiválva! Leállítás...")
        break

# --- 7. LOAD BEST MODEL ---
# Nagyon fontos: betöltjük a legjobbnak ítélt állapotot, nem az utolsót!
print("\nLegjobb modell visszatöltése...")
model.load_state_dict(torch.load(CHECKPOINT_PATH))

# --- 8. EVALUATION ---
print("\n--- Final Evaluation (Best Model) ---")
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

print(f"Accuracy: {accuracy:.4f}")
print(f"F1 Score: {f1:.4f}")

# Save Metrics
metrics_file = OUTPUT_DIR / "plan8_metrics.txt"
with open(metrics_file, "w") as f:
    f.write("Plan 8 (CNN + Early Stopping) Metrics\n")
    f.write("=====================================\n")
    f.write(f"Accuracy:  {accuracy:.4f}\n")
    f.write(f"Precision: {f1:.4f}\n") 
    f.write(f"Recall:    {recall:.4f}\n")
    f.write(f"F1 Score:  {f1:.4f}\n")
print(f"Metrics saved to {metrics_file}")

# Save Confusion Matrix
plt.figure(figsize=(8, 6))
cm = confusion_matrix(all_labels, all_preds)
target_names = [f"Class {i+1}" for i in sorted(list(set(all_labels)))]
sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', xticklabels=target_names, yticklabels=target_names)
plt.title('Confusion Matrix - Plan 8 (CNN + Early Stop)')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')

matrix_file = OUTPUT_DIR / "confusion_matrix_plan8.png"
plt.savefig(matrix_file)
print(f"Confusion Matrix saved to {matrix_file}")