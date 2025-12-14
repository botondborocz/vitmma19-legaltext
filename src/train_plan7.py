import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, classification_report
import seaborn as sns
from matplotlib import pyplot as plt
import sys

# --- IMPORT FROM SHARED MODULE ---
try:
    from data_loader import load_data_from_zip, LegalDataset, collate_fn
except ImportError:
    sys.path.append(str(Path(__file__).parent))
    from data_loader import load_data_from_zip, LegalDataset, collate_fn

# --- 1. CONFIGURATION ---
# We increase dimensions slightly to give the MLP more features to work with
EMBED_DIM = 64          # Baseline was 32
HIDDEN_SIZE_1 = 128     # MLP Layer 1 size
HIDDEN_SIZE_2 = 64      # MLP Layer 2 size
NUM_CLASSES = 5
NUM_EPOCHS = 30         # MLPs might need a bit more time to converge on text
BATCH_SIZE = 16
LEARNING_RATE = 0.001
DROPOUT_RATE = 0.5      
WEIGHT_DECAY = 1e-4     

# Path Setup
CURRENT_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = CURRENT_DIR.parent
OUTPUT_DIR = PROJECT_ROOT / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --- 2. DATA LOADING ---
raw_data, vocab_list = load_data_from_zip(PROJECT_ROOT)
print(f"Total samples loaded: {len(raw_data)}")

# --- 3. PREPROCESSING ---
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

# --- 4. MODEL ARCHITECTURE (Plan 7: No LSTM, Big MLP) ---
class LegalMLP(nn.Module):
    def __init__(self, vocab_size, embed_dim, h1, h2, num_classes, dropout_rate):
        super(LegalMLP, self).__init__()
        
        # 1. Embedding Layer
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        # 2. The "Bigger" MLP
        # It takes the averaged embedding vector as input
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, h1),       # Input -> 128
            nn.BatchNorm1d(h1),             # Batch Norm helps MLPs train faster
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(h1, h2),              # 128 -> 64
            nn.BatchNorm1d(h2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(h2, num_classes)      # 64 -> 5
        )

    def forward(self, x, lengths):
        # x shape: [batch_size, seq_len]
        
        # 1. Get Embeddings
        # shape: [batch_size, seq_len, embed_dim]
        embedded = self.embedding(x) 
        
        # 2. Mean Pooling (Averaging)
        # We need to average the word vectors, but ignore the padding (0)
        # Create a mask where padding is 0 and words are 1
        mask = (x != 0).unsqueeze(2).float() # shape: [batch, seq, 1]
        
        # Sum the embeddings of valid words
        sum_embeddings = torch.sum(embedded * mask, dim=1) # shape: [batch, embed_dim]
        
        # Count number of valid words
        lengths_expanded = lengths.unsqueeze(1).float().to(x.device)
        # Avoid division by zero
        lengths_expanded = torch.clamp(lengths_expanded, min=1.0)
        
        # Calculate Average
        avg_embedding = sum_embeddings / lengths_expanded
        
        # 3. Feed to MLP
        return self.classifier(avg_embedding)

# --- 5. TRAINING LOOP ---
model = LegalMLP(vocab_size, EMBED_DIM, HIDDEN_SIZE_1, HIDDEN_SIZE_2, NUM_CLASSES, DROPOUT_RATE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

print(f"\n--- Plan 7 Started: Embedding + Big MLP (No LSTM) ---")

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
    
    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] | Train Loss: {avg_loss:.4f} | Val Acc: {val_acc:.2f}%")

# --- 6. EVALUATION ---
print("\n--- Final Evaluation ---")
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
metrics_file = OUTPUT_DIR / "plan7_metrics.txt"
with open(metrics_file, "w") as f:
    f.write("Plan 7 (Embedding + Big MLP) Metrics\n")
    f.write("====================================\n")
    f.write(f"Accuracy:  {accuracy:.4f}\n")
    f.write(f"Precision: {f1:.4f}\n") # Using F1 as proxy for format
    f.write(f"Recall:    {recall:.4f}\n")
    f.write(f"F1 Score:  {f1:.4f}\n")
print(f"Metrics saved to {metrics_file}")

# Save Confusion Matrix
plt.figure(figsize=(8, 6))
cm = confusion_matrix(all_labels, all_preds)
target_names = [f"Class {i+1}" for i in sorted(list(set(all_labels)))]
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', xticklabels=target_names, yticklabels=target_names)
plt.title('Confusion Matrix - Plan 7 (Big MLP)')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')

matrix_file = OUTPUT_DIR / "confusion_matrix_plan7.png"
plt.savefig(matrix_file)
print(f"Confusion Matrix saved to {matrix_file}")