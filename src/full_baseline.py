import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.nn.utils.rnn import pack_padded_sequence
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, classification_report
import seaborn as sns
from matplotlib import pyplot as plt

# --- IMPORT FROM SHARED MODULE ---
# Assuming data_loader.py is in the same directory (src)
try:
    from data_loader import load_data_from_zip, LegalDataset, collate_fn
except ImportError:
    import sys
    sys.path.append(str(Path(__file__).parent))
    from data_loader import load_data_from_zip, LegalDataset, collate_fn

# --- 1. CONFIGURATION ---
HIDDEN_SIZE = 32        
EMBED_DIM = 32          
NUM_CLASSES = 5
NUM_EPOCHS = 20         
BATCH_SIZE = 16
LEARNING_RATE = 0.001
DROPOUT_RATE = 0.5      
WEIGHT_DECAY = 1e-4     

# Path Setup
CURRENT_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = CURRENT_DIR.parent
OUTPUT_DIR = PROJECT_ROOT / "output"

# Create output folder if needed
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --- 2. DATA LOADING (VIA SHARED MODULE) ---
raw_data, vocab_list = load_data_from_zip(PROJECT_ROOT)
print(f"Total samples loaded: {len(raw_data)}")

# --- 3. PREPROCESSING ---
word2idx = {w: i+1 for i, w in enumerate(vocab_list)}
word2idx["<PAD>"] = 0
vocab_size = len(word2idx)

# Use the imported Dataset class
full_dataset = LegalDataset(raw_data, word2idx)

if len(full_dataset) > 0:
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])
    
    # Use the imported collate_fn
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
else:
    print("Error: Dataset is empty.")
    exit()

# --- 4. MODEL ARCHITECTURE ---
class LegalClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size, num_classes, dropout_rate):
        super(LegalClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(hidden_size, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, num_classes)

    def forward(self, x, lengths):
        embedded = self.embedding(x)
        packed_embedded = pack_padded_sequence(embedded, lengths.cpu(), batch_first=True)
        _, (hidden_state, _) = self.lstm(packed_embedded)
        last_hidden = hidden_state[-1]
        out = self.dropout(last_hidden)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# --- 5. TRAINING LOOP ---
model = LegalClassifier(vocab_size, EMBED_DIM, HIDDEN_SIZE, NUM_CLASSES, DROPOUT_RATE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

print(f"\nStarting training on {train_size} samples, validating on {test_size} samples...")

for epoch in range(NUM_EPOCHS):
    model.train()
    total_loss = 0
    correct_train = 0
    total_train = 0
    
    for X_batch, y_batch, lengths in train_loader:
        optimizer.zero_grad()
        outputs = model(X_batch, lengths)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
        _, predicted = torch.max(outputs, 1)
        correct_train += (predicted == y_batch).sum().item()
        total_train += y_batch.size(0)
    
    avg_loss = total_loss / len(train_loader)
    train_acc = correct_train / total_train * 100
    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] | Train Loss: {avg_loss:.4f} | Train Acc: {train_acc:.2f}%")

# --- 6. EVALUATION ---
print("\n--- Final Evaluation on Test Set ---")
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
precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)

print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")

# --- 7. SAVING METRICS TO FILE ---
metrics_file = OUTPUT_DIR / "baseline_metrics.txt"
with open(metrics_file, "w") as f:
    f.write("Baseline Model Evaluation Metrics\n")
    f.write("=================================\n")
    f.write(f"Accuracy:  {accuracy:.4f}\n")
    f.write(f"Precision: {precision:.4f}\n")
    f.write(f"Recall:    {recall:.4f}\n")
    f.write(f"F1 Score:  {f1:.4f}\n")

print(f"\nMetrics saved to {metrics_file}")

unique_labels = sorted(list(set(all_labels)))
target_names = [f"Class {i+1}" for i in unique_labels]
print("\n" + classification_report(all_labels, all_preds, labels=unique_labels, target_names=target_names, zero_division=0))

# --- 8. VISUALIZATION (SAVING TO PNG) ---
print("\n--- Generating Confusion Matrix ---")

cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=target_names, 
            yticklabels=target_names)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix - Legal Text Classification')

matrix_file = OUTPUT_DIR / "confusion_matrix_baseline.png"
plt.savefig(matrix_file)
print(f"Confusion matrix saved to {matrix_file}")