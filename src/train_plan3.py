import zipfile
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pack_padded_sequence
import json
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, precision_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. CONFIGURATION (PLAN 3: DEEP LSTM) ---
HIDDEN_SIZE = 64
EMBED_DIM = 64
NUM_CLASSES = 5
NUM_EPOCHS = 30
BATCH_SIZE = 16
LEARNING_RATE = 0.001
# NEW: 2 Layers and Aggressive Dropout
NUM_LAYERS = 2          
DROPOUT_RATE = 0.5      
WEIGHT_DECAY = 1e-4     

# Path Setup
CURRENT_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = CURRENT_DIR.parent 

# --- 2. DATA LOADING (ZIP Loading & Ignore Consensus) ---
def load_data_from_zip(project_root):
    data = []
    vocab = set()
    
    zip_files = list(project_root.glob("*.zip"))
    
    if not zip_files:
        print(f"No .zip file found in {project_root}.")
        return data, list(vocab)
        
    target_zip = zip_files[0]
    print(f"Loading data from: {target_zip.name}")

    with zipfile.ZipFile(target_zip, 'r') as z:
        for filename in z.namelist():
            if not filename.endswith('.json'): continue
            if "__MACOSX" in filename: continue
            
            # CRITICAL: Ignore 'consensus' folder
            if "consensus/" in filename: continue

            try:
                with z.open(filename) as f:
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
                            if len(tokens) == 0: continue
                            
                            data.append({"text": tokens, "label": label})
                            for w in tokens: vocab.add(w)
                        except: continue
            except Exception: continue

    return data, list(vocab)

raw_data, vocab_list = load_data_from_zip(PROJECT_ROOT)
print(f"Total samples loaded: {len(raw_data)}")

# --- 3. PREPROCESSING ---
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

if len(full_dataset) > 0:
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
else:
    print("Error: Dataset is empty.")
    exit()

# --- 4. MODEL ARCHITECTURE (PLAN 3: DEEP & BIDIRECTIONAL) ---
class LegalModelPlan3(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size, num_classes, dropout_rate, num_layers):
        super(LegalModelPlan3, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        # num_layers=2, bidirectional=True, dropout between layers
        self.lstm = nn.LSTM(embed_dim, 
                            hidden_size, 
                            num_layers=num_layers, 
                            batch_first=True, 
                            bidirectional=True,
                            dropout=dropout_rate if num_layers > 1 else 0)
        
        self.dropout = nn.Dropout(dropout_rate)
        
        # Input size is hidden_size * 2 due to bidirectionality
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x, lengths):
        embedded = self.embedding(x)
        packed_embedded = pack_padded_sequence(embedded, lengths.cpu(), batch_first=True)
        
        _, (hidden_state, _) = self.lstm(packed_embedded)
        
        # For Deep LSTM, hidden_state shape: [num_layers * num_directions, batch, hidden]
        # We need the last layer's forward and backward states
        hidden_forward = hidden_state[-2]
        hidden_backward = hidden_state[-1]
        
        cat_hidden = torch.cat((hidden_forward, hidden_backward), dim=1)
        
        out = self.dropout(cat_hidden)
        out = self.fc(out)
        return out

# --- 5. TRAINING LOOP ---
model = LegalModelPlan3(vocab_size, EMBED_DIM, HIDDEN_SIZE, NUM_CLASSES, DROPOUT_RATE, NUM_LAYERS)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

train_losses = []
val_accuracies = []

print(f"\n--- Plan 3 Started: Deep Bi-LSTM (Layers={NUM_LAYERS}, Hidden={HIDDEN_SIZE}) ---")

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

# --- 6. EVALUATION AND VISUALIZATION ---
print("\n--- Final Evaluation ---")

# 1. Learning Curve
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(val_accuracies, label='Validation Accuracy', color='red')
plt.title('Validation Accuracy')
plt.xlabel('Epoch')
plt.legend()
plt.tight_layout()
plt.savefig('learning_curve_plan3.png')
print("Learning curve saved: learning_curve_plan3.png")

# 2. Detailed Metrics
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

print(f"\nFinal Accuracy: {accuracy:.4f}")
print(f"Final F1 Score: {f1:.4f}")

unique_labels = sorted(list(set(all_labels)))
target_names = [f"Class {i+1}" for i in unique_labels]
print("\nClassification Report:")
print(classification_report(all_labels, all_preds, labels=unique_labels, target_names=target_names, zero_division=0))

# --- 7. SAVING METRICS TO FILE ---
metrics_file = "plan3_metrics.txt"
with open(metrics_file, "w") as f:
    f.write("Plan 3 Model Evaluation Metrics\n")
    f.write("===============================\n")
    f.write(f"Accuracy:  {accuracy:.4f}\n")
    f.write(f"Precision: {precision:.4f}\n")
    f.write(f"Recall:    {recall:.4f}\n")
    f.write(f"F1 Score:  {f1:.4f}\n")

print(f"Metrics saved to {metrics_file}")

# 3. Confusion Matrix
plt.figure(figsize=(8, 6))
cm = confusion_matrix(all_labels, all_preds)
sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', xticklabels=target_names, yticklabels=target_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix - Plan 3')
plt.savefig('confusion_matrix_plan3.png')
print("Confusion Matrix saved: confusion_matrix_plan3.png")