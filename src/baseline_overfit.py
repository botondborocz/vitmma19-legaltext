import torch
import torch.nn as nn
import torch.optim as optim
import json
import glob
from pathlib import Path

# --- 1. CONFIGURATION ---
# Increased dimensions slightly to help memorization
HIDDEN_SIZE = 64 
EMBED_DIM = 64   
NUM_CLASSES = 5
NUM_EPOCHS = 150 # Increased epochs slightly
BATCH_SIZE = 32
LEARNING_RATE = 0.01

CURRENT_DIR = Path(__file__).parent.absolute()
DATA_DIR = CURRENT_DIR.parent / "consensus"

# --- 2. DATA LOADING (Same as before) ---
def load_labelstudio_data(data_dir, target_count=32):
    data = []
    vocab = set()
    search_path = data_dir / "**" / "*.json"
    json_files = glob.glob(str(search_path), recursive=True)
    
    count = 0
    for file_path in json_files:
        if count >= target_count: break
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = json.load(f)
                if isinstance(content, dict): content = [content]
                for task in content:
                    if count >= target_count: break
                    try:
                        text = task.get('data', {}).get('text', "")
                        if not text: continue
                        
                        # Label extraction
                        annotations = task.get('annotations', [])
                        if not annotations: continue
                        result = annotations[0].get('result', [])
                        if not result: continue
                        choice_str = result[0]['value']['choices'][0]
                        label = int(choice_str.split('-')[0])
                        
                        data.append({"text": text, "label": label})
                        for w in text.split(): vocab.add(w)
                        count += 1
                    except: continue
        except: continue
    return data, list(vocab)

raw_data, vocab_list = load_labelstudio_data(DATA_DIR)

if not raw_data:
    raise ValueError("No data found!")

# --- 3. PREPROCESSING (UPDATED) ---
word2idx = {w: i+1 for i, w in enumerate(vocab_list)}
word2idx["<PAD>"] = 0
vocab_size = len(word2idx)

def encode_batch(data_batch, word2idx):
    texts = [d['text'].split() for d in data_batch]
    labels = [d['label'] - 1 for d in data_batch]
    
    # Store the actual lengths of each sentence
    lengths = [len(t) for t in texts]
    max_len = max(lengths)
    
    padded_input = []
    for tokens in texts:
        indices = [word2idx.get(t, 0) for t in tokens]
        padding = [0] * (max_len - len(indices))
        padded_input.append(indices + padding)
        
    return (torch.tensor(padded_input, dtype=torch.long), 
            torch.tensor(labels, dtype=torch.long), 
            torch.tensor(lengths, dtype=torch.long)) # Return lengths!

X_train, y_train, train_lengths = encode_batch(raw_data, word2idx)

# --- 4. MODEL ARCHITECTURE (FIXED) ---
class LegalClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size, num_classes):
        super(LegalClassifier, self).__init__()
        
        # Padding_idx=0 tells PyTorch this index is empty space
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        self.lstm = nn.LSTM(embed_dim, hidden_size, batch_first=True)
        
        self.fc1 = nn.Linear(hidden_size, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, num_classes)

    def forward(self, x, lengths):
        # x: [batch, seq_len]
        embedded = self.embedding(x)
        
        # --- THE FIX: Pack the sequence ---
        # This tells LSTM to ignore the zeros at the end
        packed_embedded = nn.utils.rnn.pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        
        # LSTM returns packed output and the final hidden state
        # hidden_state is (h_n, c_n). h_n is the last *valid* hidden state.
        _, (hidden_state, _) = self.lstm(packed_embedded)
        
        # hidden_state shape: [num_layers, batch, hidden_size]
        last_hidden = hidden_state[-1] 
        
        out = self.fc1(last_hidden)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# --- 5. TRAINING LOOP ---
model = LegalClassifier(vocab_size, EMBED_DIM, HIDDEN_SIZE, NUM_CLASSES)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

print(f"\nStarting training on {len(raw_data)} samples. Class distribution:")
labels_list = [d['label'] for d in raw_data]
print(f"Labels: {labels_list}") 

for epoch in range(NUM_EPOCHS):
    model.train()
    optimizer.zero_grad()
    
    # Pass lengths to the model
    outputs = model(X_train, train_lengths)
    loss = criterion(outputs, y_train)
    
    loss.backward()
    optimizer.step()
    
    _, predicted = torch.max(outputs, 1)
    correct = (predicted == y_train).sum().item()
    accuracy = correct / len(y_train) * 100
    
    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] | Loss: {loss.item():.4f} | Accuracy: {accuracy:.2f}%")

print("\nTraining finished.")

# Final check
model.eval()
test_out = model(X_train, train_lengths)
_, pred = torch.max(test_out, 1)
print(f"\nTarget Labels:    {y_train.tolist()}")
print(f"Predicted Labels: {pred.tolist()}")