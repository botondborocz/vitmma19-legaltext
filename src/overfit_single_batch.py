import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence
# Feltételezzük, hogy a baseline.py-ban lévő adatbetöltő függvényeid elérhetőek
# Ha külön fájlban vannak, importáld őket, pl: from baseline import load_labelstudio_data, encode_batch
import json
import glob
from pathlib import Path

# --- KONFIGURÁCIÓ (STEP 1: Legkisebb modell & STEP 2: Single Batch) ---
# A dokumentum szerint keressük a legkisebb modellt [cite: 23, 39]
HIDDEN_SIZE = 8       # Extrém kicsi
EMBED_DIM = 8         # Extrém kicsi
NUM_CLASSES = 5
NUM_EPOCHS = 200      # Addig megyünk, amíg a loss < 0.001 
BATCH_SIZE = 32       # Egyetlen batch 
LEARNING_RATE = 0.01

# Elérési utak (igazítsd a saját projektedhez)
CURRENT_DIR = Path(__file__).parent.absolute() if '__file__' in locals() else Path(".").absolute()
DATA_DIR = CURRENT_DIR.parent / "consensus"

# --- 1. ADATBETÖLTÉS (Csak 32 minta kell!) ---
# Ugyanaz a logika, mint a baseline-nál, de limitáljuk 32-re a betöltést
def get_single_batch_data(data_dir, target_count=32):
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
                    text = task.get('data', {}).get('text', "")
                    if not text: continue
                    try:
                        choice_str = task['annotations'][0]['result'][0]['value']['choices'][0]
                        label = int(choice_str.split('-')[0])
                        tokens = text.split()
                        data.append({"text": tokens, "label": label})
                        for w in tokens: vocab.add(w)
                        count += 1
                    except: continue
        except: continue
    return data, list(vocab)

# Adat előkészítése
raw_data, vocab_list = get_single_batch_data(DATA_DIR, target_count=32)
print(f"Betöltött minták száma: {len(raw_data)} (Cél: 32)")

word2idx = {w: i+1 for i, w in enumerate(vocab_list)}
word2idx["<PAD>"] = 0
vocab_size = len(word2idx)

# Batch enkódolása (ugyanaz a logika, mint korábban)
def encode_batch(data_batch, word2idx):
    texts = [d['text'] for d in data_batch] # már tokenizálva van
    labels = [d['label'] - 1 for d in data_batch]
    lengths = [len(t) for t in texts]
    max_len = max(lengths)
    
    padded_input = []
    for tokens in texts:
        indices = [word2idx.get(t, 0) for t in tokens]
        padding = [0] * (max_len - len(indices))
        padded_input.append(indices + padding)
        
    return (torch.tensor(padded_input, dtype=torch.long), 
            torch.tensor(labels, dtype=torch.long), 
            torch.tensor(lengths, dtype=torch.long))

X_batch, y_batch, lengths = encode_batch(raw_data, word2idx)

# --- 2. MODELL ARCHITEKTÚRA (TINY LSTM) ---
class TinyLegalClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size, num_classes):
        super(TinyLegalClassifier, self).__init__()
        # Nincs dropout, nincs extra réteg 
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes) # Csak egy kimeneti réteg

    def forward(self, x, lengths):
        embedded = self.embedding(x)
        packed_embedded = pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (hidden_state, _) = self.lstm(packed_embedded)
        last_hidden = hidden_state[-1]
        out = self.fc(last_hidden)
        return out

# --- 3. TANÍTÁS (OVERFITTING) ---
model = TinyLegalClassifier(vocab_size, EMBED_DIM, HIDDEN_SIZE, NUM_CLASSES)
criterion = nn.CrossEntropyLoss()
# Kikapcsoljuk a weight decay-t! 
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=0.0)

print("\n--- 2. Lépés: Overfit on a Single Batch ---")
print("Cél: Training Loss < 0.001 és 100% pontosság [cite: 48, 49]")

for epoch in range(NUM_EPOCHS):
    model.train()
    optimizer.zero_grad()
    
    outputs = model(X_batch, lengths)
    loss = criterion(outputs, y_batch)
    
    loss.backward()
    optimizer.step()
    
    # Kiértékelés minden 10. epochban
    if (epoch + 1) % 10 == 0:
        _, predicted = torch.max(outputs, 1)
        correct = (predicted == y_batch).sum().item()
        acc = correct / len(y_batch) * 100
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] | Loss: {loss.item():.6f} | Acc: {acc:.2f}%")
        
        # Stop condition 
        if loss.item() < 0.001 and acc == 100.0:
            print(f"\nSIKER! A modell túltanulta a batch-et a {epoch+1}. epochban.")
            break

if loss.item() > 0.01:
    print("\nFIGYELEM: Nem sikerült tökéletesen túltanulni. Ellenőrizd az adatot vagy a kódot! ")