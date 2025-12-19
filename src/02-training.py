import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Dataset
from torch.nn.utils.rnn import pack_padded_sequence
import numpy as np
import sys
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Import Config & Utils
try:
    import config as cfg
    from utils import setup_logger, count_parameters
except ImportError:
    sys.path.append(str(Path(__file__).parent))
    import config as cfg
    from utils import setup_logger, count_parameters

logger = setup_logger("Training")
cfg.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
cfg.LOG_DIR.mkdir(parents=True, exist_ok=True)

# Set Seeds
torch.manual_seed(cfg.RANDOM_SEED)
np.random.seed(cfg.RANDOM_SEED)

def log_configuration():
    logger.info("="*30)
    logger.info("CONFIGURATION START (Loaded from src/config.py)")
    logger.info("="*30)
    logger.info(f"Device: {cfg.DEVICE}")
    logger.info(f"Batch Size: {cfg.BATCH_SIZE}")
    logger.info("-" * 20)
    logger.info(f"Baseline LSTM: Epochs={cfg.LSTM_EPOCHS}, LR={cfg.LSTM_LR}, Hidden={cfg.LSTM_HIDDEN_SIZE}")
    logger.info(f"Improved CNN:  Epochs={cfg.CNN_EPOCHS}, LR={cfg.CNN_LR}, Filters={cfg.CNN_NUM_FILTERS}")
    logger.info("="*30 + "\n")

# --- DATASETS ---
class LegalDataset(Dataset):
    def __init__(self, data, word2idx):
        self.data, self.word2idx = data, word2idx
    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        item = self.data[idx]
        tokens = [self.word2idx.get(w, 0) for w in item['text']]
        return torch.tensor(tokens, dtype=torch.long), torch.tensor(item['label']-1, dtype=torch.long)

def collate_fn(batch):
    batch.sort(key=lambda x: len(x[0]), reverse=True)
    seqs, labels = zip(*batch)
    lengths = [len(s) for s in seqs]
    padded = torch.zeros(len(seqs), max(lengths), dtype=torch.long)
    for i, s in enumerate(seqs): padded[i, :len(s)] = s
    return padded, torch.stack(labels), torch.tensor(lengths)

# --- MODELS ---
class BaselineLSTM(nn.Module):
    def __init__(self, vs):
        super().__init__()
        self.embedding = nn.Embedding(vs, cfg.LSTM_EMBED_DIM, padding_idx=0)
        self.lstm = nn.LSTM(cfg.LSTM_EMBED_DIM, cfg.LSTM_HIDDEN_SIZE, batch_first=True)
        self.fc = nn.Linear(cfg.LSTM_HIDDEN_SIZE, cfg.NUM_CLASSES)
    def forward(self, x, l):
        _, (h, _) = self.lstm(pack_padded_sequence(self.embedding(x), l.cpu(), batch_first=True))
        return self.fc(h[-1])

class ImprovedCNN(nn.Module):
    def __init__(self, vs):
        super().__init__()
        self.embedding = nn.Embedding(vs, cfg.CNN_EMBED_DIM, padding_idx=0)
        self.convs = nn.ModuleList([nn.Conv1d(cfg.CNN_EMBED_DIM, cfg.CNN_NUM_FILTERS, k) for k in cfg.CNN_FILTER_SIZES])
        self.dropout = nn.Dropout(cfg.CNN_DROPOUT)
        self.fc = nn.Linear(cfg.CNN_NUM_FILTERS * len(cfg.CNN_FILTER_SIZES), cfg.NUM_CLASSES)
    def forward(self, x, l):
        e = self.embedding(x).permute(0, 2, 1)
        c = [torch.nn.functional.max_pool1d(torch.nn.functional.relu(cv(e)), cv(e).shape[2]).squeeze(2) for cv in self.convs]
        return self.fc(self.dropout(torch.cat(c, 1)))

# --- TRAINING ---
def train_model(model, train_loader, test_loader, epochs, lr, name, patience=None):
    logger.info(f"\nStarting training for: {name}")
    logger.info(str(model))
    model = model.to(cfg.DEVICE)
    crit = nn.CrossEntropyLoss()
    opt = optim.Adam(model.parameters(), lr=lr)
    
    best_loss, counter = float('inf'), 0
    save_path = cfg.OUTPUT_DIR / f"{name.lower()}_best.pt"

    for epoch in range(epochs):
        model.train()
        t_loss = 0
        for X, y, l in train_loader:
            opt.zero_grad()
            loss = crit(model(X.to(cfg.DEVICE), l), y.to(cfg.DEVICE))
            loss.backward()
            opt.step()
            t_loss += loss.item()
            
        model.eval()
        v_loss, corr, tot = 0, 0, 0
        with torch.no_grad():
            for X, y, l in test_loader:
                X, y = X.to(cfg.DEVICE), y.to(cfg.DEVICE)
                out = model(X, l)
                v_loss += crit(out, y).item()
                corr += (out.argmax(1) == y).sum().item()
                tot += y.size(0)
        
        avg_v = v_loss/len(test_loader)
        logger.info(f"[{name}] Ep {epoch+1}/{epochs} | Train: {t_loss/len(train_loader):.4f} | Val: {avg_v:.4f} | Acc: {corr/tot*100:.2f}%")
        
        if patience:
            if avg_v < best_loss:
                best_loss, counter = avg_v, 0
                torch.save(model.state_dict(), save_path)
            else:
                counter += 1
                if counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    model.load_state_dict(torch.load(save_path))
                    break
    
    if not patience: torch.save(model.state_dict(), save_path)
    logger.info(f"Model saved to {save_path}")
    return model

def evaluate_and_save(model, loader, name):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for X, y, l in loader:
            y_true.extend(y.numpy())
            y_pred.extend(model(X.to(cfg.DEVICE), l).argmax(1).cpu().numpy())
    
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    logger.info(f"Final {name} | Acc: {acc:.4f} | F1: {f1:.4f}")
    
    with open(cfg.OUTPUT_DIR / f"{name.lower()}_metrics.txt", "w") as f:
        f.write(classification_report(y_true, y_pred, zero_division=0))
    
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.savefig(cfg.OUTPUT_DIR / f"{name.lower()}_confusion_matrix.png")
    plt.close()

def main():
    log_configuration()
    if not cfg.PROCESSED_DATA_PATH.exists(): return
    saved = torch.load(cfg.PROCESSED_DATA_PATH)
    data, w2i = saved['data'], saved['word2idx']
    vocab_size = len(w2i) + 1
    
    ds = LegalDataset(data, w2i)
    tr_len = int(0.8 * len(ds))
    train_ds, test_ds = random_split(ds, [tr_len, len(ds)-tr_len])
    
    tr_load = DataLoader(train_ds, cfg.BATCH_SIZE, True, collate_fn=collate_fn)
    te_load = DataLoader(test_ds, cfg.BATCH_SIZE, False, collate_fn=collate_fn)

    # Train Baseline
    bl = train_model(BaselineLSTM(vocab_size), tr_load, te_load, cfg.LSTM_EPOCHS, cfg.LSTM_LR, "Baseline_LSTM")
    evaluate_and_save(bl, te_load, "Baseline_LSTM")

    # Train Improved
    imp = train_model(ImprovedCNN(vocab_size), tr_load, te_load, cfg.CNN_EPOCHS, cfg.CNN_LR, "Improved_CNN", cfg.CNN_PATIENCE)
    evaluate_and_save(imp, te_load, "Improved_CNN")

if __name__ == "__main__":
    main()