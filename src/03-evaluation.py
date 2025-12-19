import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, Dataset
from torch.nn.utils.rnn import pack_padded_sequence
import numpy as np
import pandas as pd
import sys
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

try:
    import config as cfg
    from utils import setup_logger
except ImportError:
    sys.path.append(str(Path(__file__).parent))
    import config as cfg
    from utils import setup_logger

logger = setup_logger("Evaluation")

# --- REUSED CLASSES (Ideally move these to a models.py file) ---
class LegalDataset(Dataset):
    def __init__(self, data, word2idx): self.data, self.word2idx = data, word2idx
    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        return torch.tensor([self.word2idx.get(w,0) for w in self.data[idx]['text']]), torch.tensor(self.data[idx]['label']-1)

def collate_fn(batch):
    batch.sort(key=lambda x: len(x[0]), reverse=True)
    seqs, labels = zip(*batch)
    padded = torch.zeros(len(seqs), max(len(s) for s in seqs), dtype=torch.long)
    for i, s in enumerate(seqs): padded[i, :len(s)] = s
    return padded, torch.stack(labels), torch.tensor([len(s) for s in seqs])

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
        self.fc = nn.Linear(cfg.CNN_NUM_FILTERS * len(cfg.CNN_FILTER_SIZES), cfg.NUM_CLASSES)
    def forward(self, x, l):
        e = self.embedding(x).permute(0, 2, 1)
        c = [F.max_pool1d(F.relu(cv(e)), cv(e).shape[2]).squeeze(2) for cv in self.convs]
        return self.fc(F.dropout(torch.cat(c, 1), cfg.CNN_DROPOUT))

def main():
    if not cfg.PROCESSED_DATA_PATH.exists(): return
    saved = torch.load(cfg.PROCESSED_DATA_PATH)
    vs = len(saved['word2idx']) + 1
    
    ds = LegalDataset(saved['data'], saved['word2idx'])
    _, test_ds = random_split(ds, [int(0.8*len(ds)), len(ds)-int(0.8*len(ds))])
    loader = DataLoader(test_ds, cfg.BATCH_SIZE, False, collate_fn=collate_fn)

    res = {}
    for name, path, model in [
        ("Baseline", cfg.BASELINE_MODEL_PATH, BaselineLSTM(vs)),
        ("Improved", cfg.IMPROVED_MODEL_PATH, ImprovedCNN(vs))
    ]:
        if path.exists():
            model.load_state_dict(torch.load(path, map_location=cfg.DEVICE))
            model.to(cfg.DEVICE).eval()
            y_t, y_p = [], []
            with torch.no_grad():
                for X, y, l in loader:
                    y_t.extend(y.numpy())
                    y_p.extend(model(X.to(cfg.DEVICE), l).argmax(1).cpu().numpy())
            
            acc = accuracy_score(y_t, y_p)
            f1 = precision_recall_fscore_support(y_t, y_p, average='weighted', zero_division=0)[2]
            res[name] = {"Accuracy": acc, "F1 Score": f1}
        else:
            logger.warning(f"{name} model not found.")

    logger.info("\n" + pd.DataFrame(res).T.to_string())

if __name__ == "__main__":
    main()