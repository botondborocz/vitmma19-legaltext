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

# --------------------------------------------------
# CONFIG / LOGGER
# --------------------------------------------------
try:
    import config as cfg
    from utils import setup_logger
except ImportError:
    sys.path.append(str(Path(__file__).parent))
    import config as cfg
    from utils import setup_logger

logger = setup_logger("Evaluation")

# FORCE CPU
cfg.DEVICE = torch.device("cpu")
torch.set_num_threads(1)

# --------------------------------------------------
# DATASET
# --------------------------------------------------
class LegalDataset(Dataset):
    def __init__(self, data, word2idx):
        self.data = data
        self.word2idx = word2idx

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        tokens = [self.word2idx.get(w, 0) for w in self.data[idx]["text"]]
        label = self.data[idx]["label"] - 1
        return torch.tensor(tokens, dtype=torch.long), torch.tensor(label, dtype=torch.long)

# --------------------------------------------------
# COLLATE
# --------------------------------------------------
def collate_fn(batch):
    batch = [(x, y) for x, y in batch if len(x) > 0]
    batch.sort(key=lambda x: len(x[0]), reverse=True)

    seqs, labels = zip(*batch)
    lengths = torch.tensor([len(s) for s in seqs], dtype=torch.long)

    padded = torch.zeros(len(seqs), max(lengths), dtype=torch.long)
    for i, s in enumerate(seqs):
        padded[i, :len(s)] = s

    return padded, torch.stack(labels), lengths

# --------------------------------------------------
# MODELS
# --------------------------------------------------
class BaselineLSTM(nn.Module):
    def __init__(self, vs):
        super().__init__()
        self.embedding = nn.Embedding(vs, cfg.LSTM_EMBED_DIM, padding_idx=0)
        self.lstm = nn.LSTM(
            cfg.LSTM_EMBED_DIM,
            cfg.LSTM_HIDDEN_SIZE,
            batch_first=True
        )
        self.fc = nn.Linear(cfg.LSTM_HIDDEN_SIZE, cfg.NUM_CLASSES)

    def forward(self, x, l):
        emb = self.embedding(x)
        packed = pack_padded_sequence(
            emb,
            l.cpu(),
            batch_first=True,
            enforce_sorted=False  # 🔥 CPU-safe & faster
        )
        _, (h, _) = self.lstm(packed)
        return self.fc(h[-1])

class ImprovedCNN(nn.Module):
    def __init__(self, vs):
        super().__init__()
        self.embedding = nn.Embedding(vs, cfg.CNN_EMBED_DIM, padding_idx=0)
        self.convs = nn.ModuleList(
            [nn.Conv1d(cfg.CNN_EMBED_DIM, cfg.CNN_NUM_FILTERS, k)
             for k in cfg.CNN_FILTER_SIZES]
        )
        self.fc = nn.Linear(
            cfg.CNN_NUM_FILTERS * len(cfg.CNN_FILTER_SIZES),
            cfg.NUM_CLASSES
        )

    def forward(self, x, l):
        e = self.embedding(x).permute(0, 2, 1)
        c = [
            F.max_pool1d(F.relu(conv(e)), conv(e).shape[2]).squeeze(2)
            for conv in self.convs
        ]
        return self.fc(F.dropout(torch.cat(c, dim=1), cfg.CNN_DROPOUT))

# --------------------------------------------------
# MAIN
# --------------------------------------------------
def main():
    logger.info("Starting evaluation...")

    if not cfg.PROCESSED_DATA_PATH.exists():
        logger.error("Processed data not found.")
        return

    saved = torch.load(cfg.PROCESSED_DATA_PATH, map_location="cpu")
    vocab_size = len(saved["word2idx"]) + 1

    dataset = LegalDataset(saved["data"], saved["word2idx"])
    _, test_ds = random_split(
        dataset,
        [int(0.8 * len(dataset)), len(dataset) - int(0.8 * len(dataset))]
    )

    loader = DataLoader(
        test_ds,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )

    results = {}

    for name, path, model in [
        ("Baseline", cfg.BASELINE_MODEL_PATH, BaselineLSTM(vocab_size)),
        ("Improved", cfg.IMPROVED_MODEL_PATH, ImprovedCNN(vocab_size))
    ]:
        if not path.exists():
            logger.warning(f"{name} model not found.")
            continue

        logger.info(f"Evaluating {name} model...")
        model.load_state_dict(torch.load(path, map_location="cpu"))
        model.to(cfg.DEVICE).eval()

        y_true_chunks = []
        y_pred_chunks = []

        with torch.no_grad():
            for X, y, l in loader:
                X = X.to(cfg.DEVICE)

                logits = model(X, l)
                preds = logits.argmax(dim=1)

                y_true_chunks.append(y)
                y_pred_chunks.append(preds.cpu())

        # 🔥 FAST: concatenate once
        y_true = torch.cat(y_true_chunks).numpy()
        y_pred = torch.cat(y_pred_chunks).numpy()

        acc = accuracy_score(y_true, y_pred)
        f1 = precision_recall_fscore_support(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0
        )[2]

        results[name] = {"Accuracy": acc, "F1 Score": f1}

    logger.info("\n" + pd.DataFrame(results).T.to_string())

if __name__ == "__main__":
    main()
