import os
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

# --------------------------------------------------
# FORCE CPU (FIX #3)
# --------------------------------------------------
torch.set_num_threads(1)
DEVICE = torch.device("cpu")

# Import Config & Utils
try:
    import config as cfg
    from utils import setup_logger, count_parameters
except ImportError:
    sys.path.append(str(Path(__file__).parent))
    import config as cfg
    from utils import setup_logger, count_parameters

cfg.DEVICE = DEVICE  # override any accidental CUDA usage

logger = setup_logger("Training")
OUTPUT_DIR = cfg.OUTPUT_DIR
os.makedirs(OUTPUT_DIR, exist_ok=True)
LOG_DIR = cfg.LOG_DIR
os.makedirs(LOG_DIR, exist_ok=True)

# Set Seeds
torch.manual_seed(cfg.RANDOM_SEED)
np.random.seed(cfg.RANDOM_SEED)

def log_configuration():
    logger.info("=" * 30)
    logger.info("CONFIGURATION START (CPU ONLY)")
    logger.info("=" * 30)
    logger.info(f"Device: {cfg.DEVICE}")
    logger.info(f"Batch Size: {cfg.BATCH_SIZE}")
    logger.info("-" * 20)
    logger.info(
        f"Baseline LSTM: Epochs={cfg.LSTM_EPOCHS}, "
        f"LR={cfg.LSTM_LR}, Hidden={cfg.LSTM_HIDDEN_SIZE}"
    )
    logger.info(
        f"Improved CNN:  Epochs={cfg.CNN_EPOCHS}, "
        f"LR={cfg.CNN_LR}, Filters={cfg.CNN_NUM_FILTERS}"
    )
    logger.info("=" * 30 + "\n")

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
        item = self.data[idx]
        tokens = [self.word2idx.get(w, 0) for w in item["text"]]
        return (
            torch.tensor(tokens, dtype=torch.long),
            torch.tensor(item["label"] - 1, dtype=torch.long),
        )

# --------------------------------------------------
# COLLATE (FIX #2: remove zero-length sequences)
# --------------------------------------------------
def collate_fn(batch):
    batch = [(x, y) for x, y in batch if len(x) > 0]

    if len(batch) == 0:
        return None

    batch.sort(key=lambda x: len(x[0]), reverse=True)
    seqs, labels = zip(*batch)

    lengths = torch.tensor([len(s) for s in seqs], dtype=torch.long)

    padded = torch.zeros(len(seqs), max(lengths), dtype=torch.long)
    for i, s in enumerate(seqs):
        padded[i, : len(s)] = s

    return padded, torch.stack(labels), lengths

# --------------------------------------------------
# MODELS
# --------------------------------------------------
class BaselineLSTM(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(
            vocab_size, cfg.LSTM_EMBED_DIM, padding_idx=0
        )
        self.lstm = nn.LSTM(
            cfg.LSTM_EMBED_DIM,
            cfg.LSTM_HIDDEN_SIZE,
            batch_first=True,
        )
        self.fc = nn.Linear(cfg.LSTM_HIDDEN_SIZE, cfg.NUM_CLASSES)

    def forward(self, x, lengths):
        emb = self.embedding(x)
        packed = pack_padded_sequence(
            emb,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,  # FIX #1
        )
        _, (h, _) = self.lstm(packed)
        return self.fc(h[-1])

class ImprovedCNN(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(
            vocab_size, cfg.CNN_EMBED_DIM, padding_idx=0
        )
        self.convs = nn.ModuleList(
            [
                nn.Conv1d(
                    cfg.CNN_EMBED_DIM,
                    cfg.CNN_NUM_FILTERS,
                    k,
                )
                for k in cfg.CNN_FILTER_SIZES
            ]
        )
        self.dropout = nn.Dropout(cfg.CNN_DROPOUT)
        self.fc = nn.Linear(
            cfg.CNN_NUM_FILTERS * len(cfg.CNN_FILTER_SIZES),
            cfg.NUM_CLASSES,
        )

    def forward(self, x, lengths):
        e = self.embedding(x).permute(0, 2, 1)
        c = [
            torch.nn.functional.max_pool1d(
                torch.nn.functional.relu(conv(e)),
                conv(e).shape[2],
            ).squeeze(2)
            for conv in self.convs
        ]
        return self.fc(self.dropout(torch.cat(c, dim=1)))

# --------------------------------------------------
# TRAINING
# --------------------------------------------------
def train_model(
    model, train_loader, test_loader, epochs, lr, name, patience=None
):
    logger.info(f"\nStarting training for: {name}")
    logger.info(str(model))

    model.to(cfg.DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_loss = float("inf")
    counter = 0
    save_path = cfg.OUTPUT_DIR / f"{name.lower()}_best.pt"

    for epoch in range(epochs):
        logger.info(
            f"Epoch {epoch + 1}/{epochs} - "
            f"Parameters: {count_parameters(model)}"
        )

        model.train()
        logger.info("Training...")
        train_loss = 0.0

        for batch in train_loader:
            if batch is None:
                continue

            X, y, l = batch
            X = X.to(cfg.DEVICE)
            y = y.to(cfg.DEVICE)

            optimizer.zero_grad()
            loss = criterion(model(X, l), y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        logger.info("Validating...")
        model.eval()

        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in test_loader:
                if batch is None:
                    continue

                X, y, l = batch
                X = X.to(cfg.DEVICE)
                y = y.to(cfg.DEVICE)

                out = model(X, l)
                val_loss += criterion(out, y).item()
                correct += (out.argmax(1) == y).sum().item()
                total += y.size(0)

        avg_val = val_loss / max(1, len(test_loader))
        acc = 100 * correct / max(1, total)

        logger.info(
            f"[{name}] Ep {epoch + 1}/{epochs} | "
            f"Train: {train_loss / max(1, len(train_loader)):.4f} | "
            f"Val: {avg_val:.4f} | Acc: {acc:.2f}%"
        )

        if patience:
            if avg_val < best_loss:
                best_loss = avg_val
                counter = 0
                torch.save(model.state_dict(), save_path)
            else:
                counter += 1
                if counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch + 1}")
                    model.load_state_dict(torch.load(save_path))
                    break

    if not patience:
        torch.save(model.state_dict(), save_path)

    logger.info(f"Model saved to {save_path}")
    return model

# --------------------------------------------------
# EVALUATION
# --------------------------------------------------
def evaluate_and_save(model, loader, name):
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for batch in loader:
            if batch is None:
                continue

            X, y, l = batch
            preds = model(X.to(cfg.DEVICE), l).argmax(1).cpu().numpy()
            y_true.extend(y.numpy())
            y_pred.extend(preds)

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    logger.info(f"Final {name} | Acc: {acc:.4f} | F1: {f1:.4f}")

    with open(cfg.OUTPUT_DIR / f"{name.lower()}_metrics.txt", "w") as f:
        f.write(classification_report(y_true, y_pred, zero_division=0))

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.savefig(cfg.OUTPUT_DIR / f"{name.lower()}_confusion_matrix.png")
    plt.close()

# --------------------------------------------------
# MAIN
# --------------------------------------------------
def main():
    log_configuration()

    if not cfg.PROCESSED_DATA_PATH.exists():
        return

    saved = torch.load(cfg.PROCESSED_DATA_PATH, map_location="cpu")
    data = saved["data"]
    word2idx = saved["word2idx"]

    vocab_size = len(word2idx) + 1
    dataset = LegalDataset(data, word2idx)

    train_len = int(0.8 * len(dataset))
    train_ds, test_ds = random_split(
        dataset, [train_len, len(dataset) - train_len]
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )

    bl = train_model(
        BaselineLSTM(vocab_size),
        train_loader,
        test_loader,
        cfg.LSTM_EPOCHS,
        cfg.LSTM_LR,
        "Baseline_LSTM",
    )
    evaluate_and_save(bl, test_loader, "Baseline_LSTM")

    imp = train_model(
        ImprovedCNN(vocab_size),
        train_loader,
        test_loader,
        cfg.CNN_EPOCHS,
        cfg.CNN_LR,
        "Improved_CNN",
        cfg.CNN_PATIENCE,
    )
    evaluate_and_save(imp, test_loader, "Improved_CNN")

if __name__ == "__main__":
    main()
