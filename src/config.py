import torch
from pathlib import Path

# --- PATHS ---
SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"
LOG_DIR = PROJECT_ROOT / "log"

# File Paths
ZIP_SAVE_PATH = DATA_DIR / "raw_dataset.zip"
PROCESSED_DATA_PATH = DATA_DIR / "processed_data.pt"
BASELINE_MODEL_PATH = OUTPUT_DIR / "baseline_lstm_best.pt"
IMPROVED_MODEL_PATH = OUTPUT_DIR / "improved_cnn_best.pt"
LOG_FILE_PATH = LOG_DIR / "run.log"

# --- DATA CONFIG ---
# REPLACE THIS WITH YOUR ACTUAL URL
DATA_URL = "https://bmeedu-my.sharepoint.com/:u:/g/personal/gyires-toth_balint_vik_bme_hu/IQDYwXUJcB_jQYr0bDfNT5RKARYgfKoH97zho3rxZ46KA1I?e=iFp3iz&download=1" 
RANDOM_SEED = 42
NUM_CLASSES = 5

# --- TRAINING HYPERPARAMETERS ---
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Baseline Model (LSTM)
LSTM_HIDDEN_SIZE = 64
LSTM_EMBED_DIM = 64
LSTM_EPOCHS = 20
LSTM_LR = 0.001

# Improved Model (CNN)
CNN_EMBED_DIM = 100
CNN_NUM_FILTERS = 100
CNN_FILTER_SIZES = [3, 4, 5]
CNN_DROPOUT = 0.5
CNN_EPOCHS = 30
CNN_PATIENCE = 5
CNN_LR = 0.001