import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import re
from pathlib import Path

try:
    import config as cfg
    from utils import setup_logger
except ImportError:
    sys.path.append(str(Path(__file__).parent))
    import config as cfg
    from utils import setup_logger

logger = setup_logger("Inference")

# --- MODEL DEFINITION (Must match training) ---
class ImprovedCNN(nn.Module):
    def __init__(self, vs):
        super().__init__()
        self.embedding = nn.Embedding(vs, cfg.CNN_EMBED_DIM, padding_idx=0)
        self.convs = nn.ModuleList([nn.Conv1d(cfg.CNN_EMBED_DIM, cfg.CNN_NUM_FILTERS, k) for k in cfg.CNN_FILTER_SIZES])
        self.dropout = nn.Dropout(cfg.CNN_DROPOUT)
        self.fc = nn.Linear(cfg.CNN_NUM_FILTERS * len(cfg.CNN_FILTER_SIZES), cfg.NUM_CLASSES)

    def forward(self, x, l=None):
        e = self.embedding(x).permute(0, 2, 1)
        c = [F.max_pool1d(F.relu(conv(e)), conv(e).shape[2]).squeeze(2) for conv in self.convs]
        return self.fc(self.dropout(torch.cat(c, 1)))

def predict(text, model, w2i):
    model.eval()
    tokens = re.sub(r'[^\w\s]', '', text.lower()).split()
    if not tokens: return None, 0
    
    idx = [w2i.get(w, 0) for w in tokens]
    if len(idx) < max(cfg.CNN_FILTER_SIZES): idx += [0]*(max(cfg.CNN_FILTER_SIZES)-len(idx))
    
    with torch.no_grad():
        out = F.softmax(model(torch.tensor(idx).unsqueeze(0).to(cfg.DEVICE)), dim=1)
        conf, cls = torch.max(out, 1)
    return cls.item()+1, conf.item()

def main():
    if not cfg.PROCESSED_DATA_PATH.exists() or not cfg.IMPROVED_MODEL_PATH.exists():
        logger.error("Missing data or model.")
        return

    saved = torch.load(cfg.PROCESSED_DATA_PATH)
    model = ImprovedCNN(len(saved['word2idx'])+1)
    model.load_state_dict(torch.load(cfg.IMPROVED_MODEL_PATH, map_location=cfg.DEVICE))
    model.to(cfg.DEVICE)

    samples = [
        "A felek megállapodtak abban, hogy a szerződést közös megegyezéssel megszüntetik.",
        "A vádlottat a bíróság bűnösnek találta súlyos testi sértés bűntettében.",          
        "Az ingatlan adásvételi szerződés aláírására a jövő héten kerül sor.",
        "A felperes keresetlevelében kérte a kárának megtérítését.",
        "A hatóság elutasította a kérelmet, mivel az nem felelt meg a törvényi előírásoknak."
    ]

    logger.info("--- Hungarian Inference Samples ---")
    for s in samples:
        c, p = predict(s, model, saved['word2idx'])
        logger.info(f"Pred: Class {c}, Confidence: {p:.2f} | Text: {s[:60]}...")

if __name__ == "__main__":
    main()