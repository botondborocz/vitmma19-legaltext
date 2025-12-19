import zipfile
import json
import torch
import re
import requests
import sys
from tqdm import tqdm
from pathlib import Path

# Import Config & Utils
try:
    from config import (
        DATA_URL, ZIP_SAVE_PATH, PROCESSED_DATA_PATH, DATA_DIR
    )
    from utils import setup_logger
except ImportError:
    sys.path.append(str(Path(__file__).parent))
    from config import (
        DATA_URL, ZIP_SAVE_PATH, PROCESSED_DATA_PATH, DATA_DIR
    )
    from utils import setup_logger

logger = setup_logger("Preprocessing")

# Ensure Directory Exists
DATA_DIR.mkdir(parents=True, exist_ok=True)

def download_file(url, destination):
    logger.info(f"Downloading data from {url}...")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        total_size = int(response.headers.get('content-length', 0))
        
        with open(destination, 'wb') as file, tqdm(
            desc=destination.name, total=total_size, unit='iB', unit_scale=True, unit_divisor=1024
        ) as bar:
            for data in response.iter_content(1024):
                size = file.write(data)
                bar.update(size)
        logger.info(f"Download saved to: {destination}")
    except Exception as e:
        logger.error(f"Error downloading file: {e}")
        sys.exit(1)

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def create_and_save_dataset():
    if "YOUR_LINK_HERE" in DATA_URL:
        logger.error("Please update DATA_URL in src/config.py!")
        sys.exit(1)
        
    if not ZIP_SAVE_PATH.exists():
        download_file(DATA_URL, ZIP_SAVE_PATH)
    else:
        logger.info(f"Zip file exists at {ZIP_SAVE_PATH}, skipping download.")

    logger.info(f"Processing data from: {ZIP_SAVE_PATH}")
    data = []
    vocab_counter = {} 

    try:
        with zipfile.ZipFile(ZIP_SAVE_PATH, 'r') as z:
            for filename in tqdm(z.namelist(), desc="Processing files"):
                if not filename.endswith('.json') or "__MACOSX" in filename or "consensus/" in filename:
                    continue
                try:
                    with z.open(filename) as f:
                        content = json.load(f)
                        if isinstance(content, dict): content = [content]
                        for task in content:
                            text = task.get('data', {}).get('text', "")
                            annotations = task.get('annotations', [])
                            if not text or not annotations: continue
                            
                            result = annotations[0].get('result', [])
                            if not result: continue
                            
                            choice = result[0]['value']['choices'][0]
                            label = int(choice.split('-')[0])
                            
                            text = clean_text(text)
                            tokens = text.split()
                            if not tokens: continue
                            
                            data.append({"text": tokens, "label": label})
                            for w in tokens: vocab_counter[w] = vocab_counter.get(w, 0) + 1
                except: continue
    except zipfile.BadZipFile:
        logger.error("Invalid zip file.")
        sys.exit(1)

    word2idx = {w: i+1 for i, w in enumerate(sorted(vocab_counter.keys()))}
    
    logger.info(f"Processing Complete. Samples: {len(data)}, Vocab: {len(word2idx)}")
    torch.save({'data': data, 'word2idx': word2idx}, PROCESSED_DATA_PATH)
    logger.info(f"Saved to {PROCESSED_DATA_PATH}")

if __name__ == "__main__":
    create_and_save_dataset()