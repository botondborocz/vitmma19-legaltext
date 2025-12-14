import zipfile
import json
import torch
import re  # ÚJ: Regular expressions a tisztításhoz
from torch.utils.data import Dataset
from pathlib import Path

# --- ADATTISZTÍTÓ FÜGGVÉNY (ÚJ) ---
def clean_text(text):
    """
    Elvégzi a szövegtisztítást:
    1. Kisbetűsítés.
    2. Speciális karakterek (írásjelek) törlése (csak betűk, számok és szóköz marad).
    3. Felesleges szóközök összevonása.
    """
    # 1. Kisbetűsítés
    text = text.lower()
    
    # 2. Speciális karakterek törlése
    # A [^\w\s] regex jelentése: "Minden, ami NEM szó-karakter (betű/szám) és NEM whitespace"
    # Ez a Pythonban szerencsére megtartja a magyar ékezetes betűket is (á, é, ű, stb.)
    text = re.sub(r'[^\w\s]', '', text)
    
    # 3. Felesleges spacek (több szóköz, tab, sortörés) cseréje egyetlen szóközre
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def load_data_from_zip(project_root):
    data = []
    vocab = set()
    
    data_dir = project_root.parent / "data"
    zip_files = list(data_dir.glob("*.zip"))
    
    if not zip_files:
        print(f"No .zip file found in {project_root}. Please check location.")
        return data, list(vocab)
        
    target_zip = zip_files[0]
    print(f"Loading data from: {target_zip.name}")

    with zipfile.ZipFile(target_zip, 'r') as z:
        for filename in z.namelist():
            if not filename.endswith('.json'): continue
            if "__MACOSX" in filename: continue
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
                            
                            # --- ÚJ: TISZTÍTÁS ALKALMAZÁSA ---
                            text = clean_text(text)
                            # ---------------------------------
                            
                            tokens = text.split()
                            if len(tokens) == 0: continue
                            
                            data.append({"text": tokens, "label": label})
                            for w in tokens: vocab.add(w)
                        except: continue
            except Exception: continue

    return data, list(vocab)

class LegalDataset(Dataset):
    def __init__(self, data, word2idx):
        self.data = data
        self.word2idx = word2idx

    def __len__(self):
        return len(self.data)

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
    for i, seq in enumerate(sequences):
        padded_seqs[i, :len(seq)] = seq
        
    return padded_seqs, torch.stack(labels), torch.tensor(lengths)