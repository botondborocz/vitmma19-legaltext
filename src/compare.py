import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import numpy as np
import argparse
import sys
from pathlib import Path

# --- ARGUMENTUM PARSOLÁS ---
parser = argparse.ArgumentParser(description='Baseline és egy kiválasztott Terv összehasonlítása.')
parser.add_argument('--plan', type=str, required=True, 
                    help='A terv száma vagy neve (pl. "1", "1_v2", "6").')
args = parser.parse_args()

# --- ÚTVONALAK BEÁLLÍTÁSA ---
# A script helye: project/src/compare.py
SCRIPT_DIR = Path(__file__).parent.absolute()
# A project gyökér: project/
PROJECT_ROOT = SCRIPT_DIR.parent
# A kimeneti mappa: project/output/
OUTPUT_DIR = PROJECT_ROOT / "output"

# Ellenőrizzük, hogy létezik-e az output mappa
if not OUTPUT_DIR.exists():
    print(f"HIBA: Nem létezik az output mappa: {OUTPUT_DIR}")
    print("Kérlek hozd létre és győződj meg róla, hogy a metrika fájlok ott vannak.")
    sys.exit(1)

# Fájlnevek definiálása
PLAN_NAME = f"plan{args.plan}"
BASE_METRICS_FILE = OUTPUT_DIR / "baseline_metrics.txt"
PLAN_METRICS_FILE = OUTPUT_DIR / f"{PLAN_NAME}_metrics.txt"

BASE_MATRIX_IMG = OUTPUT_DIR / "confusion_matrix_baseline.png"
PLAN_MATRIX_IMG = OUTPUT_DIR / f"confusion_matrix_{PLAN_NAME}.png"

# Kimeneti fájlok (az összehasonlítás eredménye is az outputba megy)
CHART_OUTPUT = OUTPUT_DIR / f'comparison_chart_{PLAN_NAME}.png'
MATRIX_COMP_OUTPUT = OUTPUT_DIR / f'comparison_matrices_{PLAN_NAME}.png'

def read_metrics(filepath):
    metrics = {}
    if not filepath.exists():
        print(f"HIBA: Nem található a metrika fájl: {filepath}")
        return None
    
    with open(filepath, 'r') as f:
        for line in f:
            if ":" in line:
                key, val = line.split(":", 1)
                try:
                    metrics[key.strip()] = float(val.strip())
                except ValueError:
                    continue
    return metrics

# 1. Metrikák beolvasása
print(f"Adatok betöltése innen: {OUTPUT_DIR}")
base_metrics = read_metrics(BASE_METRICS_FILE)
plan_metrics = read_metrics(PLAN_METRICS_FILE)

if not base_metrics:
    print("Kritikus hiba: A baseline metrikák hiányoznak!")
    sys.exit(1)

if not plan_metrics:
    print(f"Kritikus hiba: A {PLAN_NAME} metrikák hiányoznak!")
    sys.exit(1)

# Kiíratás konzolra
print(f"\n--- EREDMÉNYEK ÖSSZEHASONLÍTÁSA (Baseline vs {PLAN_NAME}) ---")
print(f"{'Metrika':<15} | {'Baseline':<10} | {PLAN_NAME:<10} | {'Javulás'}")
print("-" * 55)

labels = ['Accuracy', 'F1 Score', 'Recall', 'Precision']
base_vals = []
plan_vals = []

for label in labels:
    b = base_metrics.get(label, 0.0)
    p = plan_metrics.get(label, 0.0)
    base_vals.append(b)
    plan_vals.append(p)
    diff = p - b
    print(f"{label:<15} | {b:.4f}     | {p:.4f}     | {diff:+.4f}")

# 2. Vizuális összehasonlítás (Bar Chart)
x = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
rects1 = ax.bar(x - width/2, base_vals, width, label='Baseline', color='gray')
rects2 = ax.bar(x + width/2, plan_vals, width, label=f'{PLAN_NAME}', color='orange')

ax.set_ylabel('Scores')
ax.set_title(f'Model Comparison: Baseline vs {PLAN_NAME}')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()
ax.set_ylim(0, 1.05) 

ax.bar_label(rects1, padding=3, fmt='%.2f')
ax.bar_label(rects2, padding=3, fmt='%.2f')

plt.savefig(CHART_OUTPUT)
print(f"\nÖsszehasonlító diagram elmentve: {CHART_OUTPUT}")

# 3. Confusion Matrixok összefűzése
try:
    if not BASE_MATRIX_IMG.exists() or not PLAN_MATRIX_IMG.exists():
        raise FileNotFoundError("Valamelyik forráskép hiányzik.")

    img1 = mpimg.imread(str(BASE_MATRIX_IMG))
    img2 = mpimg.imread(str(PLAN_MATRIX_IMG))

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    axes[0].imshow(img1)
    axes[0].set_title('Baseline Confusion Matrix')
    axes[0].axis('off')

    axes[1].imshow(img2)
    axes[1].set_title(f'{PLAN_NAME} Confusion Matrix')
    axes[1].axis('off')

    plt.tight_layout()
    plt.savefig(MATRIX_COMP_OUTPUT)
    print(f"Confusion Mátrixok összevetése elmentve: {MATRIX_COMP_OUTPUT}")
    
except Exception as e:
    print(f"\nFIGYELEM: Nem sikerült a képek összevetése (lehet, hogy nincsenek az output mappában): {e}")