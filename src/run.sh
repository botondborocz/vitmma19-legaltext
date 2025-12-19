#!/bin/bash 
set -e 

# --- STEP 1: Setup Logging ---
mkdir -p /app/log

# Redirect output to console AND log file
exec > >(tee -a /app/log/run.log) 2>&1

echo "========================================================"
echo "PIPELINE STARTED AT: $(date)"
echo "========================================================"

# --- STEP 2: Force Navigation to Source Folder ---
# We hardcode this to match the Dockerfile structure
cd /app/src

echo "Current Directory: $(pwd)"
echo "Files in current directory:"
ls -1 *.py

# --- STEP 3: Verify Files Exist ---
if [ ! -f "01-data-preprocessing.py" ]; then
    echo "CRITICAL ERROR: '01-data-preprocessing.py' not found in $(pwd)."
    exit 1
fi

# --- STEP 4: Execute Scripts ---

echo "--- 1. Running Data Preprocessing ---"
python 01-data-preprocessing.py

echo "--- 2. Running Model Training ---" 
python 02-training.py

echo "--- 3. Running Evaluation ---"
python 03-evaluation.py

echo "--- 4. Running Inference ---"
python 04-inference.py

echo "========================================================"
echo "PIPELINE FINISHED SUCCESSFULLY AT: $(date)"
echo "========================================================"