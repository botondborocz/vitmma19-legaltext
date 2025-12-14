#!/bin/bash 
set -e 

# --- STEP 1: Fix Directory Context ---
# Get the directory where this script file is actually located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Navigate to the script's directory
cd "$SCRIPT_DIR"

# Check if data_loader.py is here. If not, assuming the script is in a 
# subdirectory (like /scripts) and the code is in the parent, we move up.
if [ ! -f "data_loader.py" ]; then
    echo "Info: 'data_loader.py' not found in $SCRIPT_DIR. Moving to parent directory..."
    cd ..
fi

echo "Working directory set to: $(pwd)"

echo "Starting the pipeline..."

# Check if the file actually exists before running to prevent cryptic errors
if [ ! -f "data_loader.py" ]; then
    echo "CRITICAL ERROR: 'data_loader.py' still not found in $(pwd)."
    echo "Please ensure you are running this script from the project root or the script folder."
    exit 1
fi

echo "Running data preprocessing..."
python data_loader.py

echo "Running baseline model training..." 
python full_baseline.py

echo "Running best model training..."
python train_plan8.py

echo "Evaluating best model vs baseline model..."
python compare.py --plan "8"

echo "Pipeline finished successfully."