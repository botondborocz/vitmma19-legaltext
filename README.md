# Deep Learning Class (VITMMA19) Project Work

## Project Details

### Project Information

* **Selected Topic:** Legal Text Decoder
* **Student Name:** Böröcz Botond
* **Aiming for +1 Mark:** No

### Solution Description

The goal of this project is to classify Hungarian legal text segments into one of five distinct categories (e.g., Contract, Criminal, etc.). The solution implements a complete Deep Learning pipeline including automated data acquisition, preprocessing, baseline modeling, and performance improvement using a more advanced architecture.

**Methodology:**
1.  **Data Processing:** The raw JSON data is downloaded and processed to extract text and labels. Text normalization includes lowercasing and removing special characters (while preserving Hungarian accents). We use a frequency-based vocabulary to map words to integer indices.
2.  **Baseline Model (LSTM):** A Long Short-Term Memory (LSTM) network was implemented as a reference. It uses an Embedding layer followed by a single LSTM layer and a fully connected classification head. This serves as a benchmark for performance.
3.  **Improved Model (CNN):** A 1D Convolutional Neural Network (CNN) was developed to capture local n-gram patterns in the text. It utilizes multiple filter sizes (3, 4, and 5) to detect phrases of varying lengths, followed by max-pooling and dropout for regularization.
4.  **Training:** Both models are trained using Cross Entropy Loss and the Adam optimizer. The pipeline includes Early Stopping to prevent overfitting.

**Results:**
The comparison shows that the Improved CNN model generally converges faster and achieves higher stability compared to the Baseline LSTM on this specific dataset, effectively capturing the key keywords that distinguish legal categories.

### Extra Credit Justification
*N/A*

---

## Data Preparation

The project includes an automated script (`src/01-data-preprocessing.py`) to handle data preparation.

**Process:**
1.  **Download:** The script automatically downloads the raw dataset (ZIP file) from the provided URL directly into the `data/` folder.
2.  **Extraction:** It extracts the JSON files from the archive, filtering out metadata.
3.  **Cleaning:** Text is normalized (lowercased, punctuation removed) to reduce vocabulary noise.
4.  **Tokenization:** A vocabulary is built from the training corpus, mapping words to unique integers. Rare words are handled by the fixed vocabulary size.
5.  **Serialization:** The processed dataset (input tensors and labels) and the `word2idx` dictionary are saved as a PyTorch tensor file (`data/processed_data.pt`) for efficient loading during training.

**How to run:**
The Docker container runs this step automatically. Alternatively, you can run `python src/01-data-preprocessing.py` manually.

---

## Docker Instructions

This project is containerized using Docker. Follow the instructions below to build and run the solution.

### Build

Run the following command in the root directory of the repository to build the Docker image:

```bash
docker build -t dl-project .
```

### Run
To run the solution, use the following command. It is critical to mount the log, data, and output directories so that the results (including the required log file) are saved to your host machine.

```bash
docker run --rm -v your/absolute/path/to/output/folder:/app/output  dl-project > log/run.log 2>&1
```

## File Structure and Functions

The repository is structured as follows:

* **src/**: Contains the source code for the machine learning pipeline.
    * `01-data-preprocessing.py`: Scripts for loading, downloading, cleaning, and preprocessing the raw data.
    * `02-training.py`: The main script for defining the LSTM/CNN models and executing the training loop.
    * `03-evaluation.py`: Scripts for evaluating the trained models on test data and generating metrics/charts.
    * `04-inference.py`: Script for running the improved model on new, unseen Hungarian text to generate predictions.
    * `config.py`: Central configuration file containing hyperparameters (epochs, learning rates) and project paths.
    * `utils.py`: Helper functions for logging configuration and parameter counting.
    * `run.sh`: Bash script that orchestrates the sequential execution of the full pipeline (Preprocessing → Training → Evaluation → Inference).
    * `train_plan1.py`: Script for trying different improved models.
    * `train_plan1_v2.py`: Script for trying different improved models.
    * `train_plan2.py`: Script for trying different improved models.
    * `train_plan3.py`: Script for trying different improved models.
    * `train_plan4.py`: Script for trying different improved models.
    * `train_plan5.py`: Script for trying different improved models.
    * `train_plan6.py`: Script for trying different improved models.
    * `train_plan7.py`: Script for trying different improved models.
    * `train_plan8.py`: Script for trying different improved models.
    * `train_plan9.py`: Script for trying different improved models.
    * `train_plan10.py`: Script for trying different improved models.
* **log/**: Contains log files.
    * `run.log`: The main log file capturing the output of the training run, required for grading.
* **Root Directory**:
    * `Dockerfile`: Configuration file for building the Docker image with the necessary environment and dependencies.
    * `requirements.txt`: List of Python dependencies required for the project (torch, pandas, sklearn, etc.).
    * `README.md`: Project documentation and instructions.
    * `.gitignore`: Specifies intentionally untracked files to ignore.