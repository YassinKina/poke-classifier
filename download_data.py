import sys
import os
from src import (sanitize_dataset, 
    download_dataset, 
    load_local_data, 
    CLEAN_DATASET_PATH,
    DATASET_PATH, 
    DATA_DIR
)

# Create "data/" directory
os.makedirs("data", exist_ok=True)

dataset_is_downloaded = os.path.exists(DATASET_PATH)

# Download data if it doesnt exist yet
if not dataset_is_downloaded:
    print(f"Data not found at path {DATASET_PATH}. Starting download...")
    download_dataset(data_dir=DATA_DIR)
else:
    print(f"Data exists locally. Loading from path {DATASET_PATH}")
    
dataset = load_local_data(dataset_path=DATASET_PATH)

# Stratify dataset and save locally
sanitize_dataset(save_path=CLEAN_DATASET_PATH, dataset_path=DATASET_PATH)
