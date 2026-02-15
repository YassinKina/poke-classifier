import os

# Get the project root directory (parent of src/)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Paths uses throughout entire project for consistency
CLEAN_DATASET_PATH = os.path.join(PROJECT_ROOT, "data", "pokemon_clean")
DATASET_PATH = os.path.join(PROJECT_ROOT, "data", "fcakyon___pokemon-classification")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "pokemon_cnn_best.pth")
CONFIG_PATH = os.path.join(PROJECT_ROOT, "config", "config.yaml")