import torch
import yaml
import os
from src.model import DynamicCNN
from src.data_setup import create_dataloaders, create_data_dir
from src.engine import evaluate_model
from src.paths import CLEAN_DATASET_PATH, CONFIG_PATH, MODEL_PATH, DATA_DIR, DATASET_PATH

def run_evaluation():
    """
    Executes the final model evaluation on the hold-out test dataset.

    This script loads the project configuration, initializes the DynamicCNN architecture, 
    and restores the optimal model weights from a local checkpoint. It performs a 
    complete forward pass over the test set to generate performance metrics, 
    serving as the final validation of the model's generalization capabilities 
    before deployment.

    The function utilizes a project-specific YAML configuration for architectural 
    consistency and manages device placement (MPS/CPU) automatically.

    Returns:
        tuple: A tuple containing:
            - labels (np.ndarray): The true ground-truth class labels from the test set.
            - preds (np.ndarray): The model's predicted class labels.
        
    """
    create_data_dir(data_dir=DATA_DIR, dataset_path=DATASET_PATH, clean_dataset_path=CLEAN_DATASET_PATH)
    # 1. Load config to get hyperparameters
    with open(CONFIG_PATH, "r") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    # 2. Get the Test DataLoader
    _, _, test_loader = create_dataloaders(clean_data_path=CLEAN_DATASET_PATH, batch_size=32)

    # 3. Initialize 
    model = DynamicCNN(
        n_layers=cfg['model']['n_layers'],
        n_filters=cfg['model']['n_filters'],
        kernel_sizes=cfg['model']['kernel_sizes'],
        dropout_rate=cfg['model']['dropout_rate'],
        fc_size=cfg['model']['fc_size'],
        num_classes=cfg["model"]["num_classes"]
    ).to(device)
    
    # Load the whole dictionary
    checkpoint = torch.load(MODEL_PATH, map_location=device)

    # Extract only the state_dict (the weights) to load into the model
    model.load_state_dict(checkpoint['state_dict'])
    
    labels, preds = evaluate_model(model, test_loader, device)
    
    return labels, preds


if __name__ == "__main__":
    run_evaluation()