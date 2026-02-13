import torch
from src import DynamicCNN
from src import create_dataloaders
from src import evaluate_model
import yaml
import os

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
    # Ensure correct path is used regardless of directory from which code is executed
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(current_dir, "config", "config.yaml")
    cleaned_data_path = os.path.join(current_dir, "data", "pokemon_clean")

    # 1. Load config to get hyperparameters
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # 2. Get the Test DataLoader
    _, _, test_loader = create_dataloaders(clean_data_path=cleaned_data_path, batch_size=32)

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
    model_path = os.path.join(current_dir, "models", "pokemon_cnn_best.pth")
    checkpoint = torch.load(model_path, map_location=device)

    # Extract only the state_dict (the weights) to load into the model
    model.load_state_dict(checkpoint['state_dict'])
    
    labels, preds = evaluate_model(model, test_loader, device)
    return labels, preds
    

if __name__ == "__main__":
    run_evaluation()