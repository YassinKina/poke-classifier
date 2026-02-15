import torch
import os
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import yaml
from typing import Tuple
from src.model import DynamicCNN
from src.data_setup import create_data_dir, split_dataset, get_train_test_transforms
from src.utils import get_class_names, get_mean_and_std
from src.paths import CLEAN_DATASET_PATH, CONFIG_PATH, MODEL_PATH, DATA_DIR, DATASET_PATH

def load_model(model_path: str, config_path: str, device: torch.device) -> DynamicCNN:
    """
    Initializes a DynamicCNN model based on a configuration file and loads pretrained weights.

    Args:
        model_path (str): Path to the model checkpoint file containing the saved state_dict.
        config_path (str): Path to the YAML configuration file specifying the model architecture and training parameters.
        device (torch.device): The device on which to load the model (CPU, CUDA, or MPS).

    Returns:
        DynamicCNN: A PyTorch model instance with loaded weights, set to evaluation mode.
    """
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    model = DynamicCNN(
        n_layers=cfg['model']['n_layers'],
        n_filters=cfg['model']['n_filters'],
        kernel_sizes=cfg['model']['kernel_sizes'],
        dropout_rate=cfg['model']['dropout_rate'],
        fc_size=cfg['model']['fc_size'],
        num_classes=cfg["training"]["num_classes"]
    ).to(device)

    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    return model

def predict(image_path: str,
            model: DynamicCNN,
            class_names: list,
            device: torch.device,
            mean: torch.Tensor,
            std: torch.Tensor) -> Tuple[str, float]:
    """
    Performs inference on a single image using the provided model.

    This function applies preprocessing to the input image, passes it through the model,
    and returns the predicted class label and the associated confidence score.

    Args:
        image_path (str): Path to the input image file.
        model (DynamicCNN): The trained PyTorch model for classification.
        class_names (list): List of class labels corresponding to model outputs.
        device (torch.device): Device on which to perform inference.
        mean (torch.Tensor): Mean values for normalization of image channels.
        std (torch.Tensor): Standard deviation values for normalization of image channels.

    Returns:
        Tuple[str, float]: A tuple containing:
            - Predicted class label as a string.
            - Confidence score of the prediction as a float (range 0-1).
    """
    _, test_transform = get_train_test_transforms(mean=mean, std=std)

    image = Image.open(image_path).convert("RGB")
    image_tensor = test_transform(image).unsqueeze(0).to(device)  # Add batch dimension

    with torch.no_grad():
        output = model(image_tensor)
        probabilities = F.softmax(output, dim=1)
        confidence, predicted_idx = torch.max(probabilities, 1)

    return class_names[predicted_idx.item()], confidence.item()

def main():
    """
    Command-line interface for interactive Pokémon image classification.

    This function:
    1. Sets up the computation device (CUDA, MPS, or CPU).
    2. Prepares the dataset directory and splits it into train/test sets.
    3. Computes normalization statistics (mean and standard deviation) for preprocessing.
    4. Loads class labels and initializes the trained model.
    5. Enters an interactive loop to accept image filenames from the user, predict their
       class using the model, and display the prediction and confidence.

    The loop continues until the user types 'exit' or 'quit'.

    """
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    create_data_dir(data_dir=DATA_DIR, dataset_path=DATASET_PATH, clean_dataset_path=CLEAN_DATASET_PATH)
    
    ds = split_dataset(DATA_DIR, DATASET_PATH, CLEAN_DATASET_PATH)
    mean, std = get_mean_and_std(dataset=ds["train"])
    class_names = get_class_names()

    model = load_model(MODEL_PATH, CONFIG_PATH, device)
    
    while True:
        image_name = input("Enter any file name from the \"samples\" folder (e.g. \"golbat.png\") or \"exit\" to quit: ").strip()
        if image_name == "exit" or image_name == "quit":
            break
        img_path = os.path.join("samples", image_name)
        
        if not os.path.exists(img_path):
            print("Invalid image name entered, be sure to enter the full file name.")
            continue
                        
        try:
            name, conf = predict(img_path, model, class_names, device, mean, std)
            print(f"\n--- Prediction Results ---")
            print(f"Pokémon: {name}")
            print(f"Confidence: {conf:.2%}\n")
        except Exception as e:
            print(f"Failed to process image: {e}")

if __name__ == "__main__":
    main()
