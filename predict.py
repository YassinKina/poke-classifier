import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import yaml
from src import DynamicCNN, split_dataset, get_mean_and_std, get_train_test_transforms
from typing import Tuple

def load_model(model_path: str, config_path: str, device: torch.device) -> DynamicCNN:
    """
    Initializes the model architecture and loads saved weights from a checkpoint.
    """
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    model = DynamicCNN(
        n_layers=cfg['model']['n_layers'],
        n_filters=cfg['model']['n_filters'],
        kernel_sizes=cfg['model']['kernel_sizes'],
        dropout_rate=cfg['model']['dropout_rate'],
        fc_size=cfg['model']['fc_size'],
        num_classes=150
    ).to(device)

    
    # Load the checkpoint dictionary and extract state_dict
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    return model

def predict(image_path: str,
            model: DynamicCNN,
            class_names: list,
            device: torch.device,
            mean:torch.Tensor,
            std:torch.Tensor) -> Tuple[str, float]:
    """
    Preprocesses an input image and returns the predicted class and confidence.
    """
  
    # 1. Image Preprocessing
    _, test_transform = get_train_test_transforms(mean=mean, std=std)

    image = Image.open(image_path).convert("RGB")
    image_tensor = test_transform(image).unsqueeze(0).to(device) # Add batch dimension

    # 2. Inference
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = F.softmax(output, dim=1)
        confidence, predicted_idx = torch.max(probabilities, 1)

    return class_names[predicted_idx.item()], confidence.item()

def main():
    """
    Entry point for CLI-based Pokémon identification.
    """
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    # Get cleaned, split dataset
    ds = split_dataset()
    mean, std = get_mean_and_std(dataset=ds["train"])
    # Load class names from the dataset metadata
    class_names = ds['train'].features['labels'].names

    # Initialize model
    model = load_model("models/pokemon_cnn_best.pth", "config/config.yaml", device)
    
    image_name = input("Enter image name: ")
    img_path = f"samples/{image_name}" 
    
    name, conf = predict(img_path, model, class_names, device, mean, std)

    print(f"\n--- Prediction Results ---")
    print(f"Pokémon: {name}")
    print(f"Confidence: {conf:.2%}")

if __name__ == "__main__":
    main()