import torch
import os
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import yaml
from typing import Tuple
from src import (DynamicCNN, 
                 split_dataset,
                 get_mean_and_std,
                 get_class_names,
                 get_train_test_transforms,
                 create_data_dir,
                 CLEAN_DATASET_PATH,
                 DATASET_PATH,
                 DATA_DIR,
                 MODEL_PATH,
                 CONFIG_PATH
) 



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
        num_classes=cfg["training"]["num_classes"]
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
    
    create_data_dir(data_dir=DATA_DIR, dataset_path=DATASET_PATH, clean_dataset_path=CLEAN_DATASET_PATH)
    
    # Get cleaned, split dataset
    ds = split_dataset(DATA_DIR, DATASET_PATH, CLEAN_DATASET_PATH)
    mean, std = get_mean_and_std(dataset=ds["train"])
    # Load class names from the dataset metadata
    class_names = get_class_names()

    # Initialize model
    model = load_model(MODEL_PATH, CONFIG_PATH, device)
    
    # Perform prediction from user input
    while True:
        image_name = input("Enter any file name from the \"samples\" folder (e.g. \"golbat.png\") or \"exit\" to quit: ")
        if image_name == "exit" or image_name == "quit":
            break
        img_path = f"samples/{image_name}" 
        
        folder_path = os.path.join("samples")
        file_names = [f for f in os.listdir(folder_path)]
        
        # If image name is invalid, have user enter a new new
        if image_name not in file_names:
            print("Invalid image name entered, be sure to enter the full file name.")
            continue
        
        name, conf = predict(img_path, model, class_names, device, mean, std)

        print(f"\n--- Prediction Results ---")
        print(f"Pokémon: {name}")
        print(f"Confidence: {conf:.2%}\n")

if __name__ == "__main__":
    main()