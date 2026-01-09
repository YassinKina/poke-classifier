import os
import torch
from datasets import load_dataset, load_from_disk, DatasetDict, concatenate_datasets
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import random
from collections import defaultdict
from imagededup.methods import CNN
from collections import defaultdict
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import transforms
from src.utils import get_mean_and_std
import glob

def download_dataset(raw_path,):
    """Download the pokemon dataaset if it doesn't exist."""

   
    os.makedirs(raw_path, exist_ok=True)

    print("Downloading pokemon dataset...")

    # Download dataset from HF
    ds = load_dataset("fcakyon/pokemon-classification", cache_dir=raw_path, revision="refs/convert/parquet")

    return

def load_local_data():
    # Use the root of your data directory
    base_search_path = "./data/fcakyon___pokemon-classification"
    
    # Look for any .arrow files recursively within that folder
    # This ignores the 'c07895...' hash folder level entirely
    arrow_files = glob.glob(os.path.join(base_search_path, "**/*.arrow"), recursive=True)
    
    # Map them to splits
    data_files = {}
    for f in arrow_files:
        if "train" in f: data_files["train"] = f
        elif "validation" in f or "valid" in f: data_files["validation"] = f
        elif "test" in f: data_files["test"] = f

    if not data_files:
        raise FileNotFoundError(f"No .arrow files found in {base_search_path}. Check if the download actually completed.")

    print(f"Found local data files: {list(data_files.keys())}")
    return load_dataset("arrow", data_files=data_files)

def view_dataset(ds, split="train", idx=None):
    # Get the total number of images in the split
    dataset_size = len(ds[split])

    # Pick a random index
    random_idx = random.randint(0, dataset_size - 1)
    
    if idx is not None and idx >= 0 and idx < dataset_size - 1:
        random_idx = idx

    # Grab the random sample
    sample = ds[split][random_idx]
    img = sample['image']
    label_id = sample['labels']

    # Extract class names from metadata
    class_names = ds[split].features['labels'].names
    pokemon_name = class_names[label_id]

    # Display pokemon
    plt.figure(figsize=(6, 6))
    plt.imshow(img)
    plt.title(f"Name: {pokemon_name}\nID: {label_id} (Index: {random_idx})")
    plt.axis('off')
    plt.show()

def sanitize_dataset(save_path):
    # 1. Load the existing arrow files (which currently have duplicates)
    ds_full_dict = load_local_data()
    
    # 2. Flatten into one single Dataset
    print("Combining all splits for global deduplication...")
    combined_ds = concatenate_datasets([
        ds_full_dict["train"], 
        ds_full_dict["validation"], 
        ds_full_dict["test"]
    ])
    
    # 3. Export images to one temp folder for the CNN
    temp_dir = "./data/temp_images_global"
    os.makedirs(temp_dir, exist_ok=True)
    
    for idx, example in enumerate(combined_ds):
        example['image'].save(f"{temp_dir}/{idx}.jpg")

    # 4. Use CNN to find duplicates globally
    cnn = CNN() 
    duplicates = cnn.find_duplicates(image_dir=temp_dir, min_similarity_threshold=0.9)

    # 5. Identify unique indices
    seen_indices = set()
    unique_indices = []
    for img_name, dupe_list in sorted(duplicates.items(), key=lambda x: int(x[0].split('.')[0])):
        idx = int(img_name.split('.')[0])
        if idx not in seen_indices:
            unique_indices.append(idx)
            seen_indices.add(idx)
            for dupe_name in dupe_list:
                seen_indices.add(int(dupe_name.split('.')[0]))

    unique_ds = combined_ds.select(unique_indices)
    print(f"Global deduplication complete: {len(unique_ds)} unique images remaining.")

    # 6. Re-split into 80/10/10 (Stratified)
    # First split: Train (80%) and Temp (20%)
    train_temp = unique_ds.train_test_split(test_size=0.2, seed=42, stratify_by_column="labels")
    
    # Second split: Divide Temp into Val (50% of temp) and Test (50% of temp)
    val_test = train_temp["test"].train_test_split(test_size=0.5, seed=42, stratify_by_column="labels")

    final_ds = DatasetDict({
        "train": train_temp["train"],
        "validation": val_test["train"],
        "test": val_test["test"]
    })

    final_ds.save_to_disk(save_path)
    print(f"Cleaned and re-stratified dataset saved at {save_path}")
    return final_ds


def save_dataset_locally(dataset):
    
    # Get the path of the current file (data_setup.py)
    current_file = Path(__file__).resolve()

    # Go up one level to the project root
    project_root = current_file.parent.parent

    # Define the data folder relative to the root
    folder_path = project_root / "data" / "pokemon_clean"

    # Ensure the directory exists
    os.makedirs(folder_path, exist_ok=True)
    
    # Save the dataset
    print(f"Saving dataset to {folder_path}...")
    dataset.save_to_disk(folder_path)
    print("Save complete!")

def split_dataset(cleaned_path="data/pokemon_clean"):
    ds =  None
    raw_path = "data/fcakyon___pokemon-classification"
    # Create cleaned dataset with balanced labels representation
    if not os.path.exists(cleaned_path):
        if not os.path.exists(raw_path):
            print("Raw dataset not found. Downloading data")
            download_dataset(raw_path)
            
        print("Clean dataset not found. Running sanitization script...")
        ds = sanitize_dataset(save_path="data/pokemon_clean")
    else:
        # Load the master sanitized dataset
        print(f"Loading cleaned data from {cleaned_path}")
        ds = load_from_disk(cleaned_path)
        
    # If we already have a DatasetDict, return
    print(f"Final Counts -> Train: {len(ds['train'])}, Val: {len(ds['validation'])}, Test: {len(ds['test'])}")
    return ds
    
  
def get_train_test_transforms(mean, std):
        """
        Creates and returns a composition of image transformations for data augmentation
        and preprocessing.

        Args:
            mean (list or tuple): A sequence of mean values for each channel.
            std (list or tuple): A sequence of standard deviation values for each channel.

        Returns:
            torchvision.transforms.Compose: A composed pipeline of train transformations.
            torchvision.transforms.Compose: A composed pipeline of test transformations without augmentations
            
        """
        
        base_transforms = [
            # Resize the input image to 256x256 pixels.
            transforms.Resize((256, 256)),
            # Crop the center 224x224 pixels of the image.
            transforms.CenterCrop(224),
             
        ]
        
        augmentations_transforms = [
      
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            
            transforms.RandomHorizontalFlip(p=0.5),
            
            transforms.RandomRotation(degrees=20),
            
            # Add contrast and saturation to ColorJitter
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            
            # Randomly turn the image grayscale (5% of the time)
            transforms.RandomGrayscale(p=0.05),
        ]
        
        main_transforms = [
            # Convert the image to a PyTorch tensor.
            transforms.ToTensor(),
            # Normalize the tensor
            transforms.Normalize(mean=mean, std=std),
        ]
        # base + augmented + main
        transforms_train = transforms.Compose(base_transforms +augmentations_transforms + main_transforms)
        transforms_test = transforms.Compose(base_transforms + main_transforms)
        
        return transforms_train, transforms_test

def create_dataloaders(clean_data_path, batch_size=64, ):
    """ Creates train, val, and test DataLoaders 

    Args:
        data_dir (_type_): _description_
        batch_size (int, optional): _description_. Defaults to 32.

    Returns:
        _type_: _description_
    """
    from src.dataset import PokemonDataset
    # Saved dataset is unsplit, so must split it first
    dataset = split_dataset(clean_data_path)
 
    # Calculate the stats using this un-normalized data
    train_mean, train_std = get_mean_and_std(dataset=dataset["train"])
    
    #Get train and test transforms
    train_transforms, test_transforms = get_train_test_transforms(mean=train_mean, std=train_std)
    
    # Setup the three distinct datasets
    train_data = PokemonDataset(dataset_split=dataset["train"], transform=train_transforms)
    val_data   = PokemonDataset(dataset_split=dataset["validation"], transform=test_transforms)
    test_data  = PokemonDataset(dataset_split=dataset["test"], transform=test_transforms)

    # Create the three loaders
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
    
def verify_dataloaders(train_dl):
    """After running create_dataloaders, check that the dataloader has the correct data

    Args:
        train_dl (_type_): _description_
    """
     # Grab the first batch from the train_loader
    images, labels = next(iter(train_dl))

    print(f"Batch Image Shape: {images.shape}") 
    # Expected: [batch_size, 3, 224, 224] -> e.g., [32, 3, 224, 224]

    print(f"Batch Label Shape: {labels.shape}")
    # Expected: [batch_size] -> e.g., [32]

    print(f"Label Data Type: {labels.dtype}")
    # Expected: torch.int64 (LongTensor)

    print(f"Image Pixel Range: Min={images.min():.2f}, Max={images.max():.2f}")
    # If normalized, you'll see negative numbers and values around 0-2.


def test_main():
    

    download_dataset()
    ds = load_local_data()
    print(ds)

    view_dataset(ds)





