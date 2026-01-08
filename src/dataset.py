from torch.utils.data import Dataset
import torch
import numpy as np
from torchvision import transforms
from datasets import load_from_disk


class PokemonDataset(Dataset):
    def __init__(self, dataset_split, transform=None):
        """
         Args:
            dataset_path: Path to the Pokemon images.
            transform: PyTorch transforms (augmentation/normalization).
        """
        # self.dataset_path = dataset_path
        self.transform = transform
        
        self.dataset = dataset_split
        self.labels = self.load_labels(self.dataset)

    def __len__(self):
       # Returns number of images in the dataset
        return len(self.dataset)

    def __getitem__(self, idx):
        """
        Args:
            idx: The index of the item to return.
        Returns:
            A tuple (image, label)
        """
        
        item = self.retrieve_image(idx)
        
        image = item['image']
        label = item['labels']

        # Make sure image is RGB
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Apply transform if it exists
        if self.transform:
            image = self.transform(image)

        return image, label
    
    def retrieve_image(self, idx):
        """
        Loads a single image from disk based on its index.

        Args:
            idx (int): The index of the image to load.

        Returns:
            PIL.Image.Image: The loaded image, converted to RGB.
        """
        return self.dataset[idx]
        
    def load_labels(self, dataset):
        """Loads dataset labels for PokemonDataset

        Returns:
            _type_: _description_
        """
        labels = dataset["labels"]
        return labels
        
    
   
    
        


