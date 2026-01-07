from torch.utils.data import Dataset
import torch
import numpy as np


class PokemonDataset(Dataset):
    def __init__(self, hf_dataset, transform=None):
        """
         Args:
            hf_dataset: The loaded Hugging Face dataset object.
            transform: PyTorch transforms (augmentation/normalization).
        """
        self.hf_dataset = hf_dataset
        self.transform = transform

    def __len__(self):
       # Returns number of images in the dataset
        return len(self.hf_dataset)

    def __getitem(self, idx):
        """
        Args:
            idx: The index of the item to return.
        Returns:
            A tuple (image, label)
        """
        item = self.hf_dataset[idx]
        image = item['image']
        label = item['label']

        # Make sure image is RGB
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Apply transform if it exists
        if self.transform:
            image = self.transform(image)

        return image, label


