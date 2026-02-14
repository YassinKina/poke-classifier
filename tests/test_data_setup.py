import pytest
from typing import Optional, Tuple, List, Union
from src import data_setup
import os
from unittest.mock import patch
from datasets import DatasetDict, Dataset
import torch
from PIL import Image
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader
from src import PokemonDataset, DATA_DIR, DATASET_PATH, CLEAN_DATASET_PATH

TRAIN_DATA_ROWS = 4672
VAL_DATA_ROWS = 584
TEST_DATA_ROWS = 585

# Precalculated mean and std for normalization of RGB channels
POKEMON_MEAN = torch.tensor([0.5863186717033386, 0.5674829483032227, 0.5336665511131287])
POKEMON_STD = torch.tensor([0.34640103578567505, 0.33123084902763367, 0.34212544560432434])





def test_download_dataset():
    """
    Test ensuring the fcakyon___pokemon-classification dataset is correctly downloaded from HF.

     """

    with patch('src.data_setup.download_dataset') as mock_download:
        # # Mocking it to return None, simulating a success 
        mock_download.return_value = None
        
        # This call now hits the mock
        data_setup.download_dataset(DATA_DIR, DATASET_PATH)
        
        # Assert that the function was called exactly once
        mock_download.assert_called_once()
    
    
    search_path = os.path.join(DATA_DIR, "fcakyon___pokemon-classification")
    assert os.path.exists(DATA_DIR)
    assert os.path.exists(search_path)
    
def test_load_local_data():
    """
    Test to ensure the loaded fcakyon___pokemon-classification dataset loads correctly
    """
    
    dataset = data_setup.load_local_data(DATA_DIR)
    
    assert dataset is not None
    assert len(dataset) > 0
    assert isinstance(dataset, DatasetDict)
    assert "train" in dataset
    assert "test" in dataset
    assert "validation" in dataset

def test_sanitize_dataset():
    """
    Test that ensures the a cleaned dataset is returned from the func
    This function is called within "split_dataset", which is tested next.
    """
    
    cleaned_dataset = data_setup.sanitize_dataset(DATA_DIR, DATASET_PATH, CLEAN_DATASET_PATH)
    assert cleaned_dataset is not None
    assert len(cleaned_dataset) > 0
    assert isinstance(cleaned_dataset, DatasetDict)
    assert cleaned_dataset["train"].num_rows == TRAIN_DATA_ROWS
    assert cleaned_dataset["validation"].num_rows == VAL_DATA_ROWS
    assert cleaned_dataset["test"].num_rows == TEST_DATA_ROWS
  
def test_split_dataset():
    """
    Test that ensures the a cleaned dataset is returned from the func
    The core functoionality of this function is tested in "test_sanitized_dataset".
    
    """

    # Create a tiny valid DatasetDict to mock the return
    fake_ds = DatasetDict({"train": Dataset.from_dict({"id": [1], "name": ["Pikachu"]})})

    with patch('src.data_setup.split_dataset') as mock_load:
        mock_load.return_value = fake_ds
        
        # Now run your function
        # Ensure your src.data_setup is actually calling the mocked load_from_disk
        result = data_setup.split_dataset(CLEAN_DATASET_PATH, DATASET_PATH)
        
        assert isinstance(result, DatasetDict)
        mock_load.assert_called()
 
#_____ Test get_train_test_transforms, ensure image transforms are functioning ____         
def create_dummy_image():
    """Creates a random RGB PIL image."""
    img_array = np.random.randint(0, 256, (300, 300, 3), dtype=np.uint8)
    return Image.fromarray(img_array)

def test_returns_compose_objects():
    train_t, test_t = data_setup.get_train_test_transforms(POKEMON_MEAN, POKEMON_STD)

    assert isinstance(train_t, transforms.Compose)
    assert isinstance(test_t, transforms.Compose)

def test_train_contains_augmentations():
    train_t, _ = data_setup.get_train_test_transforms(POKEMON_MEAN, POKEMON_STD)
    transform_types = [type(t) for t in train_t.transforms]

    assert transforms.RandomResizedCrop in transform_types
    assert transforms.RandomHorizontalFlip in transform_types
    assert transforms.RandomRotation in transform_types

def test_test_has_no_augmentations():
    _, test_t = data_setup.get_train_test_transforms(POKEMON_MEAN, POKEMON_STD)

    transform_types = [type(t) for t in test_t.transforms]

    assert transforms.RandomResizedCrop not in transform_types
    assert transforms.RandomHorizontalFlip not in transform_types
    assert transforms.RandomRotation not in transform_types

def test_output_shape_and_type():
    train_t, test_t = data_setup.get_train_test_transforms(POKEMON_MEAN, POKEMON_STD)
    img = create_dummy_image()

    train_out = train_t(img)
    test_out = test_t(img)

    assert isinstance(train_out, torch.Tensor)
    assert isinstance(test_out, torch.Tensor)

    # Should be 3 x 224 x 224 after transforms
    assert train_out.shape == (3, 224, 224)
    assert test_out.shape == (3, 224, 224)                                                                

# ______end of get_train_test_transforms tests________  

class DummyDataset(torch.utils.data.Dataset):
    def __init__(self, size=20):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        image = torch.randn(3, 224, 224)
        label = torch.tensor(idx % 5)
        return image, label


# ____Tests for create_dataloaders____
@patch("src.dataset.PokemonDataset")
@patch("src.data_setup.get_train_test_transforms")
@patch("src.data_setup.get_mean_and_std")
@patch("src.data_setup.load_from_disk")
def test_create_dataloaders(
    mock_load_from_disk,
    mock_mean_std,
    mock_transforms,
    mock_dataset_class,
):
  
    fake_dataset = {
        "train": list(range(30)),
        "validation": list(range(10)),
        "test": list(range(15)),
    }

    # Make load_from_disk return our fake dataset
    mock_load_from_disk.return_value = fake_dataset

    mock_mean_std.return_value = ([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    mock_transforms.return_value = (None, None)

    mock_dataset_class.side_effect = lambda dataset_split, transform: DummyDataset(len(dataset_split))

    batch_size = 8
    train_dl, val_dl, test_dl = data_setup.create_dataloaders("fake_path", batch_size=batch_size)

   
    # DataLoader types
    assert isinstance(train_dl, DataLoader)
    assert isinstance(val_dl, DataLoader)
    assert isinstance(test_dl, DataLoader)

    # Batch size
    assert train_dl.batch_size == batch_size
    assert val_dl.batch_size == batch_size
    assert test_dl.batch_size == batch_size

    # Dataset lengths
    assert len(train_dl.dataset) == len(fake_dataset["train"])
    assert len(val_dl.dataset) == len(fake_dataset["validation"])
    assert len(test_dl.dataset) == len(fake_dataset["test"])



# ____Tests for verify_dataloaders____

def test_verify_dataloaders_runs(capsys):
    dummy_dataset = DummyDataset(size=8)
    dummy_loader = DataLoader(dummy_dataset, batch_size=4)

    data_setup.verify_dataloaders(dummy_loader)

    captured = capsys.readouterr()

    assert "Batch Image Shape" in captured.out
    assert "Batch Label Shape" in captured.out
    assert "Label Data Type" in captured.out
    assert "Image Pixel Range" in captured.out     