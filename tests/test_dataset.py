import pytest
import torch
from PIL import Image
from torchvision import transforms

from src import PokemonDataset  

# _______Helpers______

class DummyDataset:
    """Minimal HuggingFace-like dataset mock"""

    def __init__(self, images, labels):
        self._images = images
        self._labels = labels

    def __getitem__(self, idx):
        if isinstance(idx, str):  # mimic HF dataset behavior
            if idx == "labels":
                return self._labels
        return {
            "image": self._images[idx],
            "labels": self._labels[idx],
        }

    def __len__(self):
        return len(self._labels)


def create_dummy_image(mode="RGB"):
    return Image.new(mode, (32, 32))


# _________Tests__________

def test_len_returns_correct_length():
    images = [create_dummy_image() for _ in range(5)]
    labels = [0, 1, 2, 3, 4]
    dataset = DummyDataset(images, labels)

    pokemon_ds = PokemonDataset(dataset)

    assert len(pokemon_ds) == 5


def test_getitem_returns_tensor_and_label():
    images = [create_dummy_image()]
    labels = [1]
    dataset = DummyDataset(images, labels)

    transform = transforms.ToTensor()
    pokemon_ds = PokemonDataset(dataset, transform=transform)

    image, label = pokemon_ds[0]

    assert isinstance(image, torch.Tensor)
    assert label == 1


def test_grayscale_image_is_converted_to_rgb():
    images = [create_dummy_image(mode="L")]  # grayscale
    labels = [0]
    dataset = DummyDataset(images, labels)

    pokemon_ds = PokemonDataset(dataset)

    image, _ = pokemon_ds[0]

    assert image.mode == "RGB"


def test_transform_is_applied():
    images = [create_dummy_image()]
    labels = [0]
    dataset = DummyDataset(images, labels)

    transform = transforms.Compose([
        transforms.Resize((16, 16)),
        transforms.ToTensor()
    ])

    pokemon_ds = PokemonDataset(dataset, transform=transform)

    image, _ = pokemon_ds[0]

    assert image.shape[-1] == 16
    assert isinstance(image, torch.Tensor)


def test_error_fallback_moves_to_next_index():
    class FaultyDataset(DummyDataset):
        def __getitem__(self, idx):
            if idx == 0:
                raise ValueError("Corrupted image")
            return super().__getitem__(idx)

    images = [create_dummy_image(), create_dummy_image()]
    labels = [0, 1]
    dataset = FaultyDataset(images, labels)

    pokemon_ds = PokemonDataset(dataset, transform=transforms.ToTensor())
    pokemon_ds.error_logs = [] 

    image, label = pokemon_ds[0]

    # Should fallback to index 1
    assert label == 1
    assert len(pokemon_ds.error_logs) == 1
