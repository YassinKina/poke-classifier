import os
from datasets import load_dataset, load_from_disk
from PIL import Image
import matplotlib.pyplot as plt


def download_dataset():
    """Download the pokemon dataaset if it doesn't exist."""

    data_path = "../data"
    os.makedirs(data_path, exist_ok=True)

    # Check if dataset exists locally
    if os.path.exists(os.path.join(data_path, "fcakyon___pokemon-classification")):
        print(f"The pokemon dataset is already downloaded. Loading locally from {data_path}")


        return

    print("Downloading pokemon dataset...")

    # Download dataset from HF
    ds = load_dataset("fcakyon/pokemon-classification", cache_dir=data_path, revision="refs/convert/parquet")


    return


def load_local_data():

    base_path = "../data/fcakyon___pokemon-classification/default/0.0.0/c07895408e16b7f50c16d7fb8abbcae470621248"

    #  Point to the specific files because they are all in one directory
    data_files = {
        "train": os.path.join(base_path, "pokemon-classification-train.arrow"),
        "validation": os.path.join(base_path, "pokemon-classification-validation.arrow"),
        "test": os.path.join(base_path, "pokemon-classification-test.arrow")
    }

    # Load via the arrow builder
    ds = load_dataset("arrow", data_files=data_files)
    return ds


    return ds

def view_dataset(ds):

    # 1. Grab one example from the training split
    sample = ds['train'][0]

    # 2. Extract the image and label
    img = sample['image']
    label_id = sample['labels']

    # 3. Use matplotlib to show it
    plt.imshow(img)
    plt.title(f"Pokemon Label ID: {label_id}")
    plt.axis('off')  # Hide the X and Y pixel coordinates
    plt.show()


def get_data_loaders(transform, batch_size, dataset_path):
    """Creates and returns training and validation data loaders for a dataset.

        Args:
            transform: The torchvision transforms to be applied to the images.
            batch_size: The number of samples per batch in the data loaders.
            dataset_path: The root path to the dataset directory.

        Returns:
            A tuple containing the training and validation data loaders.
        """
    #Check if dataset is alreadz downloaded
    pass

def test_main():
    download_dataset()
    ds = load_local_data()
    print(ds)

    view_dataset(ds)

test_main()


