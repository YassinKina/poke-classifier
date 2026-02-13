import pytest
import torch
import numpy as np
import random
from unittest.mock import patch, MagicMock
from omegaconf import OmegaConf

from src.utils import (
    get_mean_and_std,
    get_best_val_accuracy,
    flatten_config,
    set_seed,
    get_num_correct_in_top5,
    init_wandb_run,
    NestedProgressBar,
    POKEMON_MEAN,
    POKEMON_STD
)


# _________get_mean_and_std_________

def test_get_mean_and_std_fast_returns_constants():
    mean, std = get_mean_and_std(fast=True)
    assert torch.equal(mean, POKEMON_MEAN)
    assert torch.equal(std, POKEMON_STD)


@patch("src.dataset.PokemonDataset")
@patch("src.utils.DataLoader")
def test_get_mean_and_std_slow(mock_loader, mock_dataset):
    # Fake dataset with constant tensor values
    images = torch.ones(4, 3, 224, 224) * 0.5
    labels = torch.zeros(4)

    mock_loader.return_value = [(images, labels)]

    mean, std = get_mean_and_std(dataset="dummy", fast=False)

    assert torch.allclose(mean, torch.tensor([0.5, 0.5, 0.5]), atol=1e-4)
    assert torch.allclose(std, torch.tensor([0.0, 0.0, 0.0]), atol=1e-4)



# _________get_best_val_accuracy _________


@patch("src.utils.os.path.exists")
@patch("src.utils.torch.load")
def test_get_best_val_accuracy_exists(mock_load, mock_exists):
    mock_exists.return_value = True
    mock_load.return_value = {"accuracy": 0.87}

    acc = get_best_val_accuracy("dummy.pth")
    assert acc == 0.87


@patch("src.utils.os.path.exists")
def test_get_best_val_accuracy_no_file(mock_exists):
    mock_exists.return_value = False
    acc = get_best_val_accuracy("dummy.pth")
    assert acc == 0.0



# _________flatten_config_________


def test_flatten_config_basic():
    config = {
        "training": {"lr": 0.01, "batch_size": 32},
        "device": "cpu"
    }

    flat = flatten_config(config)

    assert flat["lr"] == 0.01
    assert flat["batch_size"] == 32
    assert flat["device"] == "cpu"



#_________ set_seed_________


def test_set_seed_reproducibility():
    set_seed(123)

    a = torch.rand(1)
    b = np.random.rand()
    c = random.random()

    set_seed(123)

    assert torch.equal(a, torch.rand(1))
    assert b == np.random.rand()
    assert c == random.random()


# _________get_num_correct_in_top5 _________

def test_get_num_correct_in_top5():
    outputs = torch.tensor([
        [0.1, 0.9, 0.2, 0.3, 0.4, 0.5],
        [0.9, 0.1, 0.2, 0.3, 0.4, 0.5],
    ])
    labels = torch.tensor([1, 0])

    correct = get_num_correct_in_top5(outputs, labels)

    assert correct == 2


# _________init_wandb_run _________


@patch("src.utils.wandb.init")
def test_init_wandb_run(mock_wandb_init):
    mock_run = MagicMock()
    mock_wandb_init.return_value = mock_run

    config = OmegaConf.create({
        "training": {"lr": 0.001},
        "device": "cpu"
    })

    run = init_wandb_run(config, run_name="test_run")

    mock_wandb_init.assert_called_once()
    assert run == mock_run


# _________NestedProgressBar _________


@patch("tqdm.auto.tqdm")
def test_nested_progress_bar_updates(mock_tqdm):
    mock_bar = MagicMock()
    mock_tqdm.return_value = mock_bar

    pbar = NestedProgressBar(
        total_epochs=5,
        total_batches=10,
        use_notebook=False,
        mode="train"
    )

    pbar.update_epoch(1)
    pbar.update_batch(1)

    assert mock_bar.update.called


@patch("tqdm.auto.tqdm")
def test_nested_progress_bar_close(mock_tqdm):
    mock_bar = MagicMock()
    mock_tqdm.return_value = mock_bar

    pbar = NestedProgressBar(
        total_epochs=2,
        total_batches=2,
        mode="train"
    )

    pbar.close("done")

    assert mock_bar.close.called
