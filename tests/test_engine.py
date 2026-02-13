import pytest
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from unittest.mock import MagicMock, patch

from src.engine import (
    train_epoch,
    validate_epoch,
    train_model,
    evaluate_model
)


# _________Fixtures_________


@pytest.fixture
def simple_model():
    model = torch.nn.Linear(10, 6)  # 6 classes (for top5 test)
    return model


@pytest.fixture
def dummy_dataloader():
    X = torch.randn(20, 10)
    y = torch.randint(0, 6, (20,))
    dataset = TensorDataset(X, y)
    return DataLoader(dataset, batch_size=5)


@pytest.fixture
def loss_fn():
    return torch.nn.CrossEntropyLoss()


@pytest.fixture
def optimizer(simple_model):
    return torch.optim.SGD(simple_model.parameters(), lr=0.01)


@pytest.fixture
def device():
    return torch.device("cpu")



# ______train_epoch tests______
@patch("src.engine.get_num_correct_in_top5")
def test_train_epoch_runs(mock_top5, simple_model, dummy_dataloader, optimizer, loss_fn, device):
    mock_top5.return_value = 3

    pbar = MagicMock()

    loss, top1, top5 = train_epoch(
        simple_model,
        dummy_dataloader,
        optimizer,
        loss_fn,
        device,
        pbar
    )

    assert isinstance(loss, float)
    assert 0 <= top1 <= 1
    assert 0 <= top5 <= 1
    assert pbar.update_batch.called


# ______validate_epoch tests______

@patch("src.engine.get_num_correct_in_top5")
def test_validate_epoch_runs(mock_top5, simple_model, dummy_dataloader, loss_fn, device):
    mock_top5.return_value = 2

    loss, top1, top5 = validate_epoch(
        simple_model,
        dummy_dataloader,
        loss_fn,
        device
    )

    assert isinstance(loss, float)
    assert 0 <= top1 <= 1
    assert 0 <= top5 <= 1



# ______train_model tests____

@patch("src.engine.get_num_correct_in_top5")
@patch("src.engine.get_best_val_accuracy")
@patch("src.engine.NestedProgressBar")
@patch("torch.save")
def test_train_model_saves_checkpoint(
    mock_save,
    mock_pbar,
    mock_best_acc,
    mock_top5,
    simple_model,
    dummy_dataloader,
    optimizer,
    loss_fn,
    device
):
    mock_best_acc.return_value = 0.0
    mock_top5.return_value = 5

    scheduler = MagicMock()
    wandb_run = MagicMock()

    best_acc = train_model(
        model=simple_model,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_func=loss_fn,
        train_dataloader=dummy_dataloader,
        val_dataloader=dummy_dataloader,
        device=device,
        n_epochs=1,
        wandb_run=wandb_run
    )

    assert mock_save.called
    assert isinstance(best_acc, float)
    assert scheduler.step.called
    assert wandb_run.log.called



# ______test_model______


@patch("src.engine.get_num_correct_in_top5")
def test_test_model_outputs_arrays(mock_top5, simple_model, dummy_dataloader, device):
    mock_top5.return_value = 3

    labels, preds = evaluate_model(simple_model, dummy_dataloader, device)

    assert isinstance(labels, np.ndarray)
    assert isinstance(preds, np.ndarray)
    assert len(labels) == len(preds)
