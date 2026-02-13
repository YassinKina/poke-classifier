import pytest
import torch
from unittest.mock import patch, MagicMock

from src import DynamicCNN


# _________Fixtures_________


@pytest.fixture
def model():
    return DynamicCNN(
        n_layers=2,
        n_filters=[16, 32],
        kernel_sizes=[3, 3],
        dropout_rate=0.5,
        fc_size=64,
        num_classes=10,
        input_shape=(3, 32, 32)
    )


# _________Architecture Tests_________


def test_model_builds_correct_number_of_conv_layers(model):
    assert len(model.features) == 2


def test_model_has_classifier_layers(model):
    linear_layers = [m for m in model.classifier if isinstance(m, torch.nn.Linear)]
    assert len(linear_layers) == 2  # hidden + output


def test_dropout_rate_applied(model):
    dropout_layers = [m for m in model.classifier if isinstance(m, torch.nn.Dropout)]
    assert len(dropout_layers) == 2
    assert dropout_layers[0].p == 0.5


# _________Forward Pass_________


def test_forward_output_shape(model):
    x = torch.randn(4, 3, 32, 32)
    output = model(x)
    assert output.shape == (4, 10)


def test_forward_runs_without_error(model):
    x = torch.randn(1, 3, 32, 32)
    output = model(x)
    assert isinstance(output, torch.Tensor)



# _________Dynamic Flatten Size_________

def test_dynamic_flatten_size_is_correct(model):
    x = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        features_out = model.features(x)
        flattened = torch.flatten(features_out, start_dim=1)
    
    classifier_input_features = model.classifier[1].in_features
    assert flattened.shape[1] == classifier_input_features


#
# 
# _________ __repr__ Test_________


def test_repr_contains_key_information(model):
    repr_str = repr(model)

    assert "DynamicCNN" in repr_str
    assert "Convolutional Layers" in repr_str
    assert "Num Classes" in repr_str
    assert "Model Size" in repr_str



# _________ draw_CNN Test (Mocked) _________


@patch("src.model.draw_graph")
def test_draw_cnn_calls_draw_graph(mock_draw_graph, model, tmp_path):
    mock_graph = MagicMock()
    mock_graph.visual_graph = MagicMock()
    mock_draw_graph.return_value = mock_graph

    save_path = tmp_path / "model_architecture"

    graph = model.draw_CNN(
        input_size=(1, 3, 32, 32),
        save_path=str(save_path)
    )

    mock_draw_graph.assert_called_once()
    mock_graph.visual_graph.render.assert_called_once()
    assert graph == mock_graph.visual_graph
