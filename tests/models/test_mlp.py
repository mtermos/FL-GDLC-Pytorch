import torch
import pytest
from src.models.mlp import MLP


@pytest.fixture
def mlp_model(mock_mlp_cfg):
    return MLP(mock_mlp_cfg, num_features=10, num_classes=2)


def test_mlp_initialization(mock_mlp_cfg):
    """Test that MLP initializes correctly with given configuration"""
    model = MLP(mock_mlp_cfg, num_features=10, num_classes=2)
    assert isinstance(model, MLP)
    assert isinstance(model.network, torch.nn.Sequential)


def test_mlp_forward_pass(mlp_model):
    """Test that forward pass works with correct input shape"""
    batch_size = 5
    num_features = 10
    x = torch.randn(batch_size, num_features)
    output = mlp_model(x)

    assert output.shape == (batch_size, 2)  # num_classes = 2
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()


def test_mlp_layers(mlp_model):
    """Test that the model has the correct number of layers"""
    # Count the number of layers in the network
    num_layers = len(list(mlp_model.network))
    # Expected layers: 3 Linear layers + 3 ReLU activations + 3 LayerNorm layers + 3 Dropout layers + 1 Linear (classifier)
    assert num_layers == 13


def test_mlp_device(mlp_model):
    """Test that model can be moved to different devices"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mlp_model.to(device)
    assert next(mlp_model.parameters()).device == device
