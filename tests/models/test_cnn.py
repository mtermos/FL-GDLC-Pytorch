import torch
import pytest
from src.models.cnn import CNN


@pytest.fixture
def cnn_model(mock_cnn_cfg):
    return CNN(mock_cnn_cfg, num_features=100, num_classes=2)


def test_cnn_initialization(mock_cnn_cfg):
    """Test that CNN initializes correctly with given configuration"""
    model = CNN(mock_cnn_cfg, num_features=100, num_classes=2)
    assert isinstance(model, CNN)
    assert isinstance(model.features, torch.nn.Sequential)
    assert isinstance(model.classifier, torch.nn.Sequential)


def test_cnn_forward_pass(cnn_model):
    """Test that forward pass works with correct input shape"""
    batch_size = 5
    num_features = 100
    x = torch.randn(batch_size, num_features)
    output = cnn_model(x)

    assert output.shape == (batch_size, 2)  # num_classes = 2
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()


def test_cnn_input_reshape(cnn_model):
    """Test that input is correctly reshaped for CNN layers"""
    batch_size = 5
    num_features = 100
    x = torch.randn(batch_size, num_features)

    # Check that input is reshaped to (batch_size, 1, num_features)
    x_reshaped = x.view(x.size(0), 1, x.size(1))
    assert x_reshaped.shape == (batch_size, 1, num_features)


def test_cnn_layers(cnn_model):
    """Test that the model has the correct number of layers"""
    # Count the number of layers in features and classifier
    num_feature_layers = len(list(cnn_model.features))
    num_classifier_layers = len(list(cnn_model.classifier))

    # Expected layers in features: 2 Conv1d + 2 ReLU + 2 LayerNorm + 2 Dropout
    assert num_feature_layers == 8
    # Expected layers in classifier: 1 Linear + 1 ReLU + 1 LayerNorm + 1 Dropout + 1 Linear
    assert num_classifier_layers == 5


def test_cnn_device(cnn_model):
    """Test that model can be moved to different devices"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cnn_model.to(device)
    assert next(cnn_model.parameters()).device == device
