import torch
import pytest
import numpy as np
from src.models.pl_model import LitClassifier
from src.models.mlp import MLP
from src.models.cnn import CNN
from src.models.cnn_lstm import CNNLSTM

"""
LitClassifier requires: model, model_name, training_cfg, labels_mapping, weight_tensor


For model I will create one using init_model
For model_cfg and training_cfg, I will use the configurations mocked in conftest

"""


@pytest.fixture
def labels_mapping():
    return {0: "class_0", 1: "class_1", 2: "class_1"}


@pytest.fixture
def mlp_model(mock_mlp_cfg):
    return MLP(mock_mlp_cfg, num_features=10, num_classes=3)


@pytest.fixture
def cnn_model(mock_cnn_cfg):
    return CNN(mock_cnn_cfg, num_features=100, num_classes=3)


@pytest.fixture
def cnn_lstm_model(mock_cnn_lstm_cfg):
    return CNNLSTM(mock_cnn_lstm_cfg, num_features=100, num_classes=3)


@pytest.fixture
def lit_mlp_classifier(mlp_model, mock_base_cfg, labels_mapping, weight_tensor):
    training_config = mock_base_cfg.training
    return LitClassifier(
        model=mlp_model,
        model_name="mlp",
        training_cfg=training_config,
        labels_mapping=labels_mapping,
        weight_tensor=weight_tensor,
        using_wandb=False
    )


@pytest.fixture
def lit_cnn_classifier(cnn_model, mock_base_cfg, labels_mapping, weight_tensor):
    training_config = mock_base_cfg.training
    return LitClassifier(
        model=cnn_model,
        model_name="cnn",
        training_cfg=training_config,
        labels_mapping=labels_mapping,
        weight_tensor=weight_tensor,
        using_wandb=False
    )


@pytest.fixture
def lit_cnn_lstm_classifier(cnn_lstm_model, mock_base_cfg, labels_mapping, weight_tensor):
    training_config = mock_base_cfg.training
    return LitClassifier(
        model=cnn_lstm_model,
        model_name="cnn_lstm",
        training_cfg=training_config,
        labels_mapping=labels_mapping,
        weight_tensor=weight_tensor,
        using_wandb=False
    )


@pytest.fixture
def weight_tensor():
    # e.g. for 3 classes, with custom weights
    return torch.tensor([1.0, 0.5, 2.0], dtype=torch.float)


def test_lit_classifier_initialization(mlp_model, mock_base_cfg, labels_mapping, weight_tensor):
    training_config = mock_base_cfg.training
    """Test that LitClassifier initializes correctly"""
    classifier = LitClassifier(
        model=mlp_model,
        model_name="test",
        training_cfg=training_config,
        labels_mapping=labels_mapping,
        weight_tensor=weight_tensor,
        using_wandb=False
    )
    assert isinstance(classifier, LitClassifier)
    assert isinstance(classifier.model, MLP)
    assert classifier.model_name == "test"
    assert classifier.learning_rate == training_config.learning_rate
    assert classifier.weight_decay == training_config.weight_decay
    assert classifier.labels == list(labels_mapping.values())


def test_lit_classifier_forward(lit_mlp_classifier):
    """Test that forward pass works correctly"""
    batch_size = 5
    num_features = 10
    x = torch.randn(batch_size, num_features)
    output = lit_mlp_classifier(x)
    assert output.shape == (batch_size, 3)  # num_classes = 3


def test_lit_classifier_training_step(lit_mlp_classifier):
    """Test that training step works correctly"""
    batch_size = 5
    num_features = 10
    x = torch.randn(batch_size, num_features)
    y = torch.randint(0, 2, (batch_size,))
    batch = (x, y)

    loss = lit_mlp_classifier.training_step(batch, 0)
    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0  # scalar


def test_lit_classifier_validation_step(lit_mlp_classifier):
    """Test that validation step works correctly"""
    batch_size = 5
    num_features = 10
    x = torch.randn(batch_size, num_features)
    y = torch.randint(0, 2, (batch_size,))
    batch = (x, y)

    result = lit_mlp_classifier.validation_step(batch, 0)
    assert isinstance(result, dict)
    assert "val_loss" in result
    assert "val_acc" in result


def test_lit_classifier_test_step(lit_mlp_classifier):
    """Test that test step works correctly"""
    batch_size = 5
    num_features = 10
    x = torch.randn(batch_size, num_features)
    y = torch.randint(0, 2, (batch_size,))
    batch = (x, y)

    result = lit_mlp_classifier.test_step(batch, 0)
    assert isinstance(result, dict)
    assert "test_loss" in result
    assert "test_acc" in result
    assert "preds" in result
    assert "targets" in result


def test_lit_classifier_configure_optimizers(lit_mlp_classifier):
    """Test that optimizer is configured correctly"""
    optimizer = lit_mlp_classifier.configure_optimizers()
    assert isinstance(optimizer, torch.optim.Adam)
    assert optimizer.param_groups[0]['lr'] == lit_mlp_classifier.learning_rate
    assert optimizer.param_groups[0]['weight_decay'] == lit_mlp_classifier.weight_decay


def test_lit_classifier_parameters(lit_mlp_classifier):
    """Test parameter getter and setter"""
    # Test get_parameters
    params = lit_mlp_classifier.get_parameters()
    assert isinstance(params, list)
    assert all(isinstance(p, np.ndarray) for p in params)

    # Test set_parameters
    lit_mlp_classifier.set_parameters(params)
    new_params = lit_mlp_classifier.get_parameters()
    assert all(np.array_equal(p1, p2) for p1, p2 in zip(params, new_params))
