import torch
import pytest
import numpy as np
import pytorch_lightning as pl
from src.models.pl_model import LitClassifier
from src.models.mlp import MLP
from src.models.cnn import CNN
from src.models.cnn_lstm import CNNLSTM


class MockModelConfig:
    def __init__(self):
        self.dense = type('DenseConfig', (), {
            'activation': 'relu',
            'units': [64, 32],
            'batch_norm': True,
            'layer_norm': False,
            'dropout': True,
            'dropout_rate': 0.2
        })()
        self.cnn = type('CNNConfig', (), {
            'activation': 'relu',
            'filters': [32, 64],
            'kernel_sizes': [3, 3],
            'batch_norm': True,
            'dropout': True,
            'dropout_rate': 0.2
        })()
        self.lstm = type('LSTMConfig', (), {
            'activation': 'tanh',
            'hidden_size': [128, 64],
            'dropout': True,
            'dropout_rate': 0.2
        })()


class MockTrainingConfig:
    def __init__(self):
        self.learning_rate = 0.001
        self.weight_decay = 0.0001
        self.optimizer = "adam"
        self.multi_class = True
        self.batch_size = 32


@pytest.fixture
def model_config():
    return MockModelConfig()


@pytest.fixture
def training_config():
    return MockTrainingConfig()


@pytest.fixture
def labels_mapping():
    return {0: "class_0", 1: "class_1"}


@pytest.fixture
def mlp_model(model_config):
    return MLP(model_config, num_features=10, num_classes=2)


@pytest.fixture
def cnn_model(model_config):
    return CNN(model_config, num_features=100, num_classes=2)


@pytest.fixture
def cnn_lstm_model(model_config):
    return CNNLSTM(model_config, num_features=100, num_classes=2)


@pytest.fixture
def lit_mlp_classifier(mlp_model, training_config, labels_mapping):
    return LitClassifier(
        model=mlp_model,
        model_name="mlp",
        training_cfg=training_config,
        labels_mapping=labels_mapping,
        using_wandb=False
    )


@pytest.fixture
def lit_cnn_classifier(cnn_model, training_config, labels_mapping):
    return LitClassifier(
        model=cnn_model,
        model_name="cnn",
        training_cfg=training_config,
        labels_mapping=labels_mapping,
        using_wandb=False
    )


@pytest.fixture
def lit_cnn_lstm_classifier(cnn_lstm_model, training_config, labels_mapping):
    return LitClassifier(
        model=cnn_lstm_model,
        model_name="cnn_lstm",
        training_cfg=training_config,
        labels_mapping=labels_mapping,
        using_wandb=False
    )


def test_lit_classifier_initialization(mlp_model, training_config, labels_mapping):
    """Test that LitClassifier initializes correctly"""
    classifier = LitClassifier(
        model=mlp_model,
        model_name="test",
        training_cfg=training_config,
        labels_mapping=labels_mapping,
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
    assert output.shape == (batch_size, 2)  # num_classes = 2


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
