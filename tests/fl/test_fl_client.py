import pytest
import torch
import numpy as np
import pytorch_lightning as pl
from src.fl.fl_client import FLClient
from unittest.mock import MagicMock, patch


class MockDataModule:
    def __init__(self, train_size=100, val_size=20):
        self.train_dataset = torch.utils.data.TensorDataset(
            torch.randn(train_size, 10),
            torch.randint(0, 3, (train_size,))
        )
        self.val_dataset = torch.utils.data.TensorDataset(
            torch.randn(val_size, 10),
            torch.randint(0, 3, (val_size,))
        )

    def setup(self):
        pass


class MockModel:
    def __init__(self):
        self.alpha = 0.01
        self.state_dict = lambda: {"layer1.weight": torch.randn(10, 10)}

    def get_parameters(self):
        return [torch.randn(10, 10).numpy()]

    def set_parameters(self, parameters):
        pass


@pytest.fixture
def mock_data_module():
    return MockDataModule()


@pytest.fixture
def mock_model():
    return MockModel()


@pytest.fixture
def mock_logger():
    logger = MagicMock()
    logger.log_metrics = MagicMock()
    return logger


@pytest.fixture
def fl_client(mock_data_module, mock_model, mock_logger):
    return FLClient(
        data_module=mock_data_module,
        model=mock_model,
        logger=mock_logger,
        logger_type="wandb",
        num_local_epochs=1
    )


def test_get_parameters(fl_client):
    parameters = fl_client.get_parameters({})
    assert isinstance(parameters, list)
    assert all(isinstance(p, np.ndarray) for p in parameters)


def test_set_parameters(fl_client):
    parameters = [torch.randn(10, 10).numpy()]
    fl_client.set_parameters(parameters)  # Should not raise any errors


@patch('pytorch_lightning.Trainer.fit')
@patch('pytorch_lightning.Trainer.validate')
def test_fit(mock_validate, mock_fit, fl_client):
    # Mock the trainer's callback_metrics
    fl_client.train_trainer.callback_metrics.clear()
    fl_client.train_trainer.callback_metrics.update({
        'train_loss':    torch.tensor(0.5),
        'train_f1_score': torch.tensor(0.8),
        'val_loss':      torch.tensor(0.4),
        'val_acc':       torch.tensor(0.9),
        'val_f1_score':  torch.tensor(0.85),
    })

    parameters = [torch.randn(10, 10).numpy()]
    config = {
        'lr': 0.01,
        'server_round': 1
    }

    parameters_prime, num_examples, metrics = fl_client.fit(parameters, config)

    assert isinstance(parameters_prime, list)
    assert num_examples == len(fl_client.data_module.train_dataset)
    assert isinstance(metrics, dict)
    assert 'train_loss' in metrics
    assert 'train_f1_score' in metrics
    assert metrics['round'] == 1
    assert metrics['lr'] == 0.01


@patch('pytorch_lightning.Trainer.validate')
def test_evaluate(mock_validate, fl_client):
    # Mock the validation results
    mock_validate.return_value = [{
        'val_loss': torch.tensor(0.4),
        'val_acc': torch.tensor(0.9),
        'val_f1_score': torch.tensor(0.85)
    }]

    parameters = [torch.randn(10, 10).numpy()]
    loss, num_examples, metrics = fl_client.evaluate(parameters, {})

    assert isinstance(loss, float)
    assert num_examples == len(fl_client.data_module.val_dataset)
    assert isinstance(metrics, dict)
    assert 'val_loss' in metrics
    assert 'val_accuracy' in metrics
    assert 'val_f1s' in metrics


def test_evaluate_no_validation(fl_client):
    # Set do_validate to False
    fl_client.data_module.do_validate = False

    parameters = [torch.randn(10, 10).numpy()]
    loss, num_examples, metrics = fl_client.evaluate(parameters, {})

    assert loss == 0.0
    assert num_examples == 0
    assert metrics == {}
