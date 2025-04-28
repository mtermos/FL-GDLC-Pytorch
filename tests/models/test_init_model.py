import pytest
from src.models.init_model import init_model
from src.models.mlp import MLP
from src.models.cnn import CNN
from src.models.cnn_lstm import CNNLSTM
from src.models.pl_model import LitClassifier


@pytest.fixture
def labels_mapping():
    return {0: "class_0", 1: "class_1"}


def test_init_mlp_model(labels_mapping, mock_base_cfg, mock_mlp_cfg):
    """Test initialization of MLP model"""
    input_shape = 10

    model = init_model(mock_base_cfg.training, mock_mlp_cfg,
                       input_shape, labels_mapping, False)

    assert isinstance(model, LitClassifier)
    assert isinstance(model.model, MLP)
    assert model.model_name == "mlp"


def test_init_cnn_model(labels_mapping, mock_base_cfg, mock_cnn_cfg):
    """Test initialization of CNN model"""
    input_shape = 100

    model = init_model(mock_base_cfg.training, mock_cnn_cfg,
                       input_shape, labels_mapping, False)

    assert isinstance(model, LitClassifier)
    assert isinstance(model.model, CNN)
    assert model.model_name == "cnn"


def test_init_cnn_lstm_model(labels_mapping, mock_base_cfg, mock_cnn_lstm_cfg):
    """Test initialization of CNN-LSTM model"""
    input_shape = 100

    model = init_model(mock_base_cfg.training, mock_cnn_lstm_cfg,
                       input_shape, labels_mapping, False)

    assert isinstance(model, LitClassifier)
    assert isinstance(model.model, CNNLSTM)
    assert model.model_name == "cnn_lstm"


def test_multi_class_model(labels_mapping, mock_base_cfg, mock_mlp_cfg):
    """Test initialization with multi-class configuration"""
    mock_base_cfg.training.multi_class = True
    input_shape = 10

    model = init_model(mock_base_cfg.training, mock_mlp_cfg,
                       input_shape, labels_mapping, False)

    assert isinstance(model, LitClassifier)
    assert isinstance(model.model, MLP)
    assert model.model_name == "mlp"
