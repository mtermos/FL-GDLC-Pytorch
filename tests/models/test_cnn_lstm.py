# import torch
# import pytest
# from src.models.cnn_lstm import CNNLSTM


# class MockModelConfig:
#     def __init__(self):
#         self.cnn = type('CNNConfig', (), {
#             'activation': 'relu',
#             'filters': [32, 64],
#             'kernel_sizes': [3, 3],
#             'batch_norm': True,
#             'dropout': True,
#             'dropout_rate': 0.2
#         })()
#         self.lstm = type('LSTMConfig', (), {
#             'activation': 'tanh',
#             'hidden_size': [128, 64],
#             'dropout': True,
#             'dropout_rate': 0.2
#         })()
#         self.dense = type('DenseConfig', (), {
#             'activation': 'relu',
#             'units': [32],
#             'batch_norm': True,
#             'dropout': True,
#             'dropout_rate': 0.2
#         })()


# @pytest.fixture
# def model_config():
#     return MockModelConfig()


# @pytest.fixture
# def cnn_lstm_model(model_config):
#     return CNNLSTM(model_config, num_features=100, num_classes=2)


# def test_cnn_lstm_initialization(model_config):
#     """Test that CNNLSTM initializes correctly with given configuration"""
#     model = CNNLSTM(model_config, num_features=100, num_classes=2)
#     assert isinstance(model, CNNLSTM)
#     assert isinstance(model.features, torch.nn.Sequential)
#     assert isinstance(model.lstm_layers, torch.nn.ModuleList)
#     assert isinstance(model.classifier, torch.nn.Sequential)


# def test_cnn_lstm_forward_pass(cnn_lstm_model):
#     """Test that forward pass works with correct input shape"""
#     batch_size = 5
#     num_features = 100
#     x = torch.randn(batch_size, num_features)
#     output = cnn_lstm_model(x)

#     assert output.shape == (batch_size, 2)  # num_classes = 2
#     assert not torch.isnan(output).any()
#     assert not torch.isinf(output).any()


# def test_cnn_lstm_input_reshape(cnn_lstm_model):
#     """Test that input is correctly reshaped for CNN and LSTM layers"""
#     batch_size = 5
#     num_features = 100
#     x = torch.randn(batch_size, num_features)

#     # Check CNN input reshape
#     x_cnn = x.view(x.size(0), 1, x.size(1))
#     assert x_cnn.shape == (batch_size, 1, num_features)

#     # Check LSTM input shape after CNN
#     x_cnn = cnn_lstm_model.features(x_cnn)
#     x_lstm = x_cnn.permute(0, 2, 1)
#     assert x_lstm.shape[0] == batch_size
#     assert x_lstm.shape[2] == 64  # last CNN filter size


# def test_cnn_lstm_layers(cnn_lstm_model):
#     """Test that the model has the correct number of layers"""
#     # Count the number of layers in features
#     num_feature_layers = len(list(cnn_lstm_model.features))
#     # Expected layers in features: 2 Conv1d + 2 ReLU + 2 BatchNorm + 2 Dropout
#     assert num_feature_layers == 8

#     # Check LSTM layers
#     assert len(cnn_lstm_model.lstm_layers) == 2
#     assert len(cnn_lstm_model.lstm_activations) == 2

#     # Count the number of layers in classifier
#     num_classifier_layers = len(list(cnn_lstm_model.classifier))
#     # Expected layers in classifier: 1 Linear + 1 ReLU + 1 BatchNorm + 1 Dropout + 1 Linear
#     assert num_classifier_layers == 5


# def test_cnn_lstm_device(cnn_lstm_model):
#     """Test that model can be moved to different devices"""
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     cnn_lstm_model.to(device)
#     assert next(cnn_lstm_model.parameters()).device == device
