from src.models.mlp import MLP
from src.models.cnn import CNN
from src.models.cnn_lstm import CNNLSTM
from src.models.pl_model import LitClassifier


def init_model(training_cfg, model_cfg, input_shape, labels_mapping, using_wandb):

    num_classes = 2
    if training_cfg.multi_class:
        num_classes = len(labels_mapping)

    if model_cfg.model.type == "mlp":
        model = MLP(model_cfg, input_shape, num_classes)
    elif model_cfg.model.type == "cnn":
        model = CNN(model_cfg, input_shape, num_classes)
    elif model_cfg.model.type == "cnn_lstm":
        model = CNNLSTM(model_cfg, input_shape, num_classes)

    return LitClassifier(model, model_cfg.model.type, training_cfg, labels_mapping, using_wandb)
