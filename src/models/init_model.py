from src.models.mlp import MLP
from src.models.cnn import CNN
from src.models.cnn_lstm import CNNLSTM
from src.models.pl_model import LitClassifier


def init_model(training_cfg, model_cfg, num_features, labels_mapping, weight_tensor, using_wandb):

    num_classes = 2
    if training_cfg.multi_class:
        num_classes = len(labels_mapping)

    if model_cfg.model.type == "mlp":
        model = MLP(model_cfg, num_features, num_classes)
    elif model_cfg.model.type == "cnn":
        model = CNN(model_cfg, num_features, num_classes)
    elif model_cfg.model.type == "cnn_lstm":
        model = CNNLSTM(model_cfg, num_features, num_classes)

    return LitClassifier(model, model_cfg.model.type, training_cfg, labels_mapping, weight_tensor, using_wandb)
