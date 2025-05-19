import torch.nn as nn
from collections import OrderedDict

from src.models.activations import ACTIVATIONS


class MLP(nn.Module):
    def __init__(self, model_cfg, num_features, num_classes):
        super().__init__()

        if model_cfg.input_layer_norm:
            self.input_norm = nn.LayerNorm(num_features)
        else:
            self.input_norm = nn.Identity()

        self.dense_activation = ACTIVATIONS[model_cfg.dense.activation]

        input_dim = num_features
        layers = OrderedDict()
        idx = 0
        for i, hidden_dim in enumerate(model_cfg.dense.units, start=1):
            layers[f"linear{i}"] = nn.Linear(input_dim, hidden_dim)
            idx += 1

            layers[f"act{i}"] = self.dense_activation()

            if model_cfg.dense.batch_norm:
                layers[f"bn{i}"] = nn.BatchNorm1d(hidden_dim)
                idx += 1

            if model_cfg.dense.layer_norm:
                layers[f"ln{i}"] = nn.LayerNorm(hidden_dim)
                idx += 1

            if model_cfg.dense.dropout:
                layers[f"drop{i}"] = nn.Dropout(
                    model_cfg.dense.dropout_rate)
                idx += 1

            input_dim = hidden_dim
        # fc_layers .append(nn.Linear(input_dim, num_classes))
        layers["out"] = nn.Linear(input_dim, num_classes)

        self.network = nn.Sequential(layers)

        # for i, hidden_dim in enumerate(model_cfg.dense.units, start=1):
        #     layers.append(nn.Linear(input_dim, hidden_dim))
        #     layers.append(self.dense_activation())
        #     if model_cfg.dense.batch_norm:
        #         layers.append(nn.BatchNorm1d(hidden_dim))
        #     if model_cfg.dense.layer_norm:
        #         layers.append(nn.LayerNorm(hidden_dim))
        #     if model_cfg.dense.dropout:
        #         layers.append(nn.Dropout(model_cfg.dense.dropout_rate))
        #     input_dim = hidden_dim

        # layers.append(nn.Linear(input_dim, num_classes))
        # self.network = nn.Sequential(*layers)

    def forward(self, x):
        x = self.input_norm(x)
        return self.network(x)
