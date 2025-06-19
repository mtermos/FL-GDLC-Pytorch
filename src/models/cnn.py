import torch
import torch.nn as nn
from collections import OrderedDict

from src.models.activations import ACTIVATIONS
from src.models.pooling_layers import pooling_layer
from src.models.normalization_layers import ChannelLayerNorm


class CNN(nn.Module):
    def __init__(self, model_cfg, num_features, num_classes):
        super().__init__()

        if model_cfg.input_layer_norm:
            self.input_norm = nn.LayerNorm(num_features)
        else:
            self.input_norm = nn.Identity()

        self.cnn_activation = ACTIVATIONS[model_cfg.cnn.activation]
        self.dense_activation = ACTIVATIONS[model_cfg.dense.activation]

        conv_layers = []

        in_channels = 1
        for i, filter in enumerate(model_cfg.cnn.filters):
            k = model_cfg.cnn.kernel_sizes[i]
            cnn = nn.Conv1d(
                in_channels=in_channels,
                out_channels=filter,
                kernel_size=k,
                padding=(k - 1) // 2
            )
            conv_layers.append(cnn)
            conv_layers.append(self.cnn_activation())

            if i == len(model_cfg.cnn.filters) - 1:
                pooling_type = model_cfg.cnn.last_layer_pooling_type
            else:
                pooling_type = model_cfg.cnn.pooling_type

            if pooling_type.startswith("adaptive"):
                conv_layers.append(pooling_layer(pooling_type)(output_size=1))
            else:
                conv_layers.append(pooling_layer(pooling_type)(kernel_size=2))

            if model_cfg.cnn.batch_norm:
                conv_layers.append(nn.BatchNorm1d(filter))
            if model_cfg.cnn.layer_norm:
                conv_layers.append(ChannelLayerNorm(filter))
            if model_cfg.cnn.dropout:
                conv_layers.append(nn.Dropout(model_cfg.cnn.dropout_rate))
            in_channels = filter

        self.features = nn.Sequential(*conv_layers)
        with torch.no_grad():
            # make a dummy of shape (1, channels, length)
            dummy = torch.zeros(1, 1, num_features)
            feat = self.features(dummy)
            # feat.shape -> [1, C_last, L_final]
            input_dim = feat.size(1) * feat.size(2)

        # print(f"==>> input_dim: {input_dim}")

        # fc_layers = []
        # for hidden_dim in model_cfg.dense.units:
        #     fc_layers.append(nn.Linear(input_dim, hidden_dim))
        #     fc_layers.append(self.dense_activation())
        #     if model_cfg.dense.batch_norm:
        #         fc_layers.append(nn.BatchNorm1d(hidden_dim))
        #     if model_cfg.dense.layer_norm:
        #         fc_layers.append(nn.LayerNorm(hidden_dim))
        #     if model_cfg.dense.dropout:
        #         fc_layers.append(nn.Dropout(model_cfg.dense.dropout_rate))
        #     input_dim = hidden_dim

        fc_layers = OrderedDict()
        idx = 0
        for i, hidden_dim in enumerate(model_cfg.dense.units, start=1):
            fc_layers[f"linear{i}"] = nn.Linear(input_dim, hidden_dim)
            idx += 1

            fc_layers[f"act{i}"] = self.dense_activation()

            if model_cfg.dense.batch_norm:
                fc_layers[f"bn{i}"] = nn.BatchNorm1d(hidden_dim)
                idx += 1

            if model_cfg.dense.layer_norm:
                fc_layers[f"ln{i}"] = nn.LayerNorm(hidden_dim)
                idx += 1

            if model_cfg.dense.dropout:
                fc_layers[f"drop{i}"] = nn.Dropout(
                    model_cfg.dense.dropout_rate)
                idx += 1

            input_dim = hidden_dim
        # fc_layers .append(nn.Linear(input_dim, num_classes))
        fc_layers["out"] = nn.Linear(input_dim, num_classes)

        self.classifier = nn.Sequential(fc_layers)
        # self.classifier = nn.Sequential(*fc_layers)

    def forward(self, x):
        x = self.input_norm(x)
        x = x.view(x.size(0), 1, x.size(1))
        # print(f"==>> x.shape: {x.shape}")
        x = self.features(x)
        x = x.flatten(1)
        return self.classifier(x)
