import torch
import torch.nn as nn

from src.models.activations import ACTIVATIONS
from src.models.normalization_layers import ChannelLayerNorm, SequenceNorm1d


class CNNLSTM(nn.Module):
    def __init__(self, model_cfg, num_features, num_classes):
        super().__init__()

        self.model_cfg = model_cfg

        self.cnn_activation = ACTIVATIONS[model_cfg.cnn.activation]
        self.lstm_activation = ACTIVATIONS[model_cfg.lstm.activation]
        self.dense_activation = ACTIVATIONS[model_cfg.dense.activation]

        conv_layers = []

        in_channels = 1
        for i, filter in enumerate(model_cfg.cnn.filters):
            cnn = nn.Conv1d(
                in_channels=in_channels,
                out_channels=filter,
                kernel_size=model_cfg.cnn.kernel_sizes[i],
                padding=1
            )
            conv_layers.append(cnn)
            conv_layers.append(self.cnn_activation())
            if model_cfg.cnn.batch_norm:
                conv_layers.append(nn.BatchNorm1d(filter))
            if model_cfg.cnn.layer_norm:
                conv_layers.append(ChannelLayerNorm(filter))
            if model_cfg.cnn.dropout:
                conv_layers.append(nn.Dropout(model_cfg.cnn.dropout_rate))
            in_channels = filter

        self.features = nn.Sequential(*conv_layers)

        self.lstm_layers = nn.ModuleList()
        self.lstm_activations = nn.ModuleList()
        self.lstm_normalization = nn.ModuleList()
        lstm_input_size = filter

        for hidden_dim in model_cfg.lstm.hidden_size:
            self.lstm_layers.append(
                nn.LSTM(
                    input_size=lstm_input_size,
                    hidden_size=hidden_dim,
                    num_layers=1,
                    batch_first=True,
                    dropout=model_cfg.lstm.dropout_rate if model_cfg.lstm.dropout else 0,
                    bidirectional=False
                )
            )
            if self.lstm_activation:
                self.lstm_activations.append(self.lstm_activation())

            self.lstm_normalization.append(
                SequenceNorm1d(
                    dim=hidden_dim,
                    use_batch_norm=model_cfg.lstm.batch_norm,
                    use_layer_norm=model_cfg.lstm.layer_norm,
                    # e.g. momentum=0.1, eps=1e-5 if you want custom BN args
                )
            )

            lstm_input_size = hidden_dim

        # after LSTM, we'll take the last hidden‐state, so our
        # `input_dim` for the dense layers is just the last hidden_dim
        input_dim = model_cfg.lstm.hidden_size[-1]

        fc_layers = []
        for hidden_dim in model_cfg.dense.units:
            fc_layers .append(nn.Linear(input_dim, hidden_dim))
            fc_layers .append(self.dense_activation())
            if model_cfg.dense.batch_norm:
                fc_layers .append(nn.BatchNorm1d(hidden_dim))
            if model_cfg.dense.layer_norm:
                fc_layers.append(nn.LayerNorm(hidden_dim))
            if model_cfg.dense.dropout:
                fc_layers .append(nn.Dropout(model_cfg.dense.dropout_rate))
            input_dim = hidden_dim

        fc_layers .append(nn.Linear(input_dim, num_classes))

        self.classifier = nn.Sequential(*fc_layers)

    def forward(self, x):
        x = x.view(x.size(0), 1, x.size(1))
        # → [batch, L_final, C_last]
        x = self.features(x)
        x = x.permute(0, 2, 1)

        # pass through each LSTM layer
        for i, lstm in enumerate(self.lstm_layers):
            x, _ = lstm(x)                    # x: [batch, L, hidden_dim_i]

            if i < len(self.lstm_activations):
                x = self.lstm_activations[i](x)

            if len(self.lstm_normalization) > i:
                x = self.lstm_normalization[i](x)

        # grab last time step
        x = x[:, -1, :]                       # → [batch, hidden_dim_last]
        return self.classifier(x)
