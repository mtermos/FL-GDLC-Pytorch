import torch
import torch.nn as nn

from src.models.activations import ACTIVATIONS
from src.models.normalization_layers import SequenceNorm1d


class LSTM(nn.Module):
    def __init__(self, model_cfg, num_features, num_classes):
        super().__init__()

        self.model_cfg = model_cfg
        self.lstm_activation = ACTIVATIONS[model_cfg.lstm.activation]
        self.dense_activation = ACTIVATIONS[model_cfg.dense.activation]

        self.lstm_layers = nn.ModuleList()
        self.lstm_activations = nn.ModuleList()
        self.lstm_normalization = nn.ModuleList()
        self.lstm_drop_out = nn.ModuleList()
        lstm_input_size = num_features

        for hidden_dim in model_cfg.lstm.hidden_size:
            self.lstm_layers.append(
                nn.LSTM(
                    input_size=lstm_input_size,
                    hidden_size=hidden_dim,
                    num_layers=1,
                    batch_first=True,
                    dropout=model_cfg.lstm.dropout_rate if model_cfg.lstm.dropout else 0,
                    bidirectional=model_cfg.lstm.bidirectional
                )
            )
            if self.lstm_activation:
                self.lstm_activations.append(self.lstm_activation())

            if model_cfg.lstm.dropout:
                self.lstm_drop_out.append(
                    nn.Dropout(model_cfg.lstm.dropout_rate))

            self.lstm_normalization.append(
                SequenceNorm1d(
                    dim=hidden_dim *
                    (2 if model_cfg.lstm.bidirectional else 1),
                    use_batch_norm=model_cfg.lstm.batch_norm,
                    use_layer_norm=model_cfg.lstm.layer_norm,
                )
            )

            lstm_input_size = hidden_dim * \
                (2 if model_cfg.lstm.bidirectional else 1)

        # After LSTM, we'll take the last hidden state
        input_dim = model_cfg.lstm.hidden_size[-1] * \
            (2 if model_cfg.lstm.bidirectional else 1)

        fc_layers = []
        for hidden_dim in model_cfg.dense.units:
            fc_layers.append(nn.Linear(input_dim, hidden_dim))
            fc_layers.append(self.dense_activation())
            if model_cfg.dense.batch_norm:
                fc_layers.append(nn.BatchNorm1d(hidden_dim))
            if model_cfg.dense.layer_norm:
                fc_layers.append(nn.LayerNorm(hidden_dim))
            if model_cfg.dense.dropout:
                fc_layers.append(nn.Dropout(model_cfg.dense.dropout_rate))
            input_dim = hidden_dim

        fc_layers.append(nn.Linear(input_dim, num_classes))

        self.classifier = nn.Sequential(*fc_layers)

    def forward(self, x):
        print(f"==>> x.shape: {x.shape}")
        # x shape: [batch, sequence_length, features]
        for i, lstm in enumerate(self.lstm_layers):
            x, _ = lstm(x)  # x: [batch, sequence_length, hidden_dim_i]

            if i < len(self.lstm_activations):
                x = self.lstm_activations[i](x)

            if i < len(self.lstm_drop_out):
                x = self.lstm_drop_out[i](x)

            if len(self.lstm_normalization) > i:
                x = self.lstm_normalization[i](x)

        # Take the last time step
        x = x[:, -1, :]  # [batch, hidden_dim_last]
        res = self.classifier(x)
        print(f"==>> res: {res}")
        return res
