import torch
import torch.nn as nn

from src.models.activations import ACTIVATIONS
from src.models.normalization_layers import SequenceNorm1d


class GRU(nn.Module):
    def __init__(self, model_cfg, num_features, num_classes):
        super().__init__()

        self.model_cfg = model_cfg
        self.gru_activation = ACTIVATIONS[model_cfg.gru.activation]
        self.dense_activation = ACTIVATIONS[model_cfg.dense.activation]

        self.gru_layers = nn.ModuleList()
        self.gru_activations = nn.ModuleList()
        self.gru_normalization = nn.ModuleList()
        self.gru_drop_out = nn.ModuleList()
        gru_input_size = num_features

        for hidden_dim in model_cfg.gru.hidden_size:
            self.gru_layers.append(
                nn.GRU(
                    input_size=gru_input_size,
                    hidden_size=hidden_dim,
                    num_layers=1,
                    batch_first=True,
                    dropout=model_cfg.gru.dropout_rate if model_cfg.gru.dropout else 0,
                    bidirectional=model_cfg.gru.bidirectional
                )
            )
            if self.gru_activation:
                self.gru_activations.append(self.gru_activation())

            if model_cfg.gru.dropout:
                self.gru_drop_out.append(
                    nn.Dropout(model_cfg.gru.dropout_rate))

            self.gru_normalization.append(
                SequenceNorm1d(
                    dim=hidden_dim * (2 if model_cfg.gru.bidirectional else 1),
                    use_batch_norm=model_cfg.gru.batch_norm,
                    use_layer_norm=model_cfg.gru.layer_norm,
                )
            )

            gru_input_size = hidden_dim * \
                (2 if model_cfg.gru.bidirectional else 1)

        # After GRU, we'll take the last hidden state
        input_dim = model_cfg.gru.hidden_size[-1] * \
            (2 if model_cfg.gru.bidirectional else 1)

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
        # x shape: [batch, sequence_length, features]
        for i, gru in enumerate(self.gru_layers):
            x, _ = gru(x)  # x: [batch, sequence_length, hidden_dim_i]

            if i < len(self.gru_activations):
                x = self.gru_activations[i](x)

            if i < len(self.gru_drop_out):
                x = self.gru_drop_out[i](x)

            if len(self.gru_normalization) > i:
                x = self.gru_normalization[i](x)

        # Take the last time step
        x = x[:, -1, :]  # [batch, hidden_dim_last]
        return self.classifier(x)
