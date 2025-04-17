import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, model_cfg, num_features, num_classes):
        super().__init__()

        if model_cfg.dense.activation == "relu":
            self.activation = nn.ReLU()
        elif model_cfg.dense.activation == "leaky_relu":
            self.activation = nn.LeakyReLU()

        layers = []
        input_dim = num_features

        for hidden_dim in model_cfg.dense.units:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(self.activation)
            if model_cfg.dense.batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            if model_cfg.dense.dropout:
                layers.append(nn.Dropout(model_cfg.dense.dropout_rate))
            input_dim = hidden_dim

        layers.append(nn.Linear(input_dim, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
