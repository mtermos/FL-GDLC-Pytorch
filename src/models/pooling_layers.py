import torch.nn as nn


def pooling_layer(pooling_type: str):
    if pooling_type == "max":
        return nn.MaxPool1d
    if pooling_type == "avg":
        return nn.AvgPool1d
    if pooling_type == "adaptive_max":
        return nn.AdaptiveMaxPool1d
    if pooling_type == "adaptive_avg":
        return nn.AdaptiveAvgPool1d
