import torch.nn as nn

ACTIVATIONS = {
    "relu":      nn.ReLU,
    "leaky_relu": nn.LeakyReLU,
    "gelu":      nn.GELU,
    "tanh":      nn.Tanh,
    "none":      None,
}
