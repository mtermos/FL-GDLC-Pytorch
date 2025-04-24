import torch
import torch.nn as nn


class ChannelLayerNorm(nn.Module):
    """LayerNorm over the CHANNEL dimension of a 1D conv output."""

    def __init__(self, num_channels, eps: float = 1e-5, elementwise_affine: bool = True):
        super().__init__()
        # This LN will normalize the last dim (channels after we transpose)
        self.ln = nn.LayerNorm(num_channels, eps=eps,
                               elementwise_affine=elementwise_affine)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, L) → (B, L, C)
        x = x.transpose(1, 2)
        x = self.ln(x)            # norm across C
        return x.transpose(1, 2)  # back to (B, C, L)


class SequenceNorm1d(nn.Module):
    def __init__(
        self,
        dim: int,
        use_batch_norm: bool = False,
        use_layer_norm: bool = False,
        **bn_kwargs,
    ):
        super().__init__()
        if use_batch_norm:
            # BN over the feature dimension: expect (B, D, L)
            self.norm = nn.BatchNorm1d(dim, **bn_kwargs)
            self.is_bn = True
        elif use_layer_norm:
            # LN over the feature dimension: expect (B, L, D)
            self.norm = nn.LayerNorm(dim)
            self.is_bn = False
        else:
            self.norm = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, D)
        if self.norm is None:
            return x
        if self.is_bn:
            # shuffle so BN sees (B, D, L)
            x = x.transpose(1, 2)
            x = self.norm(x)
            return x.transpose(1, 2)
        else:
            # LN works in‐place on last dim
            return self.norm(x)
