import torch
import torch.nn as nn
import torch.nn.functional as F
from . import PositionalEncoding, register_pe


@register_pe("kerple")
class KERPLEPositionalEncoding(PositionalEncoding):
    def __init__(self, d_model, n_heads, max_seq_len, variant="log"):
        super().__init__()
        self.n_heads = n_heads
        self.variant = variant

        # Log variant: γ_h per head (scalar)
        # Power variant: γ_h, κ_h per head
        if variant == "log":
            self.gamma = nn.Parameter(torch.zeros(n_heads))
        elif variant == "power":
            self.gamma = nn.Parameter(torch.zeros(n_heads))
            self.kappa = nn.Parameter(torch.ones(n_heads))
        else:
            raise ValueError(f"Unknown KERPLE variant: {variant}")

    def get_bias(self, L, device):
        positions = torch.arange(L, device=device).view(1, 1, L, 1)
        dist = (positions - positions.transpose(-1, -2)).abs()

        gamma = torch.sigmoid(self.gamma).view(1, -1, 1, 1)

        if self.variant == "log":
            bias = -gamma * torch.log(1 + dist)
        elif self.variant == "power":
            kappa = torch.sigmoid(self.kappa - 2) + 1  # range [1, 2]
            bias = -gamma * (dist ** kappa.view(1, -1, 1, 1))

        return bias

    def forward(self, x, past_length=0):
        return x
