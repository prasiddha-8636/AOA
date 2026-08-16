import torch
import torch.nn as nn
from . import PositionalEncoding, register_pe


@register_pe("alibi")
class AlibiPositionalEncoding(PositionalEncoding):
    def __init__(self, d_model, n_heads, max_seq_len):
        super().__init__()
        self.n_heads = n_heads
        self.max_eval_len = 8192
        slopes = self._compute_slopes(n_heads)
        bias = self._build_alibi_bias(self.max_eval_len, slopes)
        self.register_buffer("bias", bias)

    def _compute_slopes(self, n_heads):
        s = 2 ** (-8 / n_heads)
        slopes = torch.tensor([s ** (i + 1) for i in range(n_heads)])
        return slopes

    def _build_alibi_bias(self, L, slopes):
        pos = torch.arange(L).view(1, 1, L, 1) - torch.arange(L).view(1, 1, 1, L)
        pos = pos.abs()
        bias = -slopes.view(1, -1, 1, 1) * pos
        return bias

    def _grow(self, L, device):
        """Grow the cached bias matrix on demand (needle generation reaches
        L=8193, one past the eval max)."""
        slopes = self._compute_slopes(self.n_heads)
        self.bias = self._build_alibi_bias(L, slopes)

    def forward(self, x, past_length=0):
        return x  # ALiBi bias is added during attention

    def get_bias(self, L, device):
        if L > self.bias.shape[2]:
            self._grow(L, device)
        return self.bias[:, :, :L, :L].to(device)
