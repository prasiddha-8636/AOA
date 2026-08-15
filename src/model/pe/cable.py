import torch
import torch.nn as nn
import torch.nn.functional as F
from . import PositionalEncoding, register_pe


@register_pe("cable")
class CABLEPositionalEncoding(PositionalEncoding):
    def __init__(self, d_model, n_heads, max_seq_len):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.hidden_dim = 16
        # Per-head context-aware bias network
        self.bias_net = nn.Sequential(
            nn.Linear(self.d_head * 2, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1),
        )
        # Chunk size for query dimension to bound peak memory on small GPUs
        self._chunk = 64

    def get_bias(self, q, k):
        B, H, L, D = q.shape
        device = q.device
        positions = torch.arange(L, device=device).view(1, 1, L, 1)
        dist = (positions - positions.transpose(-1, -2)).abs()
        dist_bias = -0.1 * dist.expand(B, H, L, L)
        bias = torch.zeros(B, H, L, L, device=device)
        for i0 in range(0, L, self._chunk):
            i1 = min(i0 + self._chunk, L)
            q_exp = q[:, :, i0:i1].unsqueeze(-2).expand(-1, -1, -1, L, -1)
            k_exp = k.unsqueeze(-3).expand(-1, -1, i1 - i0, -1, -1)
            qk_concat = torch.cat([q_exp, k_exp], dim=-1)
            bias[:, :, i0:i1] = self.bias_net(qk_concat).squeeze(-1)
        return bias + dist_bias

    def forward(self, x, past_length=0):
        return x
