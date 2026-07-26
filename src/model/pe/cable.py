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

    def get_bias(self, q, k):
        B, H, L, D = q.shape
        q_exp = q.unsqueeze(-2).expand(-1, -1, -1, L, -1)
        k_exp = k.unsqueeze(-3).expand(-1, -1, L, -1, -1)

        qk_concat = torch.cat([q_exp, k_exp], dim=-1)
        bias_logit = self.bias_net(qk_concat).squeeze(-1)

        # Add distance-based decay
        positions = torch.arange(L, device=q.device).view(1, 1, L, 1)
        dist = (positions - positions.transpose(-1, -2)).abs()
        dist_bias = -0.1 * dist.unsqueeze(1)

        return bias_logit + dist_bias.expand_as(bias_logit)

    def forward(self, x, past_length=0):
        return x
