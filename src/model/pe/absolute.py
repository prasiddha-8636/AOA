import torch
import torch.nn as nn
from . import PositionalEncoding, register_pe


@register_pe("learned")
class LearnedPositionalEncoding(PositionalEncoding):
    def __init__(self, d_model, n_heads, max_seq_len):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.embedding = nn.Embedding(max_seq_len, d_model)

    def forward(self, x, past_length=0):
        B, L, D = x.shape
        device = x.device
        # Clamp positions to training range; unseen positions get zero
        pos = torch.arange(past_length, past_length + L, device=device)
        pos = pos.clamp(0, self.max_seq_len - 1)
        pe = self.embedding(pos).unsqueeze(0)
        return x + pe


@register_pe("sinusoidal")
class SinusoidalPositionalEncoding(PositionalEncoding):
    def __init__(self, d_model, n_heads, max_seq_len):
        super().__init__()
        self.d_model = d_model
        self.max_eval_len = 8192  # max length we evaluate at
        pe = self._build_pe(self.max_eval_len, d_model)
        self.register_buffer("pe", pe)

    def _build_pe(self, L, D):
        pe = torch.zeros(L, D)
        position = torch.arange(0, L).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, D, 2).float() * -(torch.log(torch.tensor(10000.0)) / D)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    def _grow(self, L, device):
        """Grow the cached table to L rows on demand (needle generation reaches
        L=8193, one past the eval max)."""
        extra = self._build_pe(L - self.pe.shape[0], self.d_model)
        self.pe = torch.cat([self.pe, extra], dim=0)

    def forward(self, x, past_length=0):
        B, L, D = x.shape
        device = x.device
        if past_length + L > self.pe.shape[0]:
            self._grow(past_length + L, device)
        return x + self.pe[past_length : past_length + L, :].to(device)
