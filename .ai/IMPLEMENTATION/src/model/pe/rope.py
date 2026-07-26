import torch
import torch.nn as nn
from . import PositionalEncoding, register_pe


def _precompute_freqs(d_head, max_seq_len, base=10000.0):
    freqs = 1.0 / (base ** (torch.arange(0, d_head, 2).float() / d_head))
    t = torch.arange(max_seq_len)
    freqs = torch.outer(t, freqs)
    return freqs


def _apply_rotate(x, freqs):
    B, H, L, D = x.shape
    x_reshaped = x.float().reshape(B, H, L, D // 2, 2)
    x_complex = torch.view_as_complex(x_reshaped)
    freqs = freqs[:L, :].to(x.device)
    freqs_complex = torch.polar(torch.ones_like(freqs), freqs)
    freqs_complex = freqs_complex.view(1, 1, L, D // 2)
    x_rotated = x_complex * freqs_complex
    x_out = torch.view_as_real(x_rotated).reshape(B, H, L, D)
    return x_out.to(x.dtype)


def _apply_rotate_with_scale(x, freqs, scale):
    B, H, L, D = x.shape
    x_reshaped = x.float().reshape(B, H, L, D // 2, 2)
    x_complex = torch.view_as_complex(x_reshaped)
    freqs_scaled = freqs[:L, :] * scale[:L, :]
    freqs_complex = torch.polar(torch.ones_like(freqs_scaled), freqs_scaled)
    freqs_complex = freqs_complex.view(1, 1, L, D // 2).to(x.device)
    x_rotated = x_complex * freqs_complex
    x_out = torch.view_as_real(x_rotated).reshape(B, H, L, D)
    return x_out.to(x.dtype)


@register_pe("rope")
class RotaryPositionalEncoding(PositionalEncoding):
    def __init__(self, d_model, n_heads, max_seq_len, base=10000.0):
        super().__init__()
        self.d_head = d_model // n_heads
        self.n_heads = n_heads
        self.max_eval_len = 8192
        self.base = base
        freqs = _precompute_freqs(self.d_head, self.max_eval_len, base)
        self.register_buffer("freqs", freqs)

    def forward(self, x, past_length=0):
        return x  # RoPE is applied inside attention, not here

    def rotate_qk(self, q, k, position_ids=None):
        L = q.shape[2]
        freqs = self.freqs[:L].to(q.device)
        q_rot = _apply_rotate(q, freqs)
        k_rot = _apply_rotate(k, freqs)
        return q_rot, k_rot


@register_pe("position_interpolation")
class PositionInterpolation(RotaryPositionalEncoding):
    def __init__(self, d_model, n_heads, max_seq_len, base=10000.0, scale_factor=1.0):
        super().__init__(d_model, n_heads, max_seq_len, base)
        self.scale_factor = scale_factor

    def set_scale(self, train_len, eval_len):
        self.scale_factor = eval_len / train_len

    def rotate_qk(self, q, k, position_ids=None):
        L = q.shape[2]
        freqs = self.freqs[:L].to(q.device)
        scale = torch.ones(L, freqs.shape[1], device=q.device) / self.scale_factor
        q_rot = _apply_rotate_with_scale(q, freqs, scale)
        k_rot = _apply_rotate_with_scale(k, freqs, scale)
        return q_rot, k_rot
