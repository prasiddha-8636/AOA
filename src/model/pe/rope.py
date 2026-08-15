import math
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
    """Position Interpolation (Chen et al., 2023). Compresses position indices by
    scale_factor = eval_len / train_len, mapping the extended range back into the
    trained range. Defaults to 16x (512 -> 8192); eval scripts call set_scale
    explicitly. Used as an inference-time extension of a vanilla RoPE checkpoint."""

    def __init__(self, d_model, n_heads, max_seq_len, base=10000.0, scale_factor=16.0):
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


@register_pe("yarn")
class YarnPositionalEncoding(PositionalEncoding):
    """YaRN (Peng et al., 2023): NTK-by-parts interpolation combined with attention
    scaling. Implements the reference formulas:
      * inv_freq blend over dimensions between r=32 (no interpolation) and r=1 (full
        interpolation), with alpha=1, beta=32, s = ratio = eval_len/train_len.
      * attention factor sqrt(1/t) = 0.1*ln(s) + 1 applied to both q and k, which
        scales attention logits by 1/t without touching the attention code.
    Used as an inference-time extension of a vanilla RoPE checkpoint; the frequency
    buffer is stored under a distinct name so the checkpoint's vanilla RoPE buffer
    is ignored when loading with strict=False."""

    def __init__(self, d_model, n_heads, max_seq_len, base=10000.0, ratio=16.0):
        super().__init__()
        self.d_head = d_model // n_heads
        self.n_heads = n_heads
        self.max_eval_len = 8192
        self.base = base
        self.ratio = ratio
        attn_scale = 0.1 * math.log(ratio) + 1.0
        inv_freq = _yarn_inv_freq(self.d_head, self.max_eval_len, base, ratio)
        freqs = torch.outer(torch.arange(self.max_eval_len), inv_freq)
        self.register_buffer("freqs_yarn", freqs)
        self.register_buffer(
            "attn_scale", torch.tensor(attn_scale, dtype=torch.float32)
        )

    def set_ratio(self, train_len, eval_len):
        ratio = eval_len / train_len
        if ratio != self.ratio:
            self.ratio = ratio
            attn_scale = 0.1 * math.log(ratio) + 1.0
            inv_freq = _yarn_inv_freq(self.d_head, self.max_eval_len, self.base, ratio)
            self.freqs_yarn = torch.outer(torch.arange(self.max_eval_len), inv_freq)
            self.attn_scale = torch.tensor(attn_scale, dtype=torch.float32)

    def forward(self, x, past_length=0):
        return x

    def rotate_qk(self, q, k, position_ids=None):
        L = q.shape[2]
        freqs = self.freqs_yarn[:L].to(q.device)
        q_rot = _apply_rotate(q, freqs)
        k_rot = _apply_rotate(k, freqs)
        s = self.attn_scale.to(q.device)
        return q_rot * s, k_rot * s


def _yarn_inv_freq(d_head, max_seq_len, base, ratio):
    """YaRN inverse frequencies: NTK-by-parts blend between interpolation and
    extrapolation over dimensions (alpha=1, beta=32), plus ratio scaling."""
    dim = d_head
    max_position_embeddings = max_seq_len
    beta_fast = 32
    beta_slow = 1

    def find_correction_dim(num_rotations, dim_, base_, max_pos):
        return (dim_ * math.log(max_pos / (num_rotations * 2 * math.pi))) / (
            2 * math.log(base_)
        )

    low = math.floor(find_correction_dim(beta_fast, dim, base, max_position_embeddings))
    high = math.ceil(find_correction_dim(beta_slow, dim, base, max_position_embeddings))
    low = max(low, 0)
    high = min(high, dim - 1)

    idx = torch.arange(dim // 2, dtype=torch.float32)
    ramp = torch.clamp((idx - low) / max(high - low, 1e-4), 0, 1)
    ext_factor = 1 - ramp

    pos_freqs = base ** (torch.arange(0, dim, 2).float() / dim)
    inv_freq_extrapolation = 1.0 / pos_freqs
    inv_freq_interpolation = 1.0 / (ratio * pos_freqs)
    inv_freq = (
        inv_freq_interpolation * (1 - ext_factor) + inv_freq_extrapolation * ext_factor
    )
    return inv_freq
