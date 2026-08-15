import torch
import torch.nn as nn
import torch.nn.functional as F
from src.config import ModelConfig


class CausalSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig, pe_module=None):
        super().__init__()
        self.n_heads = config.n_heads
        self.d_head = config.d_head
        self.d_model = config.d_model
        self.pe_module = pe_module
        self.pe_type = type(pe_module).__name__ if pe_module else None

        self.q_proj = nn.Linear(config.d_model, config.n_heads * config.d_head)
        self.k_proj = nn.Linear(config.d_model, config.n_heads * config.d_head)
        self.v_proj = nn.Linear(config.d_model, config.n_heads * config.d_head)
        self.proj = nn.Linear(config.n_heads * config.d_head, config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        B, L, D = x.shape
        device = x.device

        q = self.q_proj(x).reshape(B, L, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(x).reshape(B, L, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x).reshape(B, L, self.n_heads, self.d_head).transpose(1, 2)

        if self.pe_type in (
            "RotaryPositionalEncoding",
            "PositionInterpolation",
            "YarnPositionalEncoding",
        ):
            q, k = self.pe_module.rotate_qk(q, k)

        att = (q @ k.transpose(-2, -1)) * (self.d_head**-0.5)

        if self.pe_type == "AlibiPositionalEncoding":
            att = att + self.pe_module.get_bias(L, device)

        if self.pe_type == "KERPLEPositionalEncoding":
            att = att + self.pe_module.get_bias(L, device)

        if self.pe_type == "CABLEPositionalEncoding":
            att = att + self.pe_module.get_bias(q, k)

        causal_mask = torch.triu(torch.ones(L, L, device=device), diagonal=1).bool()
        att = att.masked_fill(causal_mask, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)

        y = (att @ v).transpose(1, 2).reshape(B, L, -1)
        y = self.proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.d_model, config.d_ff)
        self.fc2 = nn.Linear(config.d_ff, config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig, pe_module=None):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.d_model)
        self.attn = CausalSelfAttention(config, pe_module)
        self.norm2 = nn.LayerNorm(config.d_model)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x
