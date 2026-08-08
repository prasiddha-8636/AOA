from typing import Optional
from src.config import ModelConfig
import torch
import torch.nn as nn
from .pe import PositionalEncoding
from .layers import TransformerBlock


class Transformer(nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        pe_module: PositionalEncoding,
    ):
        super().__init__()
        self.config = config
        self.pe_module = pe_module

        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.dropout = nn.Dropout(config.dropout)

        self.blocks = nn.ModuleList(
            [TransformerBlock(config, pe_module) for _ in range(config.n_layers)]
        )

        self.ln_f = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.token_embedding.weight = self.lm_head.weight

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                nn.init.zeros_(module.bias)
                nn.init.ones_(module.weight)

    def forward(self, input_ids):
        B, L = input_ids.shape
        x = self.token_embedding(input_ids)

        if not isinstance(self.pe_module, type(None)):
            pe_type = type(self.pe_module).__name__
            if pe_type in ("LearnedPositionalEncoding", "SinusoidalPositionalEncoding"):
                x = self.pe_module(x)
            elif pe_type == "NoPositionalEncoding":
                pass

        x = self.dropout(x)
        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits

    @torch.inference_mode()
    def generate(
        self,
        input_ids,
        max_new_tokens: int = 10,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ):
        """Greedy or sampled autoregressive decoding compatible with
        evaluate_needle_haystack(). Returns full sequence (prompt + generated)."""
        import torch

        self.eval()
        out = input_ids.clone()
        for _ in range(max_new_tokens):
            logits = self(out)
            next_logits = logits[:, -1, :]
            if do_sample:
                if temperature != 1.0:
                    next_logits = next_logits / temperature
                probs = torch.softmax(next_logits, dim=-1)
                if top_k is not None:
                    top_k_vals, top_k_idx = probs.topk(top_k)
                    probs = torch.zeros_like(probs).scatter_(-1, top_k_idx, top_k_vals)
                    probs = probs / probs.sum(dim=-1, keepdim=True)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = next_logits.argmax(dim=-1, keepdim=True)
            out = torch.cat([out, next_token], dim=1)
        return out
