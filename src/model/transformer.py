from src.config import ModelConfig
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

        self.blocks = nn.ModuleList([
            TransformerBlock(config, pe_module) for _ in range(config.n_layers)
        ])

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
