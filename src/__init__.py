import torch
from .model.transformer import Transformer
from .model.pe import get_positional_encoding

def build_model(pe_method, vocab_size, d_model, n_layers, n_heads, d_ff, max_seq_len, dropout):
    pe_module = get_positional_encoding(pe_method, d_model, n_heads, max_seq_len)
    model = Transformer(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        d_ff=d_ff,
        max_seq_len=max_seq_len,
        dropout=dropout,
        pe_module=pe_module,
    )
    return model
