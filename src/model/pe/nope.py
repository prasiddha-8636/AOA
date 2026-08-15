import torch.nn as nn
from . import PositionalEncoding, register_pe


@register_pe("nope")
class NoPositionalEncoding(PositionalEncoding):
    def __init__(self, d_model=None, n_heads=None, max_seq_len=None):
        super().__init__()

    def forward(self, x, past_length=0):
        return x
