import torch.nn as nn
from . import PositionalEncoding, register_pe


@register_pe("nope")
class NoPositionalEncoding(PositionalEncoding):
    def forward(self, x, past_length=0):
        return x
