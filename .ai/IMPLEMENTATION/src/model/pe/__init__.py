from abc import ABC, abstractmethod
import torch.nn as nn
import torch


class PositionalEncoding(nn.Module, ABC):
    @abstractmethod
    def forward(self, x, past_length=0):
        raise NotImplementedError


PE_REGISTRY = {}


def register_pe(name):
    def decorator(cls):
        PE_REGISTRY[name] = cls
        return cls
    return decorator


def get_positional_encoding(name, d_model, n_heads, max_seq_len):
    if name not in PE_REGISTRY:
        raise ValueError(f"Unknown PE method: {name}. Available: {list(PE_REGISTRY.keys())}")
    return PE_REGISTRY[name](d_model=d_model, n_heads=n_heads, max_seq_len=max_seq_len)
