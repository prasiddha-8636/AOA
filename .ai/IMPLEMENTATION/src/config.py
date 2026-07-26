"""
Configuration file for Colab zero-shot positional encoding benchmark.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any

MODELS_TO_EVAL: Dict[str, Dict[str, Any]] = {
    "learned_absolute": {
        "name": "GPT-2 (Learned Absolute)",
        "hf_model": "gpt2",
        "type": "Learned Absolute",
        "has_fixed_pos_limit": True,
        "max_pos": 1024,
    },
    "rope": {
        "name": "Pythia-160M (RoPE)",
        "hf_model": "EleutherAI/pythia-160m",
        "type": "Rotary (RoPE)",
        "has_fixed_pos_limit": False,
        "max_pos": 8192,
    },
    "alibi": {
        "name": "BLOOM-560M (ALiBi)",
        "hf_model": "bigscience/bloom-560m",
        "type": "Linear Biases (ALiBi)",
        "has_fixed_pos_limit": False,
        "max_pos": 8192,
    },
}


class ModelConfig:
    """Compatibility config class supporting dataclass/pydantic attribute access."""
    def __init__(self, **kwargs):
        self.vocab_size = kwargs.get("vocab_size", 50257)
        self.d_model = kwargs.get("d_model", 768)
        self.n_layers = kwargs.get("n_layers", 12)
        self.n_heads = kwargs.get("n_heads", 12)
        self.d_head = kwargs.get("d_head", 64)
        self.d_ff = kwargs.get("d_ff", 3072)
        self.max_seq_len = kwargs.get("max_seq_len", 512)

    def dict(self):
        return self.__dict__


@dataclass
class BenchmarkConfig:
    eval_lengths: List[int] = field(
        default_factory=lambda: [512, 1024, 2048, 4096, 8192]
    )
    needle_depths: List[float] = field(
        default_factory=lambda: [0.25, 0.50, 0.75, 0.90]
    )
    needle_num_trials: int = 10
    dataset_name: str = "wikitext"
    dataset_config: str = "wikitext-103-raw-v1"
    torch_dtype: str = "float16"
    device: str = "cuda"
