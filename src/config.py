"""
Configuration file for Colab zero-shot positional encoding benchmark.
"""

from dataclasses import dataclass, field
from typing import List, Dict

MODELS_TO_EVAL: Dict[str, Dict[str, str]] = {
    "learned_absolute": {
        "name": "GPT-2 (Learned Absolute)",
        "hf_model": "gpt2",
        "type": "Learned Absolute",
    },
    "rope": {
        "name": "Pythia-160M (RoPE)",
        "hf_model": "EleutherAI/pythia-160m",
        "type": "Rotary (RoPE)",
    },
    "alibi": {
        "name": "OPT-350M (ALiBi)",
        "hf_model": "facebook/opt-350m",
        "type": "Linear Biases (ALiBi)",
    },
}


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
