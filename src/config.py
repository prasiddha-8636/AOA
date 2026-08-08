"""
Configuration file for Colab zero-shot positional encoding benchmark.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

# NOTE: model choice per paradigm must match main.tex Section III-B (Data Collection
# Procedure) exactly. Paper and config use bigscience/bloom-560m for ALiBi.
# (facebook/opt-350m was used in earlier drafts but OPT does not use ALiBi --
# it uses learned position embeddings -- so it was replaced.)
MODELS_TO_EVAL: Dict[str, Dict[str, Any]] = {
    "learned_absolute": {
        "name": "GPT-2 (Learned Absolute)",
        "hf_model": "gpt2",
        "type": "Learned Absolute",
        "has_fixed_pos_limit": True,  # architectural hard cap: no embedding exists past this
        "max_pos": 1024,
    },
    "rope": {
        "name": "Pythia-160M (RoPE)",
        "hf_model": "EleutherAI/pythia-160m",
        "type": "Rotary (RoPE)",
        "has_fixed_pos_limit": False,  # soft cap: this is the *trained* window, not an
        "max_pos": 2048,  # architectural ceiling -- extrapolation past it
    },  # is exactly what we want to attempt and measure.
    "alibi": {
        "name": "BLOOM-560M (ALiBi)",
        "hf_model": "bigscience/bloom-560m",
        "type": "Linear Biases (ALiBi)",
        "has_fixed_pos_limit": False,  # ALiBi's bias is defined at any length by
        "max_pos": 2048,  # construction -- also a soft cap only.
    },
}

# Method keys usable with src.model.pe.get_positional_encoding for the custom
# from-scratch training track (profile.py / train.py). Keys must match the
# @register_pe decorators in src/model/pe/*.py ("learned", "sinusoidal",
# "rope", "alibi", "nope", "kerple", "cable", "position_interpolation").
ALL_METHODS: List[str] = [
    "learned",
    "rope",
    "alibi",
    "position_interpolation",
    "cable",
    "nope",
]


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
        self.dropout = kwargs.get("dropout", 0.1)

    def dict(self):
        return self.__dict__


@dataclass
class ExperimentConfig:
    """Top-level experiment config passed to train.py's train() entrypoint."""

    run_name: str = "pe_comparison"
    seed: int = 42


@dataclass
class TrainingConfig:
    """Optimizer / schedule settings for train.py."""

    batch_size: int = 32
    learning_rate: float = 3e-4
    beta1: float = 0.9
    beta2: float = 0.95
    epsilon: float = 1e-8
    weight_decay: float = 0.1
    warmup_steps: int = 500
    total_steps: int = 20000
    grad_clip: float = 1.0
    eval_interval: int = 500
    mixed_precision: str = "fp16"
    max_train_tokens: Optional[int] = None
    max_val_tokens: Optional[int] = None


@dataclass
class BenchmarkConfig:
    eval_lengths: List[int] = field(
        default_factory=lambda: [512, 1024, 2048, 4096, 8192]
    )
    needle_depths: List[float] = field(default_factory=lambda: [0.25, 0.50, 0.75, 0.90])
    needle_num_trials: int = 10
    # PG-19 per main.tex Data Collection Procedure. The original deepmind/pg19 repo
    # uses a Python loading script, which recent `datasets` versions refuse to run
    # (security change) -- use emozilla/pg19, a script-free parquet mirror of the
    # same data, confirmed to load directly. If unavailable, evaluate.get_eval_text()
    # falls back to Salesforce/wikitext (the current namespaced id -- bare "wikitext"
    # is the deprecated script-based repo and will also fail) and prints a warning --
    # check the run log before trusting a perplexity number if you see that warning.
    dataset_name: str = "emozilla/pg19"
    dataset_config: Optional[str] = None
    fallback_dataset_name: str = "Salesforce/wikitext"
    fallback_dataset_config: str = "wikitext-103-raw-v1"
    torch_dtype: str = "float16"
    device: str = "cuda"
