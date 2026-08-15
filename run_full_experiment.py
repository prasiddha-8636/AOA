"""
Unified Colab script: train all baselines + evaluate all PE methods.

Runs on a single T4 (16GB). Trains in sequence to avoid OOM.
Expected total time: ~5h (5 × 50min train + 10 × 5min eval).

Usage in Colab:
  !python run_full_experiment.py
  # or selective:
  !python run_full_experiment.py --train-methods rope sinusoidal
  !python run_full_experiment.py --skip-train --eval-methods all
"""

import argparse
import json
import os
import sys
import time

# Set before torch import: reduce CUDA allocator fragmentation from CABLE's
# repeated large bias-tensor allocations (recommended by the OOM message itself).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.colab_io import persist_path

# ── Training methods (must train from scratch) ──
# rope is needed for PI/YaRN; all others are new baselines
REQUIRED_TRAINS = ["rope", "learned", "alibi", "sinusoidal", "nope", "kerple", "cable"]

# ── All evaluation methods ──
# PI and YaRN are inference-only: they reuse the rope checkpoint
ALL_EVAL = [
    "learned",
    "rope",
    "alibi",
    "sinusoidal",
    "nope",
    "kerple",
    "cable",
    "position_interpolation",
    "yarn",
]


def train_one(method, steps=10000, batch=16, micro_batch=8, lr=3e-4, warmup=1000):
    """Train a single PE method using run_controlled_train.py infrastructure."""
    from src.config import ModelConfig, TrainingConfig
    from src.train import train

    # CABLE's bias net materializes O(B*H*L*L*D) expanded q/k tensors retained
    # in the autograd graph through backward. Shrink the micro-batch and grow
    # gradient accumulation: identical effective batch and summed gradient, but
    # 4x lower activation memory on small GPUs.
    if method == "cable":
        micro_batch = 2

    MODEL_DIM, N_LAYERS, N_HEADS, D_HEAD, D_FF, MAX_SEQ_LEN, VOCAB = (
        256,
        6,
        4,
        64,
        1024,
        512,
        50257,
    )
    mc = ModelConfig(
        d_model=MODEL_DIM,
        n_layers=N_LAYERS,
        n_heads=N_HEADS,
        d_head=D_HEAD,
        d_ff=D_FF,
        max_seq_len=MAX_SEQ_LEN,
        vocab_size=VOCAB,
        dropout=0.0,
    )
    tc = TrainingConfig(
        batch_size=batch,
        micro_batch=micro_batch,
        learning_rate=lr,
        warmup_steps=warmup,
        total_steps=steps,
        eval_interval=1000,
        mixed_precision="fp16",
    )

    ckpt_dir = persist_path("checkpoints", "checkpoints")
    ckpt_path = os.path.join(ckpt_dir, method, "best.pt")
    if os.path.exists(ckpt_path):
        print(f"[SKIP] checkpoint already exists: {ckpt_path}")
        return

    print(f"\n{'=' * 60}\nTraining: {method} ({steps} steps)\n{'=' * 60}")
    t0 = time.time()
    train(method, model_config=mc, train_config=tc, out_dir="checkpoints")
    elapsed = time.time() - t0
    print(f"  Training {method} done in {elapsed / 60:.1f} min")


def eval_all(methods):
    """Run evaluation on all specified methods using run_controlled_eval.py."""
    import subprocess

    methods_str = " ".join(methods)
    cmd = f"python run_controlled_eval.py checkpoints --methods {methods_str}"
    print(f"\n{'=' * 60}\nEvaluating: {methods}\n{'=' * 60}")
    print(f"  Command: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=False)
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description="Full PE experiment pipeline")
    parser.add_argument(
        "--train-methods",
        nargs="*",
        default=None,
        help="Methods to train (default: all 7)",
    )
    parser.add_argument(
        "--eval-methods",
        nargs="*",
        default=None,
        help="Methods to evaluate (default: all 9)",
    )
    parser.add_argument(
        "--skip-train", action="store_true", help="Skip training, only evaluate"
    )
    parser.add_argument(
        "--skip-eval", action="store_true", help="Skip evaluation, only train"
    )
    parser.add_argument("--steps", type=int, default=10000)
    args = parser.parse_args()

    train_methods = args.train_methods or REQUIRED_TRAINS
    eval_methods = args.eval_methods or ALL_EVAL

    # ── Phase 1: Training ──
    if not args.skip_train:
        total_t0 = time.time()
        for m in train_methods:
            train_one(m, steps=args.steps)
        print(f"\nTotal training time: {(time.time() - total_t0) / 60:.1f} min")

    # ── Phase 2: Evaluation ──
    if not args.skip_eval:
        eval_all(eval_methods)

    # ── Phase 3: Summary ──
    print("\n" + "=" * 60)
    print("EXPERIMENT COMPLETE")
    print("=" * 60)
    print("Results saved to: results/controlled_results_*.json")
    print("Checkpoints saved to: checkpoints/")
    print("\nTo update the paper, copy the perplexity numbers from the")
    print("results JSON into paper/main.tex tables.")


if __name__ == "__main__":
    main()
