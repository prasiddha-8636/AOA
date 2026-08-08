"""
Train the three controlled from-scratch models (learned / rope / alibi)
with an identical ~16M-parameter architecture on WikiText-103.

Usage:
  python run_controlled_train.py            # train all 3 methods, resuming any done
  python run_controlled_train.py --method rope
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

from src.config import ModelConfig, TrainingConfig
from src.train import train

MODEL_DIM = 384
N_LAYERS = 8
N_HEADS = 8
D_HEAD = 64
D_FF = 1536
MAX_SEQ_LEN = 512
VOCAB = 50257


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", default=None, help="train only this method")
    parser.add_argument("--steps", type=int, default=25000)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--warmup", type=int, default=1000)
    parser.add_argument("--eval-interval", type=int, default=500)
    args = parser.parse_args()

    methods = ["learned", "rope", "alibi"] if args.method is None else [args.method]
    if args.method not in ("learned", "rope", "alibi", None):
        raise SystemExit(f"Unknown method: {args.method}")

    mc = ModelConfig(
        d_model=MODEL_DIM,
        n_layers=N_LAYERS,
        n_heads=N_HEADS,
        d_head=D_HEAD,
        d_ff=D_FF,
        max_seq_len=MAX_SEQ_LEN,
        vocab_size=VOCAB,
        dropout=0.1,
    )
    tc = TrainingConfig(
        batch_size=args.batch,
        learning_rate=args.lr,
        warmup_steps=args.warmup,
        total_steps=args.steps,
        eval_interval=args.eval_interval,
        mixed_precision="fp16",
    )

    torch.manual_seed(42)
    for m in methods:
        print(f"\n{'=' * 60}\nTraining: {m}\n{'=' * 60}", flush=True)
        train(
            m,
            model_config=mc,
            train_config=tc,
            out_dir="checkpoints",
        )


if __name__ == "__main__":
    main()
