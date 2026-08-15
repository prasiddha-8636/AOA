"""
Train the three controlled from-scratch models (learned / rope / alibi)
with an identical ~17.7M-parameter architecture on WikiText-103.

Usage:
  python run_controlled_train.py            # train all 3 methods, resuming any done
  python run_controlled_train.py --method rope
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

from src.config import ALL_METHODS, ModelConfig, TrainingConfig
from src.train import train

MODEL_DIM = 256
N_LAYERS = 6
N_HEADS = 4
D_HEAD = 64
D_FF = 1024
MAX_SEQ_LEN = 512
VOCAB = 50257


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", default=None, help="train only this method")
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--micro-batch", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--warmup", type=int, default=1000)
    parser.add_argument("--eval-interval", type=int, default=1000)
    args = parser.parse_args()

    methods = ALL_METHODS if args.method is None else [args.method]
    if args.method is not None and args.method not in ALL_METHODS:
        raise SystemExit(f"Unknown method: {args.method}. Available: {ALL_METHODS}")

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
        batch_size=args.batch,
        micro_batch=args.micro_batch,
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
