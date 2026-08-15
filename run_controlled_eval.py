"""
Zero-shot length-extrapolation evaluation for the from-scratch controlled
models (learned / rope / alibi, identical architecture). Mirrors the metrics
of the pretrained zero-shot benchmark so the two sets of numbers are comparable:

1. Sliding-window perplexity at L in {512, 1024, 2048, 4096, 8192}
2. Needle-in-a-haystack retrieval accuracy at depths 25/50/75/90%
3. Per-token latency

Usage: python run_controlled_eval.py [checkpoint_dir]
"""

import sys
import os
import json
import time
import gc
import random

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

if not hasattr(np, "long"):
    np.long = np.int64
if not hasattr(np, "ulong"):
    np.ulong = np.uint64

import torch
from transformers import GPT2Tokenizer

from src.config import BenchmarkConfig, ModelConfig
from src.model.pe import get_positional_encoding
from src.model.transformer import Transformer
from src.colab_io import persist_path
from src.evaluate import (
    evaluate_perplexity_custom,
    evaluate_needle_haystack,
    get_eval_text,
)

EVAL_LENGTHS = [512, 1024, 2048, 4096, 8192]
NEEDLE_DEPTHS = [0.0, 0.25, 0.50, 0.75, 1.0]
N_TRIALS = 10
MODEL_DIM = 256
N_LAYERS = 6
N_HEADS = 4
D_HEAD = 64
D_FF = 1024
MAX_SEQ_LEN = 512
VOCAB = 50257


def build_model(method: str, checkpoint: str, device: str, strict: bool = True):
    cfg = ModelConfig(
        d_model=MODEL_DIM,
        n_layers=N_LAYERS,
        n_heads=N_HEADS,
        d_head=D_HEAD,
        d_ff=D_FF,
        max_seq_len=MAX_SEQ_LEN,
        vocab_size=VOCAB,
        dropout=0.0,
    )
    pe = get_positional_encoding(method, cfg.d_model, cfg.n_heads, cfg.max_seq_len)
    model = Transformer(cfg, pe).to(device)
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state, strict=strict)
    if method == "position_interpolation":
        pe.set_scale(MAX_SEQ_LEN, 8192)
    elif method == "yarn":
        pe.set_ratio(MAX_SEQ_LEN, 8192)
    model.eval()
    return model


# Inference-time RoPE-scaling extensions reuse the vanilla RoPE checkpoint.
ROPE_EXTENSIONS = {"position_interpolation", "yarn"}


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint_dir", nargs="?", default="checkpoints")
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["learned", "rope", "alibi"],
        help="methods to evaluate; position_interpolation/yarn load the rope checkpoint",
    )
    args = parser.parse_args()
    checkpoint_dir = persist_path("checkpoints", args.checkpoint_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    print(
        f"Device: {device} ({torch.cuda.get_device_name(0) if device == 'cuda' else 'cpu'})"
    )

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    eval_text = get_eval_text(BenchmarkConfig(), min_chars=512 * 8 * 30)

    methods = args.methods
    out = {}
    for method in methods:
        if method in ROPE_EXTENSIONS:
            ckpt = os.path.join(checkpoint_dir, "rope", "best.pt")
            strict = False
        else:
            ckpt = os.path.join(checkpoint_dir, method, "best.pt")
            strict = True
        if not os.path.exists(ckpt):
            print(f"[SKIP] no checkpoint for {method}: {ckpt}")
            continue
        print(f"\n=== {method} ===")
        model = build_model(method, ckpt, device, strict=strict)

        res = {
            "model_type": method,
            "params_m": round(sum(p.numel() for p in model.parameters()) / 1e6, 2),
            "trained_max_pos": MAX_SEQ_LEN,
            "perplexity": {},
            "needle_accuracy": {},
            "overhead": {},
        }

        for L in EVAL_LENGTHS:
            print(f"--- L={L} ---")
            try:
                ppl = evaluate_perplexity_custom(
                    model, tokenizer, eval_text, seq_len=L, device=device
                )
                res["perplexity"][str(L)] = ppl
                print(f"  ppl={ppl}")
            except Exception as e:
                res["perplexity"][str(L)] = f"Failed: {type(e).__name__}"
                print(f"  ppl FAILED: {e}")

            try:
                inputs = tokenizer(eval_text[: 512 * 8], return_tensors="pt")[
                    "input_ids"
                ][:, :L].to(device)
                torch.cuda.reset_peak_memory_stats() if device == "cuda" else None
                t0 = time.time()
                with torch.inference_mode():
                    model(inputs)
                dt = time.time() - t0
                vram = (
                    torch.cuda.max_memory_allocated() / 1e6 if device == "cuda" else 0.0
                )
                res["overhead"][str(L)] = {
                    "vram_mb": round(vram, 1),
                    "latency_ms": round(1000 * dt / L, 3),
                }
                print(
                    f"  latency={res['overhead'][str(L)]['latency_ms']} ms/tok, vram={vram:.1f} MB"
                )
            except Exception as e:
                res["overhead"][str(L)] = {"vram_mb": 0.0, "latency_ms": 0.0}
                print(f"  overhead FAILED: {e}")

            res["needle_accuracy"][str(L)] = {}
            for d in NEEDLE_DEPTHS:
                try:
                    acc = evaluate_needle_haystack(
                        model,
                        tokenizer,
                        seq_len=L,
                        depth_ratio=d,
                        device=device,
                        n_trials=N_TRIALS,
                    )
                    res["needle_accuracy"][str(L)][str(d)] = acc
                    print(f"  needle@{int(d * 100)}%={acc * 100:.0f}%", end="  ")
                except Exception as e:
                    res["needle_accuracy"][str(L)][str(d)] = None
                    print(f"  needle@{int(d * 100)}% FAILED: {e}")
            print()

        out[method] = res
        del model
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

    os.makedirs("results", exist_ok=True)
    out_name = (
        "results/controlled_results.json"
        if methods == ["learned", "rope", "alibi"]
        else f"results/controlled_results_{'_'.join(methods)}.json"
    )
    out_name = persist_path("results", out_name)
    with open(out_name, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {out_name}")


if __name__ == "__main__":
    main()
