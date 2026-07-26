"""
One-click benchmark runner for Google Colab Free Tier.
Runs zero-shot length extrapolation evaluations across:
1. GPT-2 (Learned Absolute)
2. Pythia-160M (Rotary - RoPE)
3. BLOOM-560M (Linear Biases - ALiBi)

Generates comparison tables and saves results to JSON.
"""

import sys
import os
import gc
import json
import torch

# Add repository root to path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.config import BenchmarkConfig, MODELS_TO_EVAL
from src.evaluate import run_full_model_eval


def run_all_benchmarks():
    print("=" * 60)
    print("STARTING PE EXTENSIBILITY BENCHMARK ON COLAB FREE TIER")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU Device Name: {torch.cuda.get_device_name(0)}")
    print("=" * 60)

    cfg = BenchmarkConfig()
    all_results = {}

    for model_key in MODELS_TO_EVAL.keys():
        if torch.cuda.is_available():
            gc.collect()
            torch.cuda.empty_cache()

        try:
            res = run_full_model_eval(model_key, cfg)
            all_results[model_key] = res
        except Exception as e:
            print(f"Error evaluating model {model_key}: {e}")

    os.makedirs("results", exist_ok=True)
    out_file = "results/colab_benchmark_results.json"
    with open(out_file, "w") as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "=" * 60)
    print(f"BENCHMARK COMPLETE! Results saved to {out_file}")
    print("=" * 60)

    # Print summary comparative table
    print("\n--- SUMMARY TABLE: Perplexity Extrapolation ---")
    header = f"{'Model':<25} | " + " | ".join([f"L={L:<5}" for L in cfg.eval_lengths])
    print(header)
    print("-" * len(header))

    for key, res in all_results.items():
        row_str = f"{res['model_name']:<25} | "
        row_str += " | ".join([f"{str(res['perplexity'].get(L, 'N/A')):<6}" for L in cfg.eval_lengths])
        print(row_str)


if __name__ == "__main__":
    run_all_benchmarks()
