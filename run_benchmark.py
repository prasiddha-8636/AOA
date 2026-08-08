import sys, os, gc, json, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

if not hasattr(np, "long"):
    np.long = np.int64
if not hasattr(np, "ulong"):
    np.ulong = np.uint64

import torch
from src.config import BenchmarkConfig, MODELS_TO_EVAL
from src.evaluate import run_full_model_eval
from src.plotting import plot_all


def run_all():
    print("=" * 60)
    print("PE EXTRAPOLATION BENCHMARK")
    print(f"CUDA: {torch.cuda.is_available()}")
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print("=" * 60)

    cfg = BenchmarkConfig(
        eval_lengths=[512, 1024, 2048, 4096, 8192],
        needle_num_trials=3,
    )
    all_results = {}

    for model_key in MODELS_TO_EVAL.keys():
        gc.collect()
        t0 = time.time()
        try:
            print(f"\n>>> Starting {model_key}...")
            res = run_full_model_eval(model_key, cfg)
            all_results[model_key] = res
            print(f">>> {model_key} done in {time.time() - t0:.1f}s")
        except Exception as e:
            print(f">>> {model_key} FAILED: {e}")
            import traceback

            traceback.print_exc()

        os.makedirs("results", exist_ok=True)
        with open("results/colab_benchmark_results.json", "w") as f:
            json.dump(all_results, f, indent=2, default=str)

    print("\n" + "=" * 60)
    print("BENCHMARK COMPLETE")
    print("=" * 60)

    header = f"{'Model':<30} | " + " | ".join([f"L={L:<6}" for L in cfg.eval_lengths])
    print(f"\n{header}")
    print("-" * len(header))
    for key, res in all_results.items():
        row = f"{res['model_name']:<30} | "
        row += " | ".join(
            [f"{str(res['perplexity'].get(L, 'N/A'))[:8]:<8}" for L in cfg.eval_lengths]
        )
        print(row)

    try:
        plot_all(all_results)
        print("Plots saved to results/")
    except Exception as e:
        print(f"Plotting failed: {e}")


if __name__ == "__main__":
    run_all()
