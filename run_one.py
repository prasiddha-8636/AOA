import sys, os, json, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

if not hasattr(np, "long"):
    np.long = np.int64
if not hasattr(np, "ulong"):
    np.ulong = np.uint64
import torch
from src.config import BenchmarkConfig, MODELS_TO_EVAL
from src.evaluate import run_full_model_eval

model_key = sys.argv[1]
cfg = BenchmarkConfig(eval_lengths=[512, 1024, 2048, 4096, 8192], needle_num_trials=3)

print(f"Starting {model_key}...", flush=True)
t0 = time.time()
try:
    res = run_full_model_eval(model_key, cfg)
    os.makedirs("results", exist_ok=True)
    out_file = f"results/{model_key}_results.json"
    with open(out_file, "w") as f:
        json.dump(res, f, indent=2, default=str)
    print(f"DONE {model_key} in {time.time() - t0:.1f}s -> {out_file}", flush=True)
except Exception as e:
    print(f"FAILED {model_key}: {e}", flush=True)
    import traceback

    traceback.print_exc()
