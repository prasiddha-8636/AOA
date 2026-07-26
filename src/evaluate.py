"""
Zero-shot evaluation suite for open-weight Transformers on Colab Free Tier.
Evaluates:
1. Perplexity Extrapolation across context lengths L in {512, 1024, 2048, 4096, 8192}
2. Needle-in-a-Haystack Retrieval accuracy across context depths (25%, 50%, 75%, 90%)
3. Inference Overhead (VRAM in MB, latency in ms/token)
"""

import time
import math
import torch
import numpy as np
from typing import Dict, Any, List
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.config import BenchmarkConfig, MODELS_TO_EVAL, ModelConfig


def get_max_pos_limit(model, model_info: Dict[str, Any]) -> int:
    """Safely determine max supported position limit for a model."""
    if model_info.get("has_fixed_pos_limit", False):
        return model_info.get("max_pos", 1024)

    for attr in ["max_position_embeddings", "n_positions", "max_seq_len"]:
        val = getattr(model.config, attr, None)
        if val is not None and isinstance(val, int) and val > 0:
            return val
    return 1000000


def measure_memory_and_latency(model, tokenizer, seq_len: int, max_pos: int, device: str = "cuda") -> Dict[str, float]:
    """Measure peak VRAM usage (MB) and per-token latency (ms/token)."""
    if not torch.cuda.is_available() or device == "cpu":
        return {"vram_mb": 0.0, "latency_ms": 0.0}

    if seq_len > max_pos:
        return {"vram_mb": 0.0, "latency_ms": 0.0}

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    dummy_input = torch.randint(0, min(1000, tokenizer.vocab_size), (1, seq_len), device=device)

    start_time = time.perf_counter()
    with torch.inference_mode():
        _ = model(dummy_input)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start_time

    peak_vram = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
    latency_ms = (elapsed * 1000.0) / seq_len

    return {"vram_mb": round(peak_vram, 2), "latency_ms": round(latency_ms, 3)}


def evaluate_perplexity(model, tokenizer, text: str, seq_len: int, max_pos: int, stride: int = 512, device: str = "cuda") -> float:
    """Evaluate sliding-window perplexity for a target sequence length."""
    if seq_len > max_pos:
        raise ValueError(f"Sequence length {seq_len} exceeds model max position limit {max_pos}")

    encodings = tokenizer(text, return_tensors="pt")
    seq_len_total = encodings.input_ids.size(1)

    nlls = []
    prev_end_loc = 0

    with torch.inference_mode():
        for begin_loc in range(0, seq_len_total, stride):
            end_loc = min(begin_loc + seq_len, seq_len_total)
            trg_len = end_loc - prev_end_loc

            input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
            if input_ids.size(1) > max_pos:
                raise ValueError(f"Input slice length {input_ids.size(1)} exceeds model position limit {max_pos}")

            target_ids = input_ids.clone()
            target_ids[:, :-trg_len] = -100

            if input_ids.size(1) == 0:
                break

            outputs = model(input_ids, labels=target_ids)
            neg_log_likelihood = outputs.loss * trg_len

            nlls.append(neg_log_likelihood)
            prev_end_loc = end_loc
            if end_loc == seq_len_total:
                break

    if not nlls:
        return float("nan")

    total_nll = torch.stack(nlls).sum()
    ppl = torch.exp(total_nll / prev_end_loc).item()
    return round(ppl, 2)


def evaluate_needle_haystack(
    model, tokenizer, seq_len: int, max_pos: int, depth_ratio: float, needle: str = "The secret code is 84920.", device: str = "cuda"
) -> bool:
    """Evaluate zero-shot needle-in-a-haystack retrieval accuracy at a given depth ratio."""
    if seq_len > max_pos:
        return False

    haystack_filler = "The quick brown fox jumps over the lazy dog. Random context sentence to fill space. "
    haystack = haystack_filler * (seq_len // len(haystack_filler) + 10)

    tokens = tokenizer.encode(haystack)[:seq_len]
    needle_tokens = tokenizer.encode(needle)

    insert_idx = int(len(tokens) * depth_ratio)
    tokens[insert_idx : insert_idx + len(needle_tokens)] = needle_tokens

    prompt = tokenizer.decode(tokens) + "\nWhat is the secret code? Answer:"
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    if input_ids.size(1) > max_pos:
        input_ids = input_ids[:, :max_pos]

    with torch.inference_mode():
        output = model.generate(input_ids, max_new_tokens=10, do_sample=False)

    generated_text = tokenizer.decode(output[0][input_ids.size(1) :], skip_special_tokens=True)
    return "84920" in generated_text


def run_full_model_eval(model_key: str, cfg: BenchmarkConfig = BenchmarkConfig()) -> Dict[str, Any]:
    """Run full benchmark for a specified model key."""
    if model_key not in MODELS_TO_EVAL:
        raise ValueError(f"Unknown model key: {model_key}")

    model_info = MODELS_TO_EVAL[model_key]
    print(f"\n==========================================")
    print(f"Evaluating: {model_info['name']} ({model_info['hf_model']})")
    print(f"==========================================")

    device = cfg.device if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(model_info["hf_model"])
    model = AutoModelForCausalLM.from_pretrained(
        model_info["hf_model"], torch_dtype=dtype, trust_remote_code=True
    ).to(device)

    model.eval()
    max_pos = get_max_pos_limit(model, model_info)
    has_fixed_pos = model_info.get("has_fixed_pos_limit", False)

    sample_text = (
        "Positional encoding is a crucial component of Transformer architectures. "
        "It provides sequence order information to the self-attention mechanism, which is permutation-invariant by default. "
    ) * 200

    results = {
        "model_name": model_info["name"],
        "model_type": model_info["type"],
        "perplexity": {},
        "needle_accuracy": {},
        "overhead": {},
    }

    for L in cfg.eval_lengths:
        print(f"\n--- Sequence Length L = {L} ---")

        # Safely skip out-of-bounds evaluation for fixed position embedding models (e.g. GPT-2 L > 1024)
        if has_fixed_pos and L > max_pos:
            print(f"  [SKIPPED] Length L={L} exceeds fixed embedding limit ({max_pos})")
            results["perplexity"][L] = f"Skipped (Exceeds Limit {max_pos})"
            results["overhead"][L] = {"vram_mb": 0.0, "latency_ms": 0.0}
            results["needle_accuracy"][L] = {d: 0.0 for d in cfg.needle_depths}
            continue

        try:
            ppl = evaluate_perplexity(model, tokenizer, sample_text, seq_len=L, max_pos=max_pos, device=device)
            results["perplexity"][L] = ppl
            print(f"  Perplexity: {ppl}")
        except Exception as e:
            results["perplexity"][L] = "Failed"
            print(f"  Perplexity evaluation failed: {e}")

        try:
            overhead = measure_memory_and_latency(model, tokenizer, seq_len=L, max_pos=max_pos, device=device)
            results["overhead"][L] = overhead
            print(f"  Peak VRAM: {overhead['vram_mb']} MB | Latency: {overhead['latency_ms']} ms/token")
        except Exception as e:
            results["overhead"][L] = {"vram_mb": 0.0, "latency_ms": 0.0}
            print(f"  Memory measurement failed: {e}")

        results["needle_accuracy"][L] = {}
        for d in cfg.needle_depths:
            try:
                acc = evaluate_needle_haystack(model, tokenizer, seq_len=L, max_pos=max_pos, depth_ratio=d, device=device)
                results["needle_accuracy"][L][d] = 1.0 if acc else 0.0
                print(f"  Needle @ depth {d*100:.0f}%: {'SUCCESS' if acc else 'FAIL'}")
            except Exception as e:
                results["needle_accuracy"][L][d] = 0.0
                print(f"  Needle eval failed at L={L}, depth={d}: {e}")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="learned_absolute", choices=list(MODELS_TO_EVAL.keys()))
    args = parser.parse_args()

    eval_results = run_full_model_eval(args.model)
    print("\nBenchmark Complete!")
    print(eval_results)
