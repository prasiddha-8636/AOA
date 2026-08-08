"""
Zero-shot evaluation suite for open-weight Transformers on Colab Free Tier.
Evaluates:
1. Perplexity Extrapolation across context lengths L in {512, 1024, 2048, 4096, 8192}
2. Needle-in-a-Haystack Retrieval accuracy across context depths (25%, 50%, 75%, 90%)
3. Inference Overhead (VRAM in MB, latency in ms/token)
"""

import time
import random
import torch
import numpy as np

if not hasattr(np, "long"):
    np.long = np.int64
if not hasattr(np, "ulong"):
    np.ulong = np.uint64
from typing import Dict, Any, List, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.config import BenchmarkConfig, MODELS_TO_EVAL, ModelConfig

_EVAL_TEXT_CACHE: Dict[str, str] = {}


def get_eval_text(cfg: BenchmarkConfig, min_chars: int = 400_000) -> str:
    """
    Load real evaluation text (PG-19 by default) instead of a repeated toy sentence.
    Repeated text is trivially predictable and produces meaningless (near-1.0)
    perplexity, which is exactly what the previous version of this file did.

    Falls back to WikiText-103 if the primary dataset can't be loaded (no network,
    gated dataset, etc.), and prints a clear warning so a bad number isn't silently
    reported as if it came from PG-19.
    """
    cache_key = f"{cfg.dataset_name}:{cfg.dataset_config}"
    if cache_key in _EVAL_TEXT_CACHE:
        return _EVAL_TEXT_CACHE[cache_key]

    from datasets import load_dataset

    def _concat_text(ds, n_docs: int = 20) -> str:
        chunks = []
        total_len = 0
        for i, row in enumerate(ds):
            if i >= n_docs and total_len >= min_chars:
                break
            text = row.get("text", "")
            if text:
                chunks.append(text)
                total_len += len(text)
        return "\n\n".join(chunks)

    try:
        ds = load_dataset(
            cfg.dataset_name, cfg.dataset_config, split="test", streaming=True
        )
        text = _concat_text(ds)
        if len(text) < min_chars // 4:
            raise ValueError(
                f"{cfg.dataset_name} returned suspiciously little text ({len(text)} chars)"
            )
    except Exception as e:
        print(
            f"  [WARNING] Failed to load primary dataset '{cfg.dataset_name}' ({e}). "
            f"Falling back to '{cfg.fallback_dataset_name}/{cfg.fallback_dataset_config}'. "
            f"Perplexity numbers from this run are NOT from PG-19."
        )
        try:
            ds = load_dataset(
                cfg.fallback_dataset_name,
                cfg.fallback_dataset_config,
                split="test",
                streaming=True,
            )
            text = _concat_text(ds)
        except Exception as e2:
            raise RuntimeError(
                f"Both primary dataset '{cfg.dataset_name}' and fallback "
                f"'{cfg.fallback_dataset_name}/{cfg.fallback_dataset_config}' failed to load. "
                f"Primary error: {e}. Fallback error: {e2}. "
                f"Check dataset ids are current (HF now requires namespaced repo ids)."
            ) from e2

    _EVAL_TEXT_CACHE[cache_key] = text
    return text


def get_max_pos_limit(model, model_info: Dict[str, Any]) -> Tuple[int, bool]:
    """
    Determine the position limit for a model and whether it is a HARD architectural
    ceiling or a SOFT one (the checkpoint's trained window, not a wall the mechanism
    itself enforces).

    Learned absolute embeddings have a hard ceiling: there is no embedding row past
    max_pos, so attempting to run past it is a genuine error, not a research question.

    RoPE and ALiBi have no such architectural ceiling -- their attention mechanism is
    defined at any length. model.config.max_position_embeddings for these models is
    just the length they were *trained* at. Running past it is exactly the zero-shot
    extrapolation test this benchmark exists to run, so it must not be pre-emptively
    blocked. If the underlying HF implementation itself refuses (e.g. a fixed-size
    rotary cache raising an index error), that failure is real and gets caught and
    recorded by the caller -- but this function must not manufacture that failure
    before even trying.
    """
    if model_info.get("has_fixed_pos_limit", False):
        return model_info.get("max_pos", 1024), True

    for attr in ["max_position_embeddings", "n_positions", "max_seq_len"]:
        val = getattr(model.config, attr, None)
        if val is not None and isinstance(val, int) and val > 0:
            return val, False
    return 1_000_000, False


def measure_memory_and_latency(
    model, tokenizer, seq_len: int, device: str = "cuda"
) -> Dict[str, float]:
    """Measure peak VRAM usage (MB) and per-token latency (ms/token)."""
    if not torch.cuda.is_available() or device == "cpu":
        return {"vram_mb": 0.0, "latency_ms": 0.0}

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    dummy_input = torch.randint(
        0, min(1000, tokenizer.vocab_size), (1, seq_len), device=device
    )

    start_time = time.perf_counter()
    with torch.inference_mode():
        _ = model(dummy_input)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start_time

    peak_vram = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
    latency_ms = (elapsed * 1000.0) / seq_len

    return {"vram_mb": round(peak_vram, 2), "latency_ms": round(latency_ms, 3)}


def evaluate_perplexity(
    model, tokenizer, text: str, seq_len: int, device: str = "cuda", stride: int = 512
) -> float:
    """Evaluate sliding-window perplexity for a target sequence length. No pre-emptive
    length gating here -- if the model genuinely can't handle seq_len, the forward
    pass raises and the caller records that as a real failure."""
    encodings = tokenizer(text, return_tensors="pt")
    seq_len_total = encodings.input_ids.size(1)
    if seq_len_total < seq_len:
        raise ValueError(
            f"Eval text only has {seq_len_total} tokens, need at least {seq_len}. "
            f"Increase min_chars in get_eval_text()."
        )

    nlls = []
    prev_end_loc = 0

    with torch.inference_mode():
        for begin_loc in range(0, seq_len_total, stride):
            end_loc = min(begin_loc + seq_len, seq_len_total)
            trg_len = end_loc - prev_end_loc
            if trg_len <= 0:
                break

            input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
            target_ids = input_ids.clone()
            target_ids[:, :-trg_len] = -100

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


def evaluate_ppl(model, dataloader, seq_len: int, device: str = "cuda") -> float:
    """
    Validation perplexity over a dataloader of (x, y) batches, used by train.py's
    training loop for periodic checkpoint selection. Distinct from
    evaluate_perplexity() above, which sweeps sliding windows over one long text
    for the zero-shot length-extrapolation benchmark.
    """
    model.eval()
    total_nll = 0.0
    total_tokens = 0
    with torch.inference_mode():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)), y.view(-1), reduction="sum"
            )
            total_nll += loss.item()
            total_tokens += y.numel()
    if total_tokens == 0:
        return float("nan")
    import math

    return math.exp(total_nll / total_tokens)


def evaluate_needle_haystack(
    model,
    tokenizer,
    seq_len: int,
    depth_ratio: float,
    device: str = "cuda",
    n_trials: int = 10,
) -> float:
    """
    Evaluate zero-shot needle-in-a-haystack retrieval accuracy at a given depth
    ratio, averaged over n_trials with a freshly randomized secret code each trial
    (so the model can't answer from having memorized one fixed string during
    pretraining).

    The question is reserved token budget *before* the haystack is built, so the
    final prompt never needs post-hoc truncation that could cut the question off.
    """
    question_template = "\nWhat is the secret code? Answer:"
    question_ids = tokenizer.encode(question_template)
    budget_for_haystack = max(seq_len - len(question_ids), 1)

    filler = "The quick brown fox jumps over the lazy dog. Random context sentence to fill space. "
    filler_ids = tokenizer.encode(filler)
    reps_needed = budget_for_haystack // max(len(filler_ids), 1) + 2
    haystack_ids = (filler_ids * reps_needed)[:budget_for_haystack]

    successes = 0
    for _ in range(n_trials):
        code = str(random.randint(10000, 99999))
        needle = f"The secret code is {code}."
        needle_ids = tokenizer.encode(needle)

        trial_ids = list(haystack_ids)
        insert_idx = min(
            int(len(trial_ids) * depth_ratio), max(len(trial_ids) - len(needle_ids), 0)
        )
        trial_ids[insert_idx : insert_idx + len(needle_ids)] = needle_ids

        full_ids = trial_ids + question_ids
        input_ids = torch.tensor([full_ids], device=device)

        with torch.inference_mode():
            output = model.generate(input_ids, max_new_tokens=10, do_sample=False)

        generated_text = tokenizer.decode(
            output[0][input_ids.size(1) :], skip_special_tokens=True
        )
        if code in generated_text:
            successes += 1

    return round(successes / n_trials, 3)


def run_full_model_eval(model_key: str, cfg: BenchmarkConfig = None) -> Dict[str, Any]:
    """Run full benchmark for a specified model key."""
    if cfg is None:
        cfg = BenchmarkConfig()
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

    max_pos, is_hard_limit = get_max_pos_limit(model, model_info)
    eval_text = get_eval_text(cfg, min_chars=max(cfg.eval_lengths) * 8)

    results = {
        "model_name": model_info["name"],
        "model_type": model_info["type"],
        "trained_max_pos": max_pos,
        "is_hard_limit": is_hard_limit,
        "perplexity": {},
        "needle_accuracy": {},
        "overhead": {},
    }

    for L in cfg.eval_lengths:
        print(f"\n--- Sequence Length L = {L} ---")

        if is_hard_limit and L > max_pos:
            print(
                f"  [SKIPPED] Length L={L} exceeds architectural embedding limit ({max_pos})"
            )
            results["perplexity"][L] = f"Skipped (hard limit {max_pos})"
            results["overhead"][L] = {"vram_mb": 0.0, "latency_ms": 0.0}
            results["needle_accuracy"][L] = {d: None for d in cfg.needle_depths}
            continue

        if (not is_hard_limit) and L > max_pos:
            print(
                f"  [NOTE] L={L} exceeds trained window ({max_pos}). Attempting anyway -- "
                f"this IS the extrapolation test."
            )

        try:
            ppl = evaluate_perplexity(
                model, tokenizer, eval_text, seq_len=L, device=device
            )
            results["perplexity"][L] = ppl
            print(f"  Perplexity: {ppl}")
        except Exception as e:
            results["perplexity"][L] = "Failed"
            print(f"  Perplexity evaluation failed: {e}")

        try:
            overhead = measure_memory_and_latency(
                model, tokenizer, seq_len=L, device=device
            )
            results["overhead"][L] = overhead
            print(
                f"  Peak VRAM: {overhead['vram_mb']} MB | Latency: {overhead['latency_ms']} ms/token"
            )
        except Exception as e:
            results["overhead"][L] = {"vram_mb": 0.0, "latency_ms": 0.0}
            print(f"  Memory measurement failed: {e}")

        results["needle_accuracy"][L] = {}
        for d in cfg.needle_depths:
            try:
                # PREVENT DEVICE-SIDE ASSERT:
                # If the model has a hard positional limit, generating tokens beyond max_pos
                # will trigger a CUDA index out-of-bounds assertion.
                max_new_tokens = 10
                if is_hard_limit and (L + max_new_tokens) > max_pos:
                    print(
                        f"  [SKIPPED] Needle eval at L={L} skipped: total sequence length "
                        f"({L} prompt + {max_new_tokens} generated tokens) exceeds hard limit ({max_pos})"
                    )
                    results["needle_accuracy"][L][d] = None
                    continue

                acc = evaluate_needle_haystack(
                    model,
                    tokenizer,
                    seq_len=L,
                    depth_ratio=d,
                    device=device,
                    n_trials=cfg.needle_num_trials,
                )
                results["needle_accuracy"][L][d] = acc
                print(
                    f"  Needle @ depth {d * 100:.0f}%: {acc * 100:.0f}% ({cfg.needle_num_trials} trials)"
                )
            except Exception as e:
                results["needle_accuracy"][L][d] = None
                print(f"  Needle eval failed at L={L}, depth={d}: {e}")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="learned_absolute",
        choices=list(MODELS_TO_EVAL.keys()),
    )
    args = parser.parse_args()

    eval_results = run_full_model_eval(args.model)
    print("\nBenchmark Complete!")
    print(eval_results)
