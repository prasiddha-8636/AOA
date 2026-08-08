"""
Self-contained PE extrapolation benchmark. Bypasses HuggingFace datasets library.
Downloads eval text from Project Gutenberg directly.
"""

import sys, os, json, time, gc, random, urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

if not hasattr(np, "long"):
    np.long = np.int64
if not hasattr(np, "ulong"):
    np.ulong = np.uint64
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.config import MODELS_TO_EVAL

EVAL_URLS = [
    "https://www.gutenberg.org/files/1342/1342-0.txt",
    "https://www.gutenberg.org/files/11/11-0.txt",
    "https://www.gutenberg.org/files/98/98-0.txt",
]


def download_eval_text(min_chars=500000):
    chunks = []
    total = 0
    for url in EVAL_URLS:
        if total >= min_chars:
            break
        try:
            resp = urllib.request.urlopen(url, timeout=30)
            text = resp.read().decode("utf-8", errors="ignore")
            chunks.append(text)
            total += len(text)
            print(
                f"  Downloaded {len(text)} chars from {url.split('/')[-1]}", flush=True
            )
        except Exception as e:
            print(f"  Failed {url}: {e}", flush=True)
    return "\n\n".join(chunks)


def evaluate_perplexity(model, tokenizer, text, seq_len, device="cpu", stride=512):
    encodings = tokenizer(text, return_tensors="pt")
    seq_len_total = encodings.input_ids.size(1)
    if seq_len_total < seq_len:
        return float("nan")
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


def evaluate_needle(model, tokenizer, seq_len, depth_ratio, device="cpu", n_trials=3):
    question_template = "\nWhat is the secret code? Answer:"
    question_ids = tokenizer.encode(question_template)
    budget = max(seq_len - len(question_ids), 1)
    filler = "The quick brown fox jumps over the lazy dog. Random context sentence to fill space. "
    filler_ids = tokenizer.encode(filler)
    reps = budget // max(len(filler_ids), 1) + 2
    haystack_ids = (filler_ids * reps)[:budget]
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
        try:
            with torch.inference_mode():
                output = model.generate(input_ids, max_new_tokens=10, do_sample=False)
            generated_text = tokenizer.decode(
                output[0][input_ids.size(1) :], skip_special_tokens=True
            )
            if code in generated_text:
                successes += 1
        except Exception:
            pass
    return round(successes / n_trials, 3)


def measure_memory_and_latency(model, tokenizer, seq_len, device="cpu"):
    dummy = torch.randint(0, min(1000, tokenizer.vocab_size), (1, seq_len))
    start = time.perf_counter()
    with torch.inference_mode():
        _ = model(dummy)
    elapsed = time.perf_counter() - start
    return {"vram_mb": 0.0, "latency_ms": round((elapsed * 1000) / seq_len, 3)}


def eval_model(
    model_key,
    eval_text,
    eval_lengths=[512, 1024, 2048, 4096, 8192],
    needle_depths=[0.25, 0.50, 0.75, 0.90],
    needle_trials=3,
):
    info = MODELS_TO_EVAL[model_key]
    device = "cpu"
    print(f"\n{'=' * 50}", flush=True)
    print(f"Evaluating: {info['name']} ({info['hf_model']})", flush=True)
    print(f"{'=' * 50}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(info["hf_model"])
    model = AutoModelForCausalLM.from_pretrained(
        info["hf_model"], torch_dtype=torch.float32, trust_remote_code=True
    ).to(device)
    model.eval()

    is_hard_limit = info.get("has_fixed_pos_limit", False)
    max_pos = info.get("max_pos", 1024)

    results = {
        "model_name": info["name"],
        "model_type": info["type"],
        "trained_max_pos": max_pos,
        "is_hard_limit": is_hard_limit,
        "perplexity": {},
        "needle_accuracy": {},
        "overhead": {},
    }

    for L in eval_lengths:
        print(f"\n  --- L={L} ---", flush=True)
        if is_hard_limit and L > max_pos:
            print(f"  [SKIP] L={L} exceeds hard limit {max_pos}", flush=True)
            results["perplexity"][L] = f"Skipped (hard limit {max_pos})"
            results["needle_accuracy"][L] = {d: None for d in needle_depths}
            results["overhead"][L] = {"vram_mb": 0, "latency_ms": 0}
            continue

        try:
            t0 = time.time()
            ppl = evaluate_perplexity(
                model, tokenizer, eval_text, seq_len=L, device=device
            )
            dt = time.time() - t0
            results["perplexity"][L] = ppl
            print(f"  PPL={ppl} ({dt:.1f}s)", flush=True)
        except Exception as e:
            results["perplexity"][L] = "Failed"
            print(f"  PPL failed: {e}", flush=True)

        try:
            overhead = measure_memory_and_latency(
                model, tokenizer, seq_len=L, device=device
            )
            results["overhead"][L] = overhead
            print(f"  Latency: {overhead['latency_ms']} ms/token", flush=True)
        except Exception as e:
            results["overhead"][L] = {"vram_mb": 0, "latency_ms": 0}

        results["needle_accuracy"][L] = {}
        for d in needle_depths:
            try:
                t0 = time.time()
                acc = evaluate_needle(
                    model,
                    tokenizer,
                    seq_len=L,
                    depth_ratio=d,
                    device=device,
                    n_trials=needle_trials,
                )
                dt = time.time() - t0
                results["needle_accuracy"][L][d] = acc
                print(
                    f"  Needle@{d * 100:.0f}%={acc * 100:.0f}% ({dt:.1f}s)", flush=True
                )
            except Exception as e:
                results["needle_accuracy"][L][d] = None
                print(f"  Needle@{d * 100:.0f}% failed: {e}", flush=True)

    del model, tokenizer
    gc.collect()
    return results


if __name__ == "__main__":
    eval_lengths = [512, 1024, 2048, 4096, 8192]
    needle_depths = [0.25, 0.50, 0.75, 0.90]
    needle_trials = 3

    print("Downloading evaluation text...", flush=True)
    eval_text = download_eval_text(min_chars=500000)
    print(f"Total eval text: {len(eval_text)} chars", flush=True)

    all_results = {}
    for model_key in MODELS_TO_EVAL.keys():
        try:
            res = eval_model(
                model_key, eval_text, eval_lengths, needle_depths, needle_trials
            )
            all_results[model_key] = res
        except Exception as e:
            print(f"FAILED {model_key}: {e}", flush=True)
            import traceback

            traceback.print_exc()

        os.makedirs("results", exist_ok=True)
        with open("results/colab_benchmark_results.json", "w") as f:
            json.dump(all_results, f, indent=2, default=str)

    print("\n" + "=" * 60)
    print("BENCHMARK COMPLETE")
    print("=" * 60)

    header = f"{'Model':<30} | " + " | ".join([f"L={L:<6}" for L in eval_lengths])
    print(f"\n{header}")
    print("-" * len(header))
    for key, res in all_results.items():
        row = f"{res['model_name']:<30} | "
        row += " | ".join(
            [f"{str(res['perplexity'].get(L, 'N/A'))[:8]:<8}" for L in eval_lengths]
        )
        print(row)

    print("\nNeedle Retrieval Accuracy:")
    for key, res in all_results.items():
        for L in eval_lengths:
            na = res.get("needle_accuracy", {}).get(L, {})
            vals = [f"{na.get(d, 'N/A')}" for d in needle_depths]
            print(f"  {res['model_name'][:25]:<25} L={L:<5} depths={vals}")

    print("\nInference Latency (ms/token):")
    for key, res in all_results.items():
        for L in eval_lengths:
            oh = res.get("overhead", {}).get(L, {})
            print(
                f"  {res['model_name'][:25]:<25} L={L:<5} latency={oh.get('latency_ms', 'N/A')} ms/tok"
            )
