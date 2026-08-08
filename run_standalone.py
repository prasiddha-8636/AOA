"""
Standalone eval for a single model key. Downloads text, runs eval, saves JSON.
Usage: python run_standalone.py <model_key> [eval_text_file]
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

EVAL_URLS = [
    "https://www.gutenberg.org/files/1342/1342-0.txt",
    "https://www.gutenberg.org/files/11/11-0.txt",
    "https://www.gutenberg.org/files/98/98-0.txt",
]


def download_eval_text(min_chars=500000):
    cache_file = "/tmp/eval_text.txt"
    if os.path.exists(cache_file) and os.path.getsize(cache_file) > min_chars // 2:
        with open(cache_file, "r", errors="ignore") as f:
            text = f.read()
        print(f"Using cached eval text: {len(text)} chars", flush=True)
        return text
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
    result = "\n\n".join(chunks)
    with open(cache_file, "w") as f:
        f.write(result)
    return result


def evaluate_perplexity(model, tokenizer, text, seq_len, stride=512):
    encodings = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=tokenizer.model_max_length,
    )
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
            input_ids = encodings.input_ids[:, begin_loc:end_loc]
            target_ids = input_ids.clone()
            target_ids[:, :-trg_len] = -100
            outputs = model(input_ids, labels=target_ids)
            nlls.append(outputs.loss * trg_len)
            prev_end_loc = end_loc
            if end_loc == seq_len_total:
                break
    if not nlls:
        return float("nan")
    return round(torch.exp(torch.stack(nlls).sum() / prev_end_loc).item(), 2)


def evaluate_needle(model, tokenizer, seq_len, depth_ratio, n_trials=3):
    question_ids = tokenizer.encode("\nWhat is the secret code? Answer:")
    budget = max(seq_len - len(question_ids), 1)
    filler_ids = tokenizer.encode("The quick brown fox jumps over the lazy dog. ")
    reps = budget // max(len(filler_ids), 1) + 2
    haystack_ids = (filler_ids * reps)[:budget]
    successes = 0
    for _ in range(n_trials):
        code = str(random.randint(10000, 99999))
        needle_ids = tokenizer.encode(f"The secret code is {code}.")
        trial_ids = list(haystack_ids)
        insert_idx = min(
            int(len(trial_ids) * depth_ratio), max(len(trial_ids) - len(needle_ids), 0)
        )
        trial_ids[insert_idx : insert_idx + len(needle_ids)] = needle_ids
        full_ids = trial_ids + question_ids
        input_ids = torch.tensor([full_ids])
        try:
            with torch.inference_mode():
                output = model.generate(input_ids, max_new_tokens=10, do_sample=False)
            gen = tokenizer.decode(
                output[0][input_ids.size(1) :], skip_special_tokens=True
            )
            if code in gen:
                successes += 1
        except Exception:
            pass
    return round(successes / n_trials, 3)


MODELS = {
    "learned_absolute": {
        "name": "GPT-2 (Learned Absolute)",
        "hf": "gpt2",
        "type": "Learned Absolute",
        "hard_limit": True,
        "max_pos": 1024,
    },
    "rope": {
        "name": "Pythia-160M (RoPE)",
        "hf": "EleutherAI/pythia-160m",
        "type": "Rotary (RoPE)",
        "hard_limit": False,
        "max_pos": 2048,
    },
    "alibi": {
        "name": "BLOOM-560M (ALiBi)",
        "hf": "bigscience/bloom-560m",
        "type": "Linear Biases (ALiBi)",
        "hard_limit": False,
        "max_pos": 2048,
    },
}

if __name__ == "__main__":
    model_key = sys.argv[1]
    eval_lengths = [512, 1024, 2048, 4096, 8192]
    needle_depths = [0.25, 0.50, 0.75, 0.90]
    needle_trials = 3

    info = MODELS[model_key]
    print(f"\n{'=' * 50}", flush=True)
    print(f"Evaluating: {info['name']}", flush=True)
    print(f"{'=' * 50}", flush=True)

    print("Downloading eval text...", flush=True)
    eval_text = download_eval_text(500000)
    print(f"Eval text: {len(eval_text)} chars", flush=True)

    print("Loading tokenizer and model...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(info["hf"])
    model = AutoModelForCausalLM.from_pretrained(
        info["hf"], torch_dtype=torch.float32, trust_remote_code=True
    )
    model.eval()

    results = {
        "model_name": info["name"],
        "model_type": info["type"],
        "trained_max_pos": info["max_pos"],
        "is_hard_limit": info["hard_limit"],
        "perplexity": {},
        "needle_accuracy": {},
        "overhead": {},
    }

    for L in eval_lengths:
        print(f"\n  --- L={L} ---", flush=True)
        if info["hard_limit"] and L > info["max_pos"]:
            print(f"  [SKIP] Hard limit", flush=True)
            results["perplexity"][L] = f"Skipped (hard limit {info['max_pos']})"
            results["needle_accuracy"][L] = {d: None for d in needle_depths}
            results["overhead"][L] = {"vram_mb": 0, "latency_ms": 0}
            continue

        try:
            t0 = time.time()
            ppl = evaluate_perplexity(model, tokenizer, eval_text, L)
            print(f"  PPL={ppl} ({time.time() - t0:.1f}s)", flush=True)
            results["perplexity"][L] = ppl
        except Exception as e:
            results["perplexity"][L] = "Failed"
            print(f"  PPL failed: {e}", flush=True)

        try:
            dummy = torch.randint(0, min(1000, tokenizer.vocab_size), (1, L))
            t0 = time.time()
            with torch.inference_mode():
                _ = model(dummy)
            latency = round(((time.time() - t0) * 1000) / L, 3)
            results["overhead"][L] = {"vram_mb": 0, "latency_ms": latency}
            print(f"  Latency: {latency} ms/tok", flush=True)
        except Exception as e:
            results["overhead"][L] = {"vram_mb": 0, "latency_ms": 0}

        results["needle_accuracy"][L] = {}
        for d in needle_depths:
            try:
                t0 = time.time()
                acc = evaluate_needle(model, tokenizer, L, d, needle_trials)
                print(
                    f"  Needle@{d * 100:.0f}%={acc * 100:.0f}% ({time.time() - t0:.1f}s)",
                    flush=True,
                )
                results["needle_accuracy"][L][d] = acc
            except Exception as e:
                results["needle_accuracy"][L][d] = None
                print(f"  Needle@{d * 100:.0f}% failed: {e}", flush=True)

    os.makedirs("results", exist_ok=True)
    out = f"results/{model_key}_results.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved to {out}", flush=True)
    print(json.dumps(results, indent=2, default=str), flush=True)
