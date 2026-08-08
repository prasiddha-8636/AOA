"""
Self-contained benchmark for Colab T4. Downloads eval text from Gutenberg,
loads the 3 pretrained models, evaluates PPL / needle / latency at all lengths.
Prints JSON at the end.
"""

import json, math, os, random, sys, time, urllib.request
import numpy as np

if not hasattr(np, "long"):
    np.long = np.int64
if not hasattr(np, "ulong"):
    np.ulong = np.uint64
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

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
        "name": "OPT-350M (ALiBi)",
        "hf": "facebook/opt-350m",
        "type": "Linear Biases (ALiBi)",
        "hard_limit": False,
        "max_pos": 2048,
    },
}

EVAL_URLS = [
    "https://www.gutenberg.org/files/2701/2701-0.txt",
    "https://www.gutenberg.org/files/1342/1342-0.txt",
]


def get_eval_text(min_chars=600000):
    chunks, total = [], 0
    for url in EVAL_URLS:
        if total >= min_chars:
            break
        try:
            data = (
                urllib.request.urlopen(url, timeout=60)
                .read()
                .decode("utf-8", errors="ignore")
            )
            chunks.append(data)
            total += len(data)
            print(
                f"  eval text: {len(data)} chars from {url.split('/')[-1]}", flush=True
            )
        except Exception as e:
            print(f"  download failed {url}: {e}", flush=True)
    return "\n\n".join(chunks)


def eval_ppl(model, tokenizer, text, seq_len, device, stride=512):
    text = text[:65536]
    enc = tokenizer(text, return_tensors="pt").to(device)
    total = enc.input_ids.size(1)
    if total < seq_len:
        return float("nan")
    nlls, prev = [], 0
    with torch.inference_mode():
        for b in range(0, total, stride):
            e = min(b + seq_len, total)
            tl = e - prev
            if tl <= 0:
                break
            ids = enc.input_ids[:, b:e]
            tids = ids.clone()
            tids[:, :-tl] = -100
            out = model(ids, labels=tids)
            nlls.append(out.loss.item() * tl)
            prev = e
            if e == total:
                break
    if not nlls:
        return float("nan")
    return round(math.exp(sum(nlls) / prev), 2)


def eval_needle(model, tokenizer, seq_len, depth_ratio, device, n_trials=3):
    q_ids = tokenizer.encode("\nWhat is the secret code? Answer:")
    budget = max(seq_len - len(q_ids), 1)
    f_ids = tokenizer.encode("The quick brown fox jumps over the lazy dog. ")
    reps = budget // max(len(f_ids), 1) + 2
    h_ids = (f_ids * reps)[:budget]
    successes = 0
    for _ in range(n_trials):
        code = str(random.randint(10000, 99999))
        n_ids = tokenizer.encode(f"The secret code is {code}.")
        t = list(h_ids)
        idx = min(int(len(t) * depth_ratio), max(len(t) - len(n_ids), 0))
        t[idx : idx + len(n_ids)] = n_ids
        full = torch.tensor([t + q_ids], device=device)
        with torch.inference_mode():
            out = model.generate(
                full,
                max_new_tokens=10,
                do_sample=False,
                attention_mask=torch.ones_like(full),
                pad_token_id=tokenizer.eos_token_id,
            )
        gen = tokenizer.decode(out[0][full.size(1) :], skip_special_tokens=True)
        if code in gen:
            successes += 1
    return round(successes / n_trials, 3)


def eval_model(mk, text, device):
    info = MODELS[mk]
    print(f"\n{'=' * 50}\n{info['name']}\n{'=' * 50}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(info["hf"])
    model = AutoModelForCausalLM.from_pretrained(
        info["hf"], torch_dtype=torch.float16
    ).to(device)
    model.eval()

    res = {
        "model_name": info["name"],
        "model_type": info["type"],
        "trained_max_pos": info["max_pos"],
        "is_hard_limit": info["hard_limit"],
        "perplexity": {},
        "needle_accuracy": {},
        "overhead": {},
    }
    for L in [512, 1024, 2048, 4096, 8192]:
        print(f"\n--- L={L} ---", flush=True)
        if info["hard_limit"] and L > info["max_pos"]:
            res["perplexity"][L] = f"Skipped (hard limit {info['max_pos']})"
            res["needle_accuracy"][L] = {d: None for d in [0.25, 0.5, 0.75, 0.9]}
            res["overhead"][L] = {"vram_mb": 0, "latency_ms": 0}
            print("  skipped (hard limit)", flush=True)
            continue

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        try:
            t0 = time.time()
            ppl = eval_ppl(model, tokenizer, text, L, device)
            print(f"  PPL={ppl} ({time.time() - t0:.1f}s)", flush=True)
            res["perplexity"][L] = ppl
        except Exception as e:
            res["perplexity"][L] = "Failed"
            print(f"  PPL failed: {e}", flush=True)

        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            dummy = torch.randint(
                0, min(1000, tokenizer.vocab_size), (1, L), device=device
            )
            t0 = time.time()
            with torch.inference_mode():
                _ = model(dummy)
            torch.cuda.synchronize()
            lat = round(((time.time() - t0) * 1000) / L, 3)
            vram = round(torch.cuda.max_memory_allocated() / (1024 * 1024), 1)
            res["overhead"][L] = {"vram_mb": vram, "latency_ms": lat}
            print(f"  VRAM={vram}MB latency={lat} ms/tok", flush=True)
        except Exception as e:
            res["overhead"][L] = {"vram_mb": 0, "latency_ms": 0}

        res["needle_accuracy"][L] = {}
        for d in [0.25, 0.5, 0.75, 0.9]:
            try:
                t0 = time.time()
                acc = eval_needle(model, tokenizer, L, d, device, n_trials=3)
                res["needle_accuracy"][L][d] = acc
                print(
                    f"  Needle@{d * 100:.0f}%={acc * 100:.0f}% ({time.time() - t0:.1f}s)",
                    flush=True,
                )
            except Exception as e:
                res["needle_accuracy"][L][d] = None
                print(f"  Needle@{d * 100:.0f}% failed: {e}", flush=True)
    del model, tokenizer
    torch.cuda.empty_cache()
    return res


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(
        f"Device: {device} ({torch.cuda.get_device_name(0) if device == 'cuda' else 'cpu'})",
        flush=True,
    )
    print("Downloading eval text...", flush=True)
    text = get_eval_text()
    print(f"Eval text: {len(text)} chars", flush=True)

    all_results = {}
    for mk in MODELS:
        try:
            all_results[mk] = eval_model(mk, text, device)
        except Exception as e:
            print(f"FAILED {mk}: {e}", flush=True)
            import traceback

            traceback.print_exc()
        # Partial save
        with open("/content/results.json", "w") as f:
            json.dump(all_results, f, indent=2, default=str)

    print("\n\nFINAL RESULTS JSON:")
    print(json.dumps(all_results, indent=2, default=str))
    print("\nBENCHMARK COMPLETE")
