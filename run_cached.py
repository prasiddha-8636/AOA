"""
Benchmark using cached eval text. Run per-model.
Usage: python run_cached.py <model_key>
"""

import sys, os, json, time, gc, random, math

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
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
        "name": "BLOOM-560M (ALiBi)",
        "hf": "bigscience/bloom-560m",
        "type": "Linear Biases (ALiBi)",
        "hard_limit": False,
        "max_pos": 2048,
    },
}


def get_eval_text():
    with open("/tmp/eval_text.txt", "r", errors="ignore") as f:
        return f.read()


def eval_ppl(model, tokenizer, text, seq_len, stride=512):
    text = text[:65536]
    enc = tokenizer(text, return_tensors="pt")
    total = enc.input_ids.size(1)
    if total < seq_len:
        return float("nan"), 0.0
    nlls = []
    prev = 0
    t0 = time.time()
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
    dt = time.time() - t0
    if not nlls:
        return float("nan"), dt
    return round(math.exp(sum(nlls) / prev), 2), dt


def eval_needle(model, tokenizer, seq_len, depth_ratio, n_trials=3):
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
        full = t + q_ids
        inp = torch.tensor([full])
        try:
            with torch.inference_mode():
                out = model.generate(inp, max_new_tokens=10, do_sample=False)
            gen = tokenizer.decode(out[0][inp.size(1) :], skip_special_tokens=True)
            if code in gen:
                successes += 1
        except:
            pass
    return round(successes / n_trials, 3)


if __name__ == "__main__":
    mk = sys.argv[1]
    info = MODELS[mk]
    eval_lengths = [512, 1024, 2048, 4096, 8192]
    needle_depths = [0.25, 0.50, 0.75, 0.90]
    ntrials = 3

    print(f"\n{'=' * 50}\n{info['name']}\n{'=' * 50}", flush=True)

    text = get_eval_text()
    print(f"Eval text: {len(text)} chars", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(info["hf"])
    model = AutoModelForCausalLM.from_pretrained(
        info["hf"], torch_dtype=torch.float32, trust_remote_code=True
    )
    model.eval()
    print(
        f"Model loaded. Params: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M",
        flush=True,
    )

    res = {
        "model_name": info["name"],
        "model_type": info["type"],
        "trained_max_pos": info["max_pos"],
        "is_hard_limit": info["hard_limit"],
        "perplexity": {},
        "needle_accuracy": {},
        "overhead": {},
    }

    for L in eval_lengths:
        print(f"\n--- L={L} ---", flush=True)
        if info["hard_limit"] and L > info["max_pos"]:
            res["perplexity"][L] = f"Skipped (hard limit {info['max_pos']})"
            res["needle_accuracy"][L] = {d: None for d in needle_depths}
            res["overhead"][L] = {"vram_mb": 0, "latency_ms": 0}
            print("  Skipped (hard limit)", flush=True)
            continue

        ppl, dt = eval_ppl(model, tokenizer, text, L)
        res["perplexity"][L] = ppl
        print(f"  PPL={ppl} ({dt:.1f}s)", flush=True)

        try:
            dummy = torch.randint(0, min(1000, tokenizer.vocab_size), (1, L))
            t0 = time.time()
            with torch.inference_mode():
                _ = model(dummy)
            lat = round(((time.time() - t0) * 1000) / L, 3)
            res["overhead"][L] = {"vram_mb": 0, "latency_ms": lat}
            print(f"  Latency={lat} ms/tok", flush=True)
        except:
            res["overhead"][L] = {"vram_mb": 0, "latency_ms": 0}

        res["needle_accuracy"][L] = {}
        for d in needle_depths:
            t0 = time.time()
            acc = eval_needle(model, tokenizer, L, d, ntrials)
            res["needle_accuracy"][L][d] = acc
            print(
                f"  Needle@{d * 100:.0f}%={acc * 100:.0f}% ({time.time() - t0:.1f}s)",
                flush=True,
            )

    os.makedirs("results", exist_ok=True)
    out = f"results/{mk}_results.json"
    with open(out, "w") as f:
        json.dump(res, f, indent=2, default=str)
    print(f"\nSaved {out}", flush=True)
