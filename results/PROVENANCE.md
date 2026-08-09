# Data Provenance

## Authoritative result files

| File | Status | Notes |
|------|--------|-------|
| `results/colab_benchmark_results.json` | **AUTHORITATIVE** | Zero-shot benchmark of pretrained GPT-2 (Learned), Pythia-160M (RoPE), BLOOM-560M (ALiBi). Used by the paper tables and figures. |
| `results/controlled_results.json` | **AUTHORITATIVE** | Controlled from-scratch benchmark (learned / rope / alibi, identical ~20M architecture trained on WikiText-103). Primary evidence in the paper. |
| `results/archive/learned_absolute_results.json` | **SUPERSEDED** | Earlier GPT-2-only run with different perplexity values (57.71 vs 36.92 at L=512) and different needle results. Numbers conflict with the colab run. Kept for provenance only; do not cite. |

## Known data characteristics (documented in the paper)

1. **BLOOM needle=0 at L=4096 while perplexity = CUDA OOM.** The needle harness
   generates with a KV cache, which has lower peak activation memory than the
   full-sequence perplexity forward pass over BLOOM's 250K-token output head.
   The 0.0 retrieval at 4096 is therefore a genuine (model-can't-retrieve) result
   that was measured even though the perplexity forward pass OOM'd.
2. **BLOOM VRAM=0 at L=2048/4096.** `measure_memory_and_latency` recorded 0 MB
   peak VRAM on these rows due to `torch.cuda.max_memory_allocated` being read
   after the allocator had been reset by the preceding OOM recovery. Latency was
   still recorded. The paper reports latency but omits the unreliable VRAM rows.
3. **GPT-2 hard limit at L=1024.** GPT-2's embedding table has 1024 rows; lengths
   beyond that crash. Results beyond 1024 are absent by design (not "bad numbers").
4. **Learned (controlled) uses position clamping** at L > 512 (positions clamp to
   the last training index) rather than a hard crash, mirroring the soft-failure
   behavior reported for learned absolute embeddings in the literature.
 5. **Controlled experiment (2026-08-08).** Three identical 17.6-17.7M-param
    transformers (d=256, 6L, 4H, d_head=64, d_ff=1024, vocab 50257), trained on
    WikiText-103 (~13.0M tokens, 200K-row stream cap) at seq 512, batch 16
    (micro 8 x accum 2), 10K steps, AdamW lr 3e-4 wd 0.1, warmup 1000, cosine
    decay, fp16 AMP. Only positional scheme differs: learned absolute, RoPE,
    ALiBi. Seed 42 per method. Val ppl: learned 110.2, rope 88.1, alibi 126.9
    (a value of 93.7 recorded in an earlier draft was verified incorrect by
    retraining; see `alibi_retrain_verification.json`). Eval: sliding-window
    ppl at 512-8192 on WikiText-103 validation; needle retrieval all 0.0
    (17M models lack retrieval capacity, not a PE effect); latency
    75.4/67.1/66.8K tok/s.
6. **Controlled checkpoints** live in Google Drive
   `MyDrive/AOAPaper_checkpoints/{learned,rope,alibi}/` (resume.pt + best.pt);
   local copies on Colab `/content/checkpoints/`. Checkpoints back up every
   1000 steps; Drive sync is best-effort and warned on failure.
