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
