# Status

## Current State
Work in progress paper draft. Literature review and research design complete; evaluation scripts undergoing testing.

## Current Task
Prepare lightweight Colab-compatible zero-shot evaluation runner for pre-trained models (`gpt2`, `pythia-160m`, `opt-350m`).

## Hardware Constraints
Google Colab Free Tier (T4 GPU, 15GB VRAM, 12GB System RAM).
Forward-pass zero-shot inference only (`torch.no_grad()`).

## Next Steps
1. Run zero-shot perplexity extrapolation on PG-19 test sequences.
2. Run Needle-in-a-Haystack retrieval at context depths (25%, 50%, 75%, 90%).
3. Record peak GPU memory (MB) and inference latency (ms/token).
4. Populate result figures and tables in `main.tex`.