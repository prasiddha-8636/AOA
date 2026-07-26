# Project

## Objective
Conduct a comparative study of positional encoding methods (Learned Absolute, RoPE, and ALiBi) for Transformer length extrapolation using zero-shot inference evaluation of pre-trained models.

## Research Question
How do Learned Absolute Positional Embeddings (GPT-2), Rotary Positional Encoding (Pythia), and ALiBi (OPT) compare in zero-shot length extrapolation, long-context retrieval accuracy, and GPU memory/latency overhead?

## Project Type
Work-in-progress empirical study & survey.

## Scope & Resource Constraints
Designed strictly for execution on Google Colab free-tier hardware (T4 GPU, 15GB VRAM) or a personal computer.
- No model training from scratch.
- Zero-shot forward-pass inference on open-weight Hugging Face checkpoints (`gpt2`, `EleutherAI/pythia-160m`, `facebook/opt-350m`).

## Deliverables
- IEEE conference paper draft (`IEEE-conference-template-062824/main.tex`).
- Structured literature review notes (`.ai/RESEARCH/papers/`).
- Lightweight Colab evaluation script for zero-shot benchmarks.
