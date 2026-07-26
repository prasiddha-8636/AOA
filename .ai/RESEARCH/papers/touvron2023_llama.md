# Touvron et al. (2023) — "LLaMA: Open and Efficient Foundation Language Models"

## Bibliography Key
touvron2023

## Venue
arXiv preprint arXiv:2302.13971, 2023

## Methodology
- Trained foundation LLaMA models (7B to 65B) on trillions of tokens from public datasets
- Uses RoPE (Rotary Position Embedding) as the positional encoding method
- Transformer decoder-only architecture

## Strengths
- Demonstrates that competitive models can be trained using only publicly available data
- LLaMA-13B outperforms GPT-3 (175B) on most benchmarks
- LLaMA-65B competitive with Chinchilla-70B and PaLM-540B
- Established RoPE as the de facto positional encoding for modern LLMs
- Open-source release enabled widespread research

## Weaknesses
- Context window limited to 2048 tokens during pretraining (requires extension methods for longer contexts)
- RoPE alone does not extrapolate beyond the training length without additional methods (PI, YaRN, etc.)

## Zero-Shot Capability
- N/A (LLaMA itself uses RoPE which has limited zero-shot extrapolation)

## Fine-Tuning Requirements
- RoPE requires post-training extension methods (Position Interpolation, YaRN) for longer contexts, involving fine-tuning

## Compute Cost
- Moderate: RoPE adds zero extra parameters and negligible latency; but the base model itself requires massive compute for pretraining

## Implementation Complexity
- Low for RoPE: element-wise rotation applied to query and key vectors
- Moderate for full model training and deployment

## Long-Context Retrieval
- Not directly evaluated for long-context retrieval in the base LLaMA paper; extension methods (PI, YaRN) enable longer context usage

## Recency Bias
- No explicit recency bias from RoPE itself; long-term attention decay emerges from the superposition of rotations across dimensions

## Limitations
- Original context window (2048 tokens) is too short for many long-document tasks
- Requires follow-up methods (PI, YaRN, LongRoPE) for context window extension

## Research Gaps
- The paper focuses on scaling rather than extrapolation; length generalization is a separate concern addressed by subsequent works