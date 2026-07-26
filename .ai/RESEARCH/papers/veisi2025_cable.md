# Veisi, Amirzadeh & Mansourian (2025) — "Context-aware Biases for Length Extrapolation" (CABLE)

## Bibliography Key
veisi2025cable

## Venue
arXiv preprint arXiv:2503.08067, 2025 (Accepted at EMNLP 2025 Main Conference)

## Methodology
- Proposed Context-Aware Biases for Length Extrapolation (CABLE)
- Learns token-specific, context-aware biases for each attention head in transformers
- Unlike ALiBi's fixed global slope per head, CABLE dynamically adjusts positional biases based on the input
- Uses a lightweight per-head network to produce token-specific biases
- Evaluated on GPT-2 Medium (334M parameters) on FineWeb-Edu-10B and WikiText-103
- Also applied to BERT base for long-context retrieval tasks

## Strengths
- Addresses ALiBi's rigidity by making biases content-dependent rather than purely distance-dependent
- Reduces perplexity relative to RoPE, ALiBi, and T5-style bias at sequences beyond training
- Lightweight per-head network adds minimal computational overhead compared to a full attention layer
- Improves long-context retrieval on BERT base, showing benefits for information-intensive tasks
- Content-conditioned bias is less likely to discard a relevant distant token by default

## Weaknesses
- New method with limited evaluation; only tested on GPT-2 Medium and BERT base
- Single paper at moderate model scale; results should be read as promising direction rather than established conclusion
- The per-head network introduces some implementation complexity compared to ALiBi or RoPE
- Not yet integrated into widely-used models or libraries

## Zero-Shot Capability
- Medium-High: CABLE generalizes to longer sequences without architectural changes, though the per-head network needs training

## Fine-Tuning Requirements
- Yes: CABLE is a design choice made during model creation, like ALiBi, not a post-hoc extension

## Compute Cost
- Low for inference; slightly higher than ALiBi due to the per-head network, but still far cheaper than a full attention layer

## Implementation Complexity
- Medium: per-head network for token-specific biases adds implementation overhead compared to ALiBi

## Long-Context Retrieval
- Improved: content-conditioned bias is less likely to discard relevant distant tokens compared to distance-only biases

## Recency Bias
- Softer than ALiBi: a bias that depends on what the tokens are rather than only how far apart they are is less likely to discard relevant distant tokens

## Limitations
- Evaluated on only two model scales (GPT-2 Medium, BERT base)
- Single paper; results need replication at larger scales
- The per-head network's design choices (architecture, size) were not extensively explored

## Research Gaps
- CABLE's effectiveness at larger model scales and diverse architectures is unknown
- The relationship between content-conditioned bias and the recency bias trade-off needs further characterization
- Could CABLE be combined with RoPE extensions (PI, YaRN) for even better results?