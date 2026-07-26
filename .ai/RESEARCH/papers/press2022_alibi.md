# Press, Smith & Lewis (2022) — "Train Short, Test Long: Attention with Linear Biases"

## Bibliography Key
press2022

## Venue
ICLR 2022

## Methodology
- Proposed Attention with Linear Biases (ALiBi): adds a distance-proportional penalty to query-key attention scores instead of learned/sinusoidal position embeddings
- ALiBi bias: $b_{i,j} = -r \cdot |i-j|$ where $r$ is a per-head slope
- Trained on short sequences (1024) and evaluated on long sequences (2048+)
- Evaluated on WikiText-103 and other benchmarks

## Strengths
- Strongest zero-shot extrapolation among methods compared in this study
- No learned per-position parameters; zero parameter overhead for positional encoding
- Extremely low implementation complexity: static bias addition to attention scores
- No fine-tuning required for longer sequences (plug-and-play)
- Outperforms multiple strong PE methods on WikiText-103
- 11% faster training and 11% less memory compared to sinusoidal PE at equivalent performance

## Weaknesses
- Induces strict recency bias by construction, which can hinder long-range multi-hop retrieval
- Per-head slopes are chosen at pretraining time and are not adaptive to input content
- The bias function can become too negative for very distant tokens, suppressing all attention to distant content
- No learned parameters means limited flexibility to adapt the bias to different tasks

## Zero-Shot Capability
- High: ALiBi extrapolates well out-of-the-box without any fine-tuning

## Fine-Tuning Requirements
- None: no fine-tuning required to run at longer lengths (the bias function is fixed and well-defined at any length)

## Compute Cost
- Very low: static matrix addition to attention scores; no learned parameters

## Implementation Complexity
- Very low: simple additive bias based on distance

## Long-Context Retrieval
- Poor: strict recency bias suppresses distant token associations, hindering long-range multi-hop retrieval

## Recency Bias
- Strong: penalty is proportional to distance, producing a hard recency bias by construction

## Limitations
- Recency bias is too strong for tasks requiring distant token relationships
- Fixed slopes per head do not adapt to input-specific needs
- The paper's evaluation focused on perplexity; retrieval tasks were not the primary evaluation axis

## Research Gaps
- The paper did not explore content-conditioned or adaptive biases
- No exploration of hybrid approaches combining ALiBi's extrapolation with RoPE's long-range signal preservation