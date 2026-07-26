# Vaswani et al. (2017) — "Attention is All You Need"

## Bibliography Key
vaswani2017

## Venue
NeurIPS 2017, pp. 5998–6008

## Methodology
- Introduced the Transformer architecture and the original sinusoidal positional encoding
- Self-attention mechanism without recurrence or convolution
- Sinusoidal PE: $PE_{(pos,2i)} = \sin(pos/10000^{2i/d_{model}})$, $PE_{(pos,2i+1)} = \cos(pos/10000^{2i/d_{model}})$
- Fixed function, no learned parameters

## Strengths
- Foundational work; established the Transformer as the dominant architecture
- Sinusoidal PE is mathematically defined for any position, enabling theoretical extrapolation
- Zero parameter overhead for positional encoding
- Parallelizable computation

## Weaknesses
- Sinusoidal PE does not generalize well in practice: models rarely learn the underlying rotational structure well enough to extrapolate past the training window
- Not designed for length extrapolation; the original paper focused on machine translation within fixed context lengths

## Zero-Shot Capability
- Low: sinusoidal PE is defined analytically but models fail to extrapolate in practice

## Fine-Tuning Requirements
- N/A (original model; no extrapolation method)

## Compute Cost
- Very low: static trigonometric computation at initialization only

## Implementation Complexity
- Very low: simple sine/cosine formula applied to position indices

## Long-Context Retrieval
- Not evaluated for long-context retrieval in the original paper

## Recency Bias
- None: sinusoidal PE does not impose any distance-based penalty

## Limitations
- No extrapolation capability in practice despite theoretical applicability
- Fixed context window limit

## Research Gaps
- The paper did not address length extrapolation as a design goal
- Open question: why sinusoidal PE fails to extrapolate despite being mathematically defined for all positions