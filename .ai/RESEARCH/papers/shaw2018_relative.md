# Shaw et al. (2018) — "Self-Attention with Relative Position Representations"

## Bibliography Key
shaw2018

## Venue
NAACL-HLT 2018, pp. 464–468

## Methodology
- First to introduce relative position representations to self-attention
- Learned pairwise offset vectors $a^K_{ij}$ and $a^V_{ij}$ for query-key and key-value interactions
- Clipping function limits the relative position range to $[-k, k]$
- Transformer encoder-decoder architecture evaluated on WMT translation tasks

## Strengths
- Demonstrated that relative position encoding improves translation quality over absolute PE (1.3 BLEU gain on WMT En-De)
- Translation invariant: relative position representations generalize better than absolute ones
- Conceptually elegant: encodes relationships rather than positions

## Weaknesses
- Pairwise offset vectors scale poorly in memory: $O(L^2 d)$ complexity
- Clipping limits the effective range of relative positions
- Requires fine-tuning for different context lengths
- Not designed for length extrapolation

## Zero-Shot Capability
- Medium: relative representation is translation invariant but clipping limits extreme extrapolation

## Fine-Tuning Requirements
- Yes: learned relative position embeddings are tied to the training context length

## Compute Cost
- High: pairwise offset matrices scale quadratically in sequence length

## Implementation Complexity
- Medium-high: computing and storing pairwise offset matrices for all positions

## Long-Context Retrieval
- Limited: clipping and memory constraints prevent effective long-range retrieval

## Recency Bias
- Implicit through relative position encoding, but no explicit distance penalty

## Limitations
- Quadratic memory scaling makes it impractical for very long sequences
- Clipping function creates a hard boundary for relative position information

## Research Gaps
- Shaw's RPE was generalized by T5 (bucketing), KERPLE (kernelized), and RoPE (rotary), each addressing different aspects of the relative PE framework