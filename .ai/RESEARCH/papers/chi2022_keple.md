# Chi et al. (2022) — "KERPLE: Kernelized Relative Positional Embedding for Length Extrapolation"

## Bibliography Key
chi2022

## Venue
NeurIPS 2022, pp. 8386–8399

## Methodology
- Proposed KERPLE: kernelized relative positional embeddings using conditionally positive definite (CPD) kernels
- Generalizes relative position embedding by kernelizing positional differences
- CPD kernels are transformed into positive definite (PD) kernels by adding a constant offset
- The offset is implicitly absorbed in Softmax normalization during self-attention
- Two practical variants: logarithmic kernel and Gaussian-like kernel
- The logarithmic variant achieves excellent extrapolation performance
- Evaluated on OpenWebText2, GitHub, and ArXiv datasets
- Compared against Sinusoidal, Rotary, T5, and ALiBi baselines

## Strengths
- Principled kernel-based framework for designing new relative position encodings
- Logarithmic kernel provides excellent extrapolation performance
- Only 24 additional learnable parameters (2 per head) — very parameter-efficient
- Same training speed as window attention despite much better performance
- Can be combined with window attention for even better results (KERPLE-log-windowed)
- Theoretically grounded in CPD kernel theory

## Weaknesses
- Kernel function selection is a design choice with no automatic mechanism for determining the best kernel
- Small number of learnable parameters means limited adaptability
- The log variant can approach zero for very large distances, which limits its effective range
- KERPLE is more complex to implement than ALiBi or RoPE
- Not content-adaptive: the kernel bias does not depend on token content

## Zero-Shot Capability
- High: KERPLE-log achieves strong zero-shot extrapolation, competitive with ALiBi

## Fine-Tuning Requirements
- No for KERPLE-log: plug-and-play like ALiBi, since the log kernel has no learnable slope parameters to adapt
- KERPLE with windowed attention requires some configuration but no fine-tuning per se

## Compute Cost
- Moderate: kernel computation adds some overhead compared to ALiBi's simple bias addition
- The learnable parameters are tiny (24 total), so parameter overhead is negligible

## Implementation Complexity
- Medium-high: kernel functions, CPD-to-PD transformation, and windowed attention add structural complexity
- More complex than ALiBi or RoPE but well-defined mathematically

## Long-Context Retrieval
- Good: KERPLE-log maintains attention to distant tokens better than ALiBi due to the log kernel's slower decay
- The flat kernels of the log variant help preserve long-range attention

## Recency Bias
- Milder than ALiBi: the logarithmic kernel decays more slowly than the linear ALiBi penalty, reducing the recency bias
- The log variant discerns several flat kernels, extending the effective attention window

## Limitations
- Kernel selection is a design choice with no principled automatic method
- The log kernel can approach zero for very large distances, limiting effective range
- More complex implementation than ALiBi or RoPE

## Research Gaps
- The paper focused on univariate kernels; multivariate kernel combinations could capture more nuanced distance effects
- The relationship between different CPD kernels (exponential, Gaussian, polynomial, logarithmic) and their combinations was not fully explored in the original paper