# Kazemnejad et al. (2023) — "The Impact of Positional Encoding on Length Generalization in Transformers"

## Bibliography Key
kazemnejad2023

## Venue
NeurIPS 2023

## Methodology
- Systematic empirical study comparing five PE schemes on decoder-only Transformers:
  1. Absolute Position Embedding (APE)
  2. T5's Relative PE
  3. ALiBi
  4. Rotary (RoPE)
  5. NoPE (no explicit positional encoding)
- Evaluated on battery of reasoning and mathematical tasks
- Used different length generalization splits from corresponding datasets
- Three seeds per dataset-PE pair

## Strengths
- First systematic empirical comparison of PE methods on length generalization (not just perplexity)
- Demonstrates that ALiBi, Rotary, and APE are NOT well suited for length generalization in downstream tasks
- NoPE outperforms all explicit PE methods while requiring no additional computation
- Theoretical proof that NoPE can represent both absolute and relative PEs
- Shows that NoPE trained with SGD mostly resembles T5's relative PE attention patterns

## Weaknesses
- Evaluation focused on small-scale synthetic tasks; may not fully generalize to real-world long-document tasks
- NoPE's success on small tasks does not guarantee success on all downstream tasks
- The scratchpad format findings are specific to reasoning tasks
- The paper used decoder-only Transformers only; encoder-decoder models may behave differently

## Zero-Shot Capability
- Medium-High for NoPE: surprisingly strong on length generalization tasks without explicit PE
- Low for APE, ALiBi, Rotary, T5-PE: these methods fail on downstream tasks despite working well on perplexity

## Fine-Tuning Requirements
- None for NoPE: works without any positional encoding

## Compute Cost
- Very low for NoPE: no positional parameters or computation

## Implementation Complexity
- Very low for NoPE: simply omit positional encoding

## Long-Context Retrieval
- Not directly evaluated; the paper focused on reasoning and mathematical tasks

## Recency Bias
- None for NoPE: NoPE has no explicit recency mechanism but still generalizes well

## Limitations
- Findings are based on small-scale synthetic tasks; real-world long-document scenarios may differ
- ALiBi's recency bias was shown to be problematic for retrieval tasks in other works
- The paper's "NoPE is sufficient" conclusion is specific to decoder-only architectures and reasoning tasks

## Research Gaps
- The surprising NoPE result needs replication on diverse architectures and tasks
- Whether NoPE works for encoder-decoder models was not explored
- The relationship between attention stability (which NoPE provides) and explicit positional encoding needs further investigation