# Research Gap

## Identified Gaps from Literature Review

### Gap 1: Balanced Comparisons Across Methods
- Most publications introduce or evaluate a single method rather than providing a balanced comparison across representative approaches
- The field lacks standardized, controlled comparisons using the same base model, dataset, and evaluation protocol
- This project addresses that gap by comparing Learned Absolute PE, RoPE, and ALiBi using published literature

### Gap 2: Recency Bias vs. Long-Range Retrieval Trade-off
- ALiBi's strong recency bias is clearly documented (Press et al. 2022) but its implications for retrieval tasks are underexplored
- Most papers evaluate on perplexity, not on retrieval or multi-hop reasoning
- CABLE (Veisi 2025) and BiPE (He 2024) represent early attempts to address this trade-off, but their results are at moderate scale
- The paper notes that NIAH-style evaluations may overstate a method's real long-range retrieval capability (Zhao 2024)

### Gap 3: Content-Adaptive Positional Encoding
- Most methods use fixed or distance-based biases that do not adapt to input content
- CABLE (Veisi 2025) is the first to propose content-conditioned biases, but is evaluated only at moderate scale
- The relationship between content-adaptive bias and length generalization is not well understood

### Gap 4: Post-Hoc Extension Without Fine-Tuning
- RoPE methods (PI, YaRN, LongRoPE2) all require some form of fine-tuning or rescaling
- ALiBi and KERPLE are plug-and-play but have weaker long-range retrieval
- No method currently achieves both plug-and-play operation and strong long-range retrieval

### Gap 5: Decoder-Only NoPE Findings Need Replication
- Kazemnejad et al. (2023) showed NoPE outperforms explicit PE methods on reasoning tasks
- This finding is based on small-scale synthetic tasks and needs replication on diverse architectures and real-world tasks
- Whether NoPE works for encoder-decoder models was not explored

### Gap 6: Unified Benchmarks Beyond Perplexity
- Most evaluations use perplexity on language modeling datasets
- The field lacks unified, downstream task-based benchmarks that measure real-world length generalization
- Zhao et al. (2024) highlight this as a key challenge

### Gap 7: Cross-Layer and Cross-Head Interaction in RoPE
- Liu et al. (2024) identified the critical dimension problem in RoPE but the cross-layer effects are not fully characterized
- How different layers and heads contribute to extrapolation behavior needs more study

### Gap 8: Hybrid Approaches
- Most methods are evaluated in isolation; combining approaches (e.g., BiPE + RoPE + PI, or CABLE + YaRN) is unexplored
- Hybrid encodings that combine relative coordinate properties with content-dependent indicators could address multiple gaps simultaneously