# Future Work

## Potential Directions (Evidence-Driven)

### 1. Hybrid Positional Encodings
- Combine relative coordinate properties (RoPE) with implicit positional indicators (NoPE) to potentially capture the benefits of both
- BiPE demonstrates that disentangling local and global position modeling works; combining with RoPE extensions could be explored
- CABLE's content-conditioned bias could be integrated with RoPE-based models

### 2. Content-Adaptive Extrapolation
- CABLE (Veisi 2025) shows that content-conditioned biases reduce recency bias without sacrificing extrapolation
- Research needed at larger model scales and diverse architectures
- Could be applied as a post-hoc modification to existing RoPE models

### 3. Unified, Downstream Task-Based Benchmarks
- Most evaluations rely on perplexity; the field needs benchmarks that measure real-world length generalization
- Tasks requiring genuine long-range dependency (multi-hop QA, long-document QA, retrieval) are under-represented
- Zhao et al. (2024) identify this as a key challenge

### 4. Scaling Content-Adaptive Methods
- CABLE and BiPE need evaluation at larger scales (70B+ models, diverse architectures)
- The recency bias trade-off needs characterization at scale

### 5. Post-Hoc Methods That Don't Require Fine-Tuning
- The gap between ALiBi's plug-and-play operation and RoPE's strong extrapolation (but fine-tuning requirement) remains
- Methods that combine ALiBi's zero-shot capability with RoPE's long-range retrieval are an open problem

### 6. Replication of NoPE Findings
- Kazemnejad et al. (2023) showed NoPE outperforms explicit PE on reasoning tasks
- Findings need replication on real-world tasks, diverse architectures, and at scale
- Whether NoPE is sufficient for production systems remains an open question

### 7. Automatic Data-Driven Segmentation
- BiPE's segment boundaries depend on heuristic sentence detection
- Automatic, data-driven segmentation could replace these heuristics

### 8. Integration of RoPE Extensions with Content-Conditioned Biases
- LongRoPE2's near-lossless short-context preservation + CABLE's content-adaptive bias could yield state-of-the-art results
- BiPE's disentangled position modeling + YaRN's frequency-selective interpolation is another unexplored combination