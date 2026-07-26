# TODO

## Legend
- [ ] Not started
- [x] Complete

## Research Phase — COMPLETE
- [x] Read all 20 literature papers and create structured notes in .ai/RESEARCH/papers/
- [x] Compare papers across 11 dimensions (methodology, strengths, weaknesses, zero-shot, fine-tuning, compute, implementation, retrieval, recency, limitations, gaps)
- [x] Update comparison matrix in .ai/RESEARCH/comparison_matrix.md
- [x] Update gap analysis in .ai/RESEARCH/gap_analysis.md
- [x] Update future work in .ai/RESEARCH/future_work.md

## Methodology Design — COMPLETE
- [x] Design theoretical framework and justification
- [x] Design controlled experiment (8 PE methods, fixed architecture)
- [x] Define data collection (WikiText-103, PG-19, synthetic)
- [x] Define preprocessing pipeline
- [x] Implement PE modules (Learned, Sinusoidal, RoPE, ALiBi, NoPE, KERPLE, CABLE, PI)
- [x] Implement Transformer model with modular PE integration
- [x] Define evaluation metrics and protocols
- [x] Define analysis pipeline with figure generation
- [x] Update .ai/IMPLEMENTATION/ with all files

## Epic 1: Citation Verification [High Priority]
- [ ] Verify all 25 bibliography entries against source papers
- [ ] Cross-check for duplicate bibitems
- [ ] Verify all \cite keys have matching \bibitem entries
- [ ] Confirm every \bibitem is referenced by at least one \cite

## Epic 2: Experiments [High Priority]
- [ ] Train Learned Absolute PE (100K steps)
- [ ] Train Sinusoidal PE (100K steps)
- [ ] Train RoPE (100K steps)
- [ ] Train ALiBi (100K steps)
- [ ] Train NoPE (100K steps)
- [ ] Train KERPLE (100K steps)
- [ ] Train CABLE (100K steps)
- [ ] Fine-tune Position Interpolation from RoPE (5K steps)
- [ ] Evaluate all methods at lengths 512-8192
- [ ] Profile compute (FLOPs, throughput, parameters)

## Epic 3: Analysis & Figures [High Priority]
- [ ] Run analysis.ipynb end-to-end
- [ ] Generate Figure 1: PPL vs Sequence Length
- [ ] Generate Figure 2: Needle-in-Haystack Accuracy
- [ ] Generate Figure 3: Recency Bias Bar Chart
- [ ] Generate Figure 4: Compute vs Extrapolation Scatter
- [ ] Generate Figure 5: Attention Map Comparison
- [ ] Generate Table 1: Experimental Comparison

## Epic 4: Paper Update [Medium Priority]
- [ ] Rewrite Methodology section based on experimental design
- [ ] Update Comparative Analysis with experimental results
- [ ] Update Conclusion with experimental findings
- [ ] Verify all \cite keys have matching \bibitem entries

## Epic 5: Proofreading & Format [Low Priority]
- [ ] Proofread Introduction section
- [ ] Proofread Literature Review section
- [ ] Proofread Methodology section
- [ ] Proofread Comparative Analysis section
- [ ] Proofread Conclusion section
- [ ] Clean rebuild and verify PDF output

## Epic 6: Documentation [Low Priority]
- [ ] Update .ai/RESEARCH/ with experimental findings
- [ ] Update STATUS.md, TODO.md, TASKS/ as items are completed