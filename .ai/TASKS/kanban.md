# Kanban

## Backlog

- Create root README.md describing the project — Deps: none
- Create .ai/PLAN.md with this plan snapshot — Deps: none
- Investigate hybrid positional encodings — Deps: review complete
- Evaluate length generalization in non-causal attention models — Deps: review complete
- Establish unified downstream task benchmarks beyond perplexity — Deps: review complete

## Ready

### Epic 1: Citation Verification
- Task 1.1.1: Verify Vaswani et al. 2017 citation — Easy, no deps
- Task 1.1.2: Verify Devlin et al. 2019 citation — Easy, no deps
- Task 1.1.3: Verify Radford et al. 2019 citation — Easy, no deps
- Task 1.1.4: Verify Touvron et al. 2023 citation — Easy, no deps
- Task 1.1.5: Verify Press et al. 2022 citation — Easy, no deps
- Task 1.1.6: Verify Raffel et al. 2020 citation — Easy, no deps
- Task 1.1.7: Verify Shaw et al. 2018 citation — Easy, no deps
- Task 1.2.1: Verify Su et al. 2024 RoFormer citation — Easy, no deps
- Task 1.2.2: Verify Chen et al. 2023 Position Interpolation citation — Easy, no deps
- Task 1.2.3: Verify Liu et al. 2024 Scaling Laws citation — Easy, no deps
- Task 1.2.4: Verify Peng et al. 2023 YaRN citation — Easy, no deps
- Task 1.2.5: Verify Sun et al. 2022 Lexicle citation — Easy, no deps
- Task 1.3.1: Verify Chi et al. 2022 KERPLE citation — Easy, no deps
- Task 1.3.2: Verify Zhao et al. 2024 survey citation — Easy, no deps
- Task 1.3.3: Verify Veisi et al. 2025 CABLE citation — Easy, no deps
- Task 1.3.4: Verify Shang et al. 2025 LongRoPE2 citation — Easy, no deps
- Task 1.3.5: Verify He et al. 2024 BiPE citation — Easy, no deps
- Task 1.3.6: Verify Ruoss et al. 2023 Randomized PE citation — Easy, no deps
- Task 1.3.7: Verify Kazemnejad et al. 2023 NoPE citation — Easy, no deps
- Task 1.3.8: Verify Dai et al. 2019 Transformer-XL citation — Easy, no deps
- Task 1.4.1: Check for duplicate bibitems — Medium, deps: 1.1, 1.2, 1.3
- Task 1.4.2: Verify all \cite keys have matching \bibitem entries — Medium, deps: 1.1, 1.2, 1.3
- Task 1.4.3: Confirm every \bibitem is referenced by at least one \cite — Medium, deps: 1.1, 1.2, 1.3
- Task 1.4.4: Check citation ordering matches IEEE numeric style — Easy, deps: 1.4.2

### Epic 2: Technical Accuracy Review
- Task 2.1.1: Verify Learned Absolute PE claims — Medium, no deps
- Task 2.1.2: Verify sinusoidal PE extrapolation failure claim — Medium, no deps
- Task 2.1.3: Verify representative models for absolute PE — Easy, no deps
- Task 2.2.1: Verify RoPE has zero extra parameters — Easy, no deps
- Task 2.2.2: Verify RoPE phase-shifting instability claim — Medium, no deps
- Task 2.2.3: Verify Position Interpolation requires fine-tuning — Medium, no deps
- Task 2.2.4: Verify YaRN requires fewer fine-tuning steps — Medium, no deps
- Task 2.2.5: Verify LongRoPE2 uses evolutionary search — Medium, no deps
- Task 2.3.1: Verify ALiBi has no learned per-position parameters — Easy, no deps
- Task 2.3.2: Verify ALiBi strong zero-shot extrapolation — Medium, no deps
- Task 2.3.3: Verify ALiBi induces strict recency bias — Medium, no deps
- Task 2.3.4: Verify ALiBi has no per-token runtime overhead — Easy, no deps
- Task 2.4.1: Verify NoPE claim (Kazemnejad et al.) — Medium, no deps
- Task 2.4.2: Verify CABLE claims (single paper, moderate scale) — Medium, no deps
- Task 2.4.3: Verify BiPE claims — Medium, no deps
- Task 2.4.4: Verify KERPLE claims — Medium, no deps
- Task 2.5.1: Verify each row in Table 1 against source claims — Hard, deps: 2.1, 2.2, 2.3, 2.4
- Task 2.5.2: Check Extrapolation ratings match source evidence — Hard, deps: 2.5.1
- Task 2.5.3: Check Extra Parameters column accuracy — Medium, deps: 2.5.1
- Task 2.5.4: Check Representative Models column accuracy — Medium, deps: 2.5.1

### Epic 3: Comparative Analysis Strengthening
- Task 3.1.1: Review Section 3.1 (Zero-Shot Extrapolation) — Medium, deps: 2.1, 2.2, 2.3, 2.4
- Task 3.1.2: Review Section 3.2 (Fine-Tuning Requirements) — Medium, deps: 2.1, 2.2, 2.3, 2.4
- Task 3.1.3: Review Section 3.3 (Recency Bias) — Medium, deps: 2.1, 2.2, 2.3, 2.4
- Task 3.1.4: Review Section 3.4 (Long-Range Retrieval) — Medium, deps: 2.1, 2.2, 2.3, 2.4
- Task 3.1.5: Review Section 3.5 (Computational Cost) — Medium, deps: 2.1, 2.2, 2.3, 2.4
- Task 3.2.1: Add nuance to NoPE discussion (decoder-only caveat) — Medium, deps: 2.4.1
- Task 3.2.2: Strengthen CABLE caveat (single paper, moderate scale) — Easy, deps: 2.4.2
- Task 3.2.3: Clarify NIAH test limitations for long-range retrieval — Easy, deps: 2.4.4
- Task 3.2.4: Ensure RoPE discussion covers both strengths and phase-shifting tradeoff — Medium, deps: 2.2.2
- Task 3.3.1: Verify Table 1 aligns with Section 3 narrative — Medium, deps: 2.5
- Task 3.3.2: Add methods mentioned in text but missing from Table 1 — Medium, deps: 2.5
- Task 3.3.3: Remove methods from Table 1 not discussed in narrative — Medium, deps: 2.5

### Epic 4: Terminology & Consistency
- Task 4.1.1: Ensure "positional encoding" is defined on first use — Easy, no deps
- Task 4.1.2: Ensure "length extrapolation" is defined on first use — Easy, no deps
- Task 4.1.3: Ensure "zero-shot extrapolation" is defined on first use — Easy, no deps
- Task 4.1.4: Ensure all key terms are defined on first use — Medium, no deps
- Task 4.2.1: Ensure consistent use of "Learned Absolute Positional Embedding" — Difficulty: Medium, Deps: none
- Task 4.2.2: Ensure consistent use of "length extrapolation" vs "generalization" — Difficulty: Medium, Deps: none
- Task 4.2.3: Ensure consistent capitalization and naming of methods — Difficulty: Easy, Deps: none
- Task 4.3.1: Verify Fig 1 taxonomy reference is correct and complete — Difficulty: Easy, Deps: none
- Task 4.3.2: Verify Table 1 reference is correct — Difficulty: Easy, Deps: none
- Task 4.3.3: Ensure all figures and tables are referenced in text before they appear — Difficulty: Easy, Deps: none
- Task 4.3.4: Check that figure and table numbering is sequential and correct — Difficulty: Easy, Deps: none

### Epic 5: Proofreading & Language Quality
- Task 5.1.1: Proofread Introduction section — Easy, deps: none
- Task 5.1.2: Proofread Literature Review section — Easy, deps: none
- Task 5.1.3: Proofread Methodology section — Easy, deps: none
- Task 5.1.4: Proofread Comparative Analysis section — Difficulty: Hard, deps: none
- Task 5.1.5: Proofread Conclusion section — Easy, deps: none
- Task 5.2.1: Check paragraph flow and transitions between sections — Difficulty: Medium, deps: 5.1
- Task 5.2.2: Verify abstract accurately summarizes scope and findings — Difficulty: Medium, deps: 5.1
- Task 5.2.3: Check that introduction builds logical narrative — Difficulty: Medium, deps: 5.1

### Epic 6: Compilation & Format Verification
- Task 6.1.1: Remove auxiliary files from previous build — Easy, deps: none
- Task 6.1.2: Perform clean pdflatex compilation — Easy, deps: 6.1.1
- Task 6.1.3: Verify no compilation errors or warnings — Easy, deps: 6.1.2
- Task 6.1.4: Verify PDF output renders correctly — Difficulty: Medium, deps: 6.1.2
- Task 6.2.1: Verify IEEEtran class options are correct — Easy, deps: none
- Task 6.2.2: Verify abstract format matches IEEE requirements — Easy, deps: none
- Task 6.2.3: Verify keywords are properly formatted — Easy, deps: none
- Task 6.2.4: Verify bibliography format matches IEEE style — Easy, deps: 1.4.4

### Epic 7: Documentation & Workspace Sync
- Task 7.1.1: Update comparison_matrix.md from paper findings — Difficulty: Medium, deps: 2.5.1
- Task 7.1.2: Update gap_analysis.md if new gaps identified — Difficulty: Medium, deps: 3.1
- Task 7.1.3: Update future_work.md with findings-driven directions — Difficulty: Medium, deps: 3.2
- Task 7.2.1: Update STATUS.md with review progress — Easy, deps: 3
- Task 7.2.2: Update TODO.md as individual items are completed — Easy, deps: 1-6
- Task 7.2.3: Record decisions in DECISIONS.md — Easy, deps: any decision made during review

## In Progress

## Review

## Blocked

## Done