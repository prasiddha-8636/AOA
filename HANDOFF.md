# AOAPaper Handoff

## What this paper is
IEEE conference paper comparing three positional encoding paradigms (Learned Absolute, RoPE, ALiBi) under controlled conditions. Primary result: ALiBi extrapolates best (126.9→124.1, -2.2%) while Learned degrades 2.8x and RoPE 3.5x at 16x training length. PDF compiles at 6 pages, no overfull boxes.

## Git state
- Repo: `/home/admin/AOAPaper`
- 3 prior commits on master
- All current work is uncommitted (paper + results + scripts)

## File locations
| File | Purpose |
|------|---------|
| `paper/main.tex` | LaTeX source |
| `paper/IEEEtran.cls` | Template class |
| `paper/figures/` | 5 PDFs (needle heatmaps, PPL curves, extrapolation ratio, taxonomy, comparison radar) |
| `results/controlled_results.json` | Corrected ground truth numbers |
| `results/bootstrap_ci_results.json` | Per-method CIs + gap CIs |
| `results/alibi_retrain_verification.json` | ALiBi retraining trajectory |
| `results/PROVENANCE.md` | Full audit trail |
| `paper/main.pdf` | Compiled paper |

## Numbers (use these, not computed)
| Method | @512 | @1024 | @2048 | @4096 | @8192 |
|--------|------|-------|-------|-------|-------|
| Learned | 110.2 [105.7,114.7] | 149.9 [142.5,157.7] | 199.7 [185.6,212.3] | 253.6 [236.7,270.5] | 307.2 [283.2,330.7] |
| RoPE | 88.1 [84.4,92.0] | 114.9 [109.1,120.8] | 176.6 [164.3,187.5] | 245.7 [228.6,263.8] | 310.4 [285.7,334.2] |
| ALiBi | 126.9 [121.9,132.1] | 125.3 [118.9,131.9] | 124.7 [115.8,132.7] | 124.3 [115.5,133.5] | 124.1 [114.9,133.1] |

Gap (RoPE-ALiBi) at L=512: -38.8, CI [-40.4, -37.3] — excludes zero.
L=1024 marginal CIs overlap; resolved via paired gap CI [-40.4, -37.3].

## Dead-ends (don't repeat)
1. **ALiBi 93.7**: recording error. Retrained: trajectory 585→246→163→148→~127 at step 2000. Verified slopes (0.250, 0.0625, 0.015625, 0.003906). Done.
2. **Needle retrieval 0%**: capacity-bound at 17.7M. Non-diagnostic. Pre-trained models (124M-559M) retrieve fine within training windows.
3. **NoPE arm**: step 4000 checkpoint lost (val ppl 123.28). Needs fresh 10k-step GPU retraining (~50 min T4). No final result exists.
4. **Pythia/BLOOM @ L>=4096**: CUDA OOM on 16GB T4. No data.
5. **Across-run CIs**: only ALiBi has retraining verification. No budget for multi-seed bootstrap.
6. **Flash-attention / fp16 full attention**: both fail on T4. fp16 + chunked attention works (25% speedup at L=8192).

## Constraints (hard limits)
- **Single T4 GPU, 16GB VRAM**
- **No per-method hyperparameter tuning** — shared AdamW config. ALiBi in-window gap vs Press et al. is a known limitation
- **Fixed seed 42 per method**, single run (except ALiBi retraining verification)
- **10k training steps** per method, ~50 min T4 per run
- **Bootstrap**: 1000 resamples, 482→30 windows shrinking by length
- **Paper**: 6 pages IEEE format, pdflatex ×2, manual thebibliography (no .bib)
- **Three Colab accounts used** — GPU quota exhausted across sessions

## What's still open (from reviewer feedback, already applied)
- Paired-bootstrap clarification at L=1024: DONE
- Window stride=512 disclosed: DONE
- Slope values in reproducibility note: DONE
- Tokenization confound sentence: DONE
- Needle depth convention clarified: DONE
- Simple-style + stop-slop pass: DONE

## Style rules
- Short sentences, one idea each. Active voice. No adverbs, no hedging.
- No em-dashes. No "not X, it's Y" contrasts.
- "We" not "This paper" for ownership.
- Accuracy over sounding sophisticated.

## Build
```bash
cd /home/admin/AOAPaper/paper
pdflatex -interaction=nonstopmode main.tex  # run twice
```
No bibtex needed (manual bibliography). Figures are pre-generated PDFs in `figures/`.
