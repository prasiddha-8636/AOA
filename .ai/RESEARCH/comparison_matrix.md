# Literature Comparison

## Verified Data from Primary Source Papers

| Method | Type | Zero-Shot | Fine-Tuning | Recency Bias | Compute | Params | Long-Range Retrieval | Source |
|--------|------|-----------|-------------|--------------|---------|--------|---------------------|--------|
| Sinusoidal (Vaswani 2017) | Absolute | Low | N/A | None | Very Low | No | Unknown | vaswani2017 |
| Learned Absolute (Devlin 2019) | Absolute | Poor | Yes | None | Low | Yes | Unknown | devlin2019 |
| ALiBi (Press 2022) | Linear Bias | High | No | Strong | Very Low | No | Poor | press2022 |
| T5 Relative (Raffel 2020) | Relative (Bucketed) | Medium | Yes | Mild | Moderate | Yes | Moderate | raffel2020 |
| Shaw Relative (Shaw 2018) | Relative | Medium | Yes | None | High | Yes | Limited | shaw2018 |
| Transformer-XL (Dai 2019) | Segment Recurrence | Medium | Yes | None | Moderate | Yes | Good | dai2019 |
| RoPE (Su 2024) | Rotary | High | No (base) | Mild | Very Low | No | Moderate | su2024roformer |
| ALiBi Recency (Press 2022) | Bias | Strong recency bias | N/A | Strongest | Very Low | No | Poor | press2022 |
| RoPE Recency (Su 2024) | Rotary | Mild, unpredictable | N/A | Mildest | Very Low | No | Moderate | su2024roformer |
| No Positional (Kazemnejad 2023) | None | Medium-High | No | None | Very Low | No | Variable | kazemnejad2023 |
| Position Interpolation (Chen 2023) | RoPE Extension | Very High | Yes | Inherits RoPE | Low | No | Good | chen2023 |
| YaRN (Peng 2023) | RoPE Extension | Very High | Yes (reduced) | Inherits RoPE | Low | No | Good | peng2023yarn |
| Frequency Rescaling (Liu 2024) | RoPE Extension | Very High | No (small base) | Inherits RoPE | Moderate | No | Good | liu2024scaling |
| KERPLE (Chi 2022) | Kernelized Bias | High | No (log variant) | Milder than ALiBi | Moderate | Yes (24 params) | Good | chi2022 |
| Randomized PE (Ruoss 2023) | Absolute (Randomized) | High | Yes (training) | Inherits base | Low | No | Variable | ruoss2023 |
| CABLE (Veisi 2025) | Context-Aware Bias | High | No | Softer than ALiBi | Low-Moderate | Yes (lightweight) | Improved | veisi2025cable |
| LongRoPE2 (Shang 2025) | RoPE Extension | Very High | Yes | Inherits RoPE | Low | No | Good | shang2025longrope2 |
| BiPE (He 2024) | Bilevel | Very High | Yes (from base) | Depends on inter-segment | Low | No | Improved | he2024bipe |
| Survey (Zhao 2024) | Review | N/A | N/A | N/A | N/A | N/A | N/A | zhao2024survey |

## Key Findings from Paper-Level Analysis

### Learned Absolute PE (BERT/GPT-2)
- Extrapolation: Poor — no representation for positions beyond training maximum
- Fine-tuning: Required for any length extension
- Source: Devlin et al. 2019, Radford et al. 2019

### Sinusoidal PE (Original Transformer)
- Extrapolation: Theoretically defined for any position, but models fail to learn the rotational structure for extrapolation
- Fine-tuning: Not designed for extrapolation
- Source: Vaswani et al. 2017

### ALiBi
- Extrapolation: Strongest zero-shot among methods compared
- Recency bias: Strongest — linear penalty proportional to distance
- Long-range retrieval: Worst — distances are suppressed too aggressively
- Compute: Very low — static bias addition, no learned parameters
- Source: Press et al. 2022 (ICLR 2022)

### RoPE (LLaMA-style)
- Extrapolation: Strong but limited by high-frequency phase-shifting at extreme lengths
- Recency bias: Mild and less predictable (emergent, not explicit)
- Long-range retrieval: Moderate — preserves signal better than bias-based methods
- Fine-tuning: Base RoPE works without fine-tuning; extensions (PI, YaRN, LongRoPE2) require it
- Source: Su et al. 2024 (Neurocomputing), arXiv:2104.09864

### KERPLE
- Extrapolation: High — logarithmic kernel provides slower decay than ALiBi
- Recency bias: Milder than ALiBi due to log kernel's flat regions
- Long-range retrieval: Better than ALiBi — log kernel preserves attention to distant tokens
- Compute: Moderate — kernel computation adds some overhead
- Parameters: Only 24 additional learnable parameters (2 per head)
- Source: Chi et al. 2022 (NeurIPS 2022), arXiv:2205.09921

### Position Interpolation (PI)
- Extrapolation: Very High — compresses positions to fit training range
- Fine-tuning: Required (brief, ~1000 steps)
- Source: Chen et al. 2023, arXiv:2306.15595

### YaRN
- Extrapolation: Very High — frequency-selective interpolation + temperature adjustment
- Fine-tuning: Required but 10x fewer tokens and 2.5x fewer steps than PI
- Source: Peng et al. 2023, arXiv:2309.00071

### NoPE (Kazemnejad et al. 2023)
- Extrapolation: Surprising — outperforms ALiBi, RoPE, and learned PE on length generalization tasks
- Caveat: Only for decoder-only architectures; relies on causal mask
- Source: Kazemnejad et al. 2023 (NeurIPS 2023), arXiv:2305.19466

### CABLE (Veisi et al. 2025)
- Extrapolation: High — context-aware biases dynamically adjust
- Recency bias: Softer than ALiBi — content-dependent bias reduces hard distance penalty
- Long-range retrieval: Improved — content-conditioned bias less likely to discard relevant distant tokens
- Single paper at moderate scale; read as promising direction
- Source: Veisi et al. 2025, arXiv:2503.08067

### LongRoPE2 (Shang et al. 2025)
- Extrapolation: Very High — 128K effective context with >98.5% short-context preservation
- Data-efficient: 10B tokens (80x fewer than Meta's approach)
- One-time evolutionary search cost during setup; no per-token overhead
- Source: Shang et al. 2025, arXiv:2502.20082

### BiPE (He et al. 2024)
- Extrapolation: Very High — intra-segment absolute + inter-segment relative encoding
- Arithmetic: 97% accuracy vs. <70% for baselines at comparable parameters (hidden=48)
- PG-19 LM: BiPE-ALiBi perplexity 25.2 vs. RoPE 158, ALiBi 28.6
- SCROLLS: +3.98 improvement over RoPE
- Source: He et al. 2024 (ICML 2024), arXiv:2401.16421