# Results

## Table 1: Perplexity Across Sequence Lengths

| Method | L=512 | L=1024 | L=2048 | L=4096 | L=8192 | PPL Ratio (8192/512) |
|--------|-------|--------|--------|--------|--------|---------------------|
| Learned | | | | | | |
| Sinusoidal | | | | | | |
| RoPE | | | | | | |
| ALiBi | | | | | | |
| NoPE | | | | | | |
| KERPLE | | | | | | |
| CABLE | | | | | | |
| PI (ft) | | | | | | |

## Table 2: Needle-in-a-Haystack Accuracy

| Method | 25% depth | 50% depth | 75% depth | 90% depth |
|--------|-----------|-----------|-----------|-----------|
| Learned | | | | |
| Sinusoidal | | | | |
| RoPE | | | | |
| ALiBi | | | | |
| NoPE | | | | |
| KERPLE | | | | |
| CABLE | | | | |
| PI (ft) | | | | |

## Table 3: Recency Bias and Compute Profile

| Method | Recency Bias Score | FLOPs @ L=512 | Params (PE) | Throughput (tok/s) |
|--------|-------------------|---------------|-------------|-------------------|
| Learned | | | | |
| Sinusoidal | | | | |
| RoPE | | | | |
| ALiBi | | | | |
| NoPE | | | | |
| KERPLE | | | | |
| CABLE | | | | |
| PI (ft) | | | | |

## Table 4: Fine-Tuning Cost (Position Interpolation)

| Target Length | Steps to Recover | Final PPL | Compute (GPU-hours) |
|--------------|-----------------|-----------|-------------------|
| L=1024 | | | |
| L=2048 | | | |
| L=4096 | | | |
| L=8192 | | | |
