# Experiments

## Setup

```bash
pip install torch transformers datasets tqdm wandb
```

## Train All Methods

```bash
bash experiments/run_all.sh [gpu_id]
```

## Train Single Method

```bash
python -m src.train --method rope
```

## Fine-tune Position Interpolation

```bash
python -m src.train_pi \
    --pretrained checkpoints/rope/final.pt \
    --target_len 2048
```

## Evaluate

```bash
python -m src.evaluate --method rope --checkpoint checkpoints/rope/best.pt
```

## Profile

```bash
python -m src.profile
```

## Output

All results stored in:
- `checkpoints/{method}/best.pt` — best checkpoint
- `checkpoints/{method}/final.pt` — final checkpoint
- Results printed to stdout and logged by wandb
