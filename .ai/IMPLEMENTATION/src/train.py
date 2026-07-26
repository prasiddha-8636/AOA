import os
import time
import math
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
import wandb

from src.config import ExperimentConfig, ModelConfig, TrainingConfig
from src.model.transformer import Transformer
from src.model.pe import get_positional_encoding
from src.data.wikitext import get_wikitext_dataloader
from src.evaluate import evaluate_ppl


def cosine_schedule(step, warmup_steps, total_steps):
    if step < warmup_steps:
        return step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def train(config, method, resume_from=None):
    model_config = ModelConfig()
    train_config = TrainingConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pe_module = get_positional_encoding(
        method, model_config.d_model, model_config.n_heads, model_config.max_seq_len
    )
    model = Transformer(model_config, pe_module).to(device)

    if resume_from:
        model.load_state_dict(torch.load(resume_from, map_location=device))

    train_loader = get_wikitext_dataloader(
        "train", model_config.max_seq_len, train_config.batch_size
    )
    val_loader = get_wikitext_dataloader(
        "validation", model_config.max_seq_len, train_config.batch_size
    )

    optimizer = AdamW(
        model.parameters(),
        lr=train_config.learning_rate,
        betas=(train_config.beta1, train_config.beta2),
        eps=train_config.epsilon,
        weight_decay=train_config.weight_decay,
    )

    scheduler = LambdaLR(
        optimizer,
        lambda step: cosine_schedule(step, train_config.warmup_steps, train_config.total_steps),
    )

    scaler = torch.cuda.amp.GradScaler(enabled=(train_config.mixed_precision == "fp16"))

    model.train()
    step = 0
    best_val_ppl = float("inf")
    train_iter = iter(train_loader)

    while step < train_config.total_steps:
        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, y = next(train_iter)

        x, y = x.to(device), y.to(device)

        with torch.cuda.amp.autocast(enabled=(train_config.mixed_precision == "fp16")):
            logits = model(x)
            loss = nn.CrossEntropyLoss()(logits.view(-1, model_config.vocab_size), y.view(-1))

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), train_config.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        optimizer.zero_grad()

        if step % train_config.eval_interval == 0:
            model.eval()
            val_ppl = evaluate_ppl(model, val_loader, model_config.max_seq_len, device)
            model.train()

            print(f"[{method}] Step {step}: loss={loss.item():.4f}, val_ppl={val_ppl:.4f}, lr={scheduler.get_last_lr()[0]:.2e}")

            if val_ppl < best_val_ppl:
                best_val_ppl = val_ppl
                os.makedirs(f"checkpoints/{method}", exist_ok=True)
                torch.save(model.state_dict(), f"checkpoints/{method}/best.pt")

        step += 1

    # Save final checkpoint
    os.makedirs(f"checkpoints/{method}", exist_ok=True)
    torch.save(model.state_dict(), f"checkpoints/{method}/final.pt")
    print(f"[{method}] Training complete. Best val_ppl: {best_val_ppl:.4f}")
    return model


def train_pi(pretrained_path, train_len=512, target_len=2048):
    """Fine-tune RoPE checkpoint with Position Interpolation."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_config = ModelConfig()
    pe_module = get_positional_encoding("position_interpolation", model_config.d_model, model_config.n_heads, model_config.max_seq_len)
    pe_module.set_scale(train_len, target_len)

    model = Transformer(model_config, pe_module).to(device)
    model.load_state_dict(torch.load(pretrained_path, map_location=device))

    train_loader = get_wikitext_dataloader("train", target_len, 32)
    val_loader = get_wikitext_dataloader("validation", target_len, 32)

    optimizer = AdamW(model.parameters(), lr=1e-5, weight_decay=0.1)
    scaler = torch.cuda.amp.GradScaler()

    model.train()
    n_steps = 5000
    for step in range(n_steps):
        x, y = next(iter(train_loader))
        x, y = x.to(device), y.to(device)

        with torch.cuda.amp.autocast():
            logits = model(x)
            loss = nn.CrossEntropyLoss()(logits.view(-1, model_config.vocab_size), y.view(-1))

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        if step % 500 == 0:
            print(f"[PI] Step {step}: loss={loss.item():.4f}")

    torch.save(model.state_dict(), f"checkpoints/position_interpolation/L{target_len}.pt")
    return model


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, required=True)
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    train(ExperimentConfig(), args.method, args.resume)
