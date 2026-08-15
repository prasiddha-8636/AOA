import os
import time
import math
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from src.config import ExperimentConfig, ModelConfig, TrainingConfig
from src.model.transformer import Transformer
from src.model.pe import get_positional_encoding
from src.data.wikitext import get_wikitext_dataloader
from src.evaluate import evaluate_ppl
from src.colab_io import persist_path


def cosine_schedule(step, warmup_steps, total_steps):
    if step < warmup_steps:
        return step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def train(
    method,
    model_config=None,
    train_config=None,
    resume_from=None,
    out_dir="checkpoints",
):
    if model_config is None:
        model_config = ModelConfig()
    if train_config is None:
        train_config = TrainingConfig()
    out_dir = persist_path("checkpoints", out_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pe_module = get_positional_encoding(
        method, model_config.d_model, model_config.n_heads, model_config.max_seq_len
    )
    model = Transformer(model_config, pe_module).to(device)

    train_loader = get_wikitext_dataloader(
        "train",
        model_config.max_seq_len,
        train_config.batch_size,
        num_workers=0,
        max_tokens=train_config.max_train_tokens,
    )
    val_loader = get_wikitext_dataloader(
        "validation",
        model_config.max_seq_len,
        min(4, train_config.batch_size),
        num_workers=0,
        max_tokens=train_config.max_val_tokens,
    )

    optimizer = AdamW(
        model.parameters(),
        lr=train_config.learning_rate,
        betas=(train_config.beta1, train_config.beta2),
        eps=train_config.epsilon,
        weight_decay=train_config.weight_decay,
    )

    step = 0
    best_val_ppl = float("inf")
    os.makedirs(out_dir, exist_ok=True)
    resume_path = resume_from or os.path.join(out_dir, method, "resume.pt")
    if os.path.exists(resume_path):
        ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        step = ckpt["step"] + 1
        best_val_ppl = ckpt.get("best_val_ppl", float("inf"))
        print(f"[{method}] Resuming from step {step} (best_val_ppl={best_val_ppl:.4f})")

    scheduler = LambdaLR(
        optimizer,
        lambda s: cosine_schedule(
            s, train_config.warmup_steps, train_config.total_steps
        ),
    )
    for _ in range(step):
        scheduler.step()

    scaler = torch.cuda.amp.GradScaler(
        enabled=(train_config.mixed_precision == "fp16" and device.type == "cuda")
    )

    model.train()
    train_iter = iter(train_loader)
    best_path = os.path.join(out_dir, method, "best.pt")

    accum_steps = max(1, train_config.batch_size // max(1, train_config.micro_batch))
    while step < train_config.total_steps:
        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, y = next(train_iter)

        x, y = x.to(device), y.to(device)
        x, y = (
            x[: train_config.micro_batch * accum_steps],
            y[: train_config.micro_batch * accum_steps],
        )
        for k in range(accum_steps):
            xb = x[k * train_config.micro_batch : (k + 1) * train_config.micro_batch]
            yb = y[k * train_config.micro_batch : (k + 1) * train_config.micro_batch]
            if xb.shape[0] == 0:
                continue  # partial last batch in the dataloader
            with torch.cuda.amp.autocast(
                enabled=(
                    train_config.mixed_precision == "fp16" and device.type == "cuda"
                )
            ):
                logits = model(xb)
                loss = nn.CrossEntropyLoss()(
                    logits.view(-1, model_config.vocab_size), yb.view(-1)
                )
            scaler.scale(loss / accum_steps).backward()
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

            print(
                f"[{method}] Step {step}: loss={loss.item():.4f}, "
                f"val_ppl={val_ppl:.4f}, lr={scheduler.get_last_lr()[0]:.2e}",
                flush=True,
            )

            if val_ppl < best_val_ppl:
                best_val_ppl = val_ppl
                os.makedirs(os.path.dirname(best_path), exist_ok=True)
                torch.save(model.state_dict(), best_path)

            # Durable resume point (Colab-safe)
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "step": step,
                    "best_val_ppl": best_val_ppl,
                },
                resume_path,
            )

        step += 1

    # Save final checkpoint
    os.makedirs(os.path.dirname(best_path), exist_ok=True)
    torch.save(model.state_dict(), os.path.join(out_dir, method, "final.pt"))
    print(f"[{method}] Training complete. Best val_ppl: {best_val_ppl:.4f}")
    return model


def train_pi(pretrained_path, train_len=512, target_len=2048):
    """Fine-tune RoPE checkpoint with Position Interpolation."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_config = ModelConfig()
    pe_module = get_positional_encoding(
        "position_interpolation",
        model_config.d_model,
        model_config.n_heads,
        model_config.max_seq_len,
    )
    pe_module.set_scale(train_len, target_len)

    model = Transformer(model_config, pe_module).to(device)
    model.load_state_dict(torch.load(pretrained_path, map_location=device))

    train_loader = get_wikitext_dataloader("train", target_len, 32, num_workers=0)
    val_loader = get_wikitext_dataloader("validation", target_len, 32, num_workers=0)

    optimizer = AdamW(model.parameters(), lr=1e-5, weight_decay=0.1)
    scaler = torch.cuda.amp.GradScaler()

    model.train()
    n_steps = 5000
    for step in range(n_steps):
        x, y = next(iter(train_loader))
        x, y = x.to(device), y.to(device)

        with torch.cuda.amp.autocast():
            logits = model(x)
            loss = nn.CrossEntropyLoss()(
                logits.view(-1, model_config.vocab_size), y.view(-1)
            )

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        if step % 500 == 0:
            print(f"[PI] Step {step}: loss={loss.item():.4f}")

    torch.save(model.state_dict(), "checkpoints/position_interpolation/final.pt")
    print("[PI] Fine-tuning complete.")
