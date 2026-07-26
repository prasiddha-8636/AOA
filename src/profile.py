import torch
from src.config import ModelConfig
from src.model.transformer import Transformer
from src.model.pe import get_positional_encoding


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}


def count_pe_parameters(model):
    pe_params = sum(p.numel() for p in model.pe_module.parameters())
    return pe_params


@torch.no_grad()
def estimate_flops(model, seq_len, vocab_size=50257):
    """Estimate FLOPs for one forward pass (multiply-adds only)."""
    config = model.config
    B = 1

    # Token embedding: B * L * d_model
    flops_embed = B * seq_len * config.d_model

    # Per transformer block
    flops_per_block = 0

    # QKV projections: B * L * d_model * (3 * d_model)
    flops_qkv = B * seq_len * config.d_model * (3 * config.d_model)
    flops_per_block += flops_qkv

    # Attention: Q @ K^T = B * n_heads * L * d_head * L
    flops_attn = B * config.n_heads * seq_len * config.d_head * seq_len
    flops_per_block += flops_attn

    # Softmax: B * n_heads * L * L (exp + sum)
    flops_softmax = B * config.n_heads * seq_len * seq_len * 2
    flops_per_block += flops_softmax

    # Attention @ V: B * n_heads * L * L * d_head
    flops_attv = B * config.n_heads * seq_len * seq_len * config.d_head
    flops_per_block += flops_attv

    # Output projection: B * L * d_model * d_model
    flops_out = B * seq_len * config.d_model * config.d_model
    flops_per_block += flops_out

    # MLP: B * L * d_model * d_ff + B * L * d_ff * d_model
    flops_mlp = 2 * B * seq_len * config.d_model * config.d_ff
    flops_per_block += flops_mlp

    # LayerNorm: B * L * d_model * 2
    flops_norm = B * seq_len * config.d_model * 2
    flops_per_block += flops_norm * 2  # two layer norms per block

    # LM head: B * L * d_model * vocab_size
    flops_head = B * seq_len * config.d_model * vocab_size

    total_flops = flops_embed + config.n_layers * flops_per_block + flops_head
    return total_flops


def measure_throughput(model, seq_len, batch_size, device, n_warmup=10, n_iters=50):
    """Measure tokens per second."""
    import time

    model.eval()
    dummy = torch.randint(0, 100, (batch_size, seq_len), device=device)

    # Warmup
    for _ in range(n_warmup):
        _ = model(dummy)

    # Benchmark
    torch.cuda.synchronize(device)
    start = time.perf_counter()
    for _ in range(n_iters):
        _ = model(dummy)
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start

    tokens_per_sec = batch_size * seq_len * n_iters / elapsed
    return tokens_per_sec


def profile_all_methods(device="cuda"):
    from src.config import ALL_METHODS  # now defined in config.py

    results = {}
    for method in ALL_METHODS:
        print(f"Profiling: {method}")
        model_config = ModelConfig()
        pe_module = get_positional_encoding(method, model_config.d_model, model_config.n_heads, model_config.max_seq_len)
        model = Transformer(model_config, pe_module).to(device)

        params = count_parameters(model)
        pe_params = count_pe_parameters(model)
        flops = estimate_flops(model, model_config.max_seq_len)
        throughput = measure_throughput(model, model_config.max_seq_len, 64, device)

        results[method] = {
            "total_params": params["total"],
            "pe_params": pe_params,
            "flops": flops,
            "throughput_tok_s": throughput,
        }
        print(f"  Params: {params['total']:,}, PE params: {pe_params}, FLOPs: {flops:,}, Throughput: {throughput:.0f} tok/s")

    return results


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    profile_all_methods(device)
