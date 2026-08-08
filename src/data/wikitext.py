import os
import torch
from torch.utils.data import Dataset

import numpy as np

if not hasattr(np, "long"):
    np.long = np.int64
if not hasattr(np, "ulong"):
    np.ulong = np.uint64

_WIKITEXT_REPO = "Salesforce/wikitext"
_WIKITEXT_CONFIG = "wikitext-103-raw-v1"
_TOKEN_CACHE_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "cache"
)


class WikiTextDataset(Dataset):
    """Tokenized WikiText-103. Tokenized tensors are cached to disk so repeated
    loader construction (train + validation, or a Colab restart after a
    disconnect) does not re-download / re-encode the corpus."""

    def __init__(self, split, seq_len, tokenizer_name="gpt2", max_tokens=None):
        from transformers import GPT2Tokenizer

        self.tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.seq_len = seq_len
        self.tokens = _get_tokens(split, self.tokenizer, max_tokens)

    def __len__(self):
        return max((len(self.tokens) - 1) // self.seq_len, 0)

    def __getitem__(self, idx):
        start = idx * self.seq_len
        end = start + self.seq_len + 1
        chunk = self.tokens[start:end]
        if len(chunk) < self.seq_len + 1:
            chunk = torch.cat(
                [
                    chunk,
                    self.tokenizer.pad_token_id
                    * torch.ones(self.seq_len + 1 - len(chunk), dtype=torch.long),
                ]
            )
        return chunk[:-1], chunk[1:]


def _get_tokens(split, tokenizer, max_tokens=None):
    os.makedirs(_TOKEN_CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(_TOKEN_CACHE_DIR, f"wikitext103_{split}.pt")
    if max_tokens is not None:
        cache_path = os.path.join(
            _TOKEN_CACHE_DIR, f"wikitext103_{split}_{max_tokens}.pt"
        )
    if os.path.exists(cache_path):
        return torch.load(cache_path, weights_only=True)
    from datasets import load_dataset

    ds = load_dataset(_WIKITEXT_REPO, _WIKITEXT_CONFIG, split=split)
    text = "\n\n".join(ds["text"])
    # Cap raw text before tokenizing: full WikiText-103 train (~500MB / ~103M
    # tokens) can exceed memory on constrained hardware. Tokenize only enough
    # text to cover max_tokens (chars roughly ~4x token count for GPT-2).
    if max_tokens is not None:
        max_chars = max_tokens * 4 + 1_000_000
        if len(text) > max_chars:
            text = text[:max_chars]
    tokens = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    if max_tokens is not None and len(tokens) > max_tokens:
        tokens = tokens[:max_tokens]
    torch.save(tokens, cache_path)
    return tokens


def get_wikitext_dataloader(split, seq_len, batch_size, num_workers=0, max_tokens=None):
    dataset = WikiTextDataset(split, seq_len, max_tokens=max_tokens)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        pin_memory=True,
    )
    return loader
