import torch
from torch.utils.data import Dataset


class PG19Dataset(Dataset):
    def __init__(self, seq_len):
        from transformers import GPT2Tokenizer
        from datasets import load_dataset

        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.seq_len = seq_len

        dataset = load_dataset("pg19", split="test")
        self.texts = dataset["text"]

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        tokens = self.tokenizer.encode(self.texts[idx])
        tokens = torch.tensor(tokens, dtype=torch.long)
        if len(tokens) < self.seq_len + 1:
            tokens = torch.cat([tokens, torch.zeros(self.seq_len + 1 - len(tokens), dtype=torch.long)])
        start = torch.randint(0, len(tokens) - self.seq_len - 1, (1,)).item()
        chunk = tokens[start:start + self.seq_len + 1]
        return chunk[:-1], chunk[1:]
