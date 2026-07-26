import torch
from torch.utils.data import Dataset


class WikiTextDataset(Dataset):
    def __init__(self, split, seq_len, tokenizer_name="gpt2"):
        from transformers import GPT2Tokenizer
        from datasets import load_dataset

        self.tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.seq_len = seq_len

        dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split=split)
        text = "\n\n".join(dataset["text"])
        tokens = self.tokenizer.encode(text)
        self.tokens = torch.tensor(tokens, dtype=torch.long)

    def __len__(self):
        return (len(self.tokens) - 1) // self.seq_len

    def __getitem__(self, idx):
        start = idx * self.seq_len
        end = start + self.seq_len + 1
        chunk = self.tokens[start:end]
        if len(chunk) < self.seq_len + 1:
            chunk = torch.cat([chunk, self.tokenizer.pad_token_id * torch.ones(self.seq_len + 1 - len(chunk), dtype=torch.long)])
        return chunk[:-1], chunk[1:]


def get_wikitext_dataloader(split, seq_len, batch_size, num_workers=2):
    dataset = WikiTextDataset(split, seq_len)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        pin_memory=True,
    )
    return loader
