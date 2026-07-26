# Dataset

## Training

| Property | Value |
|----------|-------|
| **Dataset** | WikiText-103 (Merity et al., 2017) |
| **Tokens** | 103M training, 218K validation, 246K test |
| **Domain** | Wikipedia articles |
| **Tokenizer** | GPT-2 (HuggingFace `GPT2Tokenizer`) |
| **Vocabulary** | 50,257 tokens |
| **Chunk size** | 512 tokens (training), variable for eval |
| **Format** | Cached `.npy` file of token IDs per split |
| **File** | `src/data/wikitext.py` — `WikiTextDataset` class |

### Loading

```python
dataset = WikiText103(
    root="data/wikitext-103",
    split="train",
    seq_len=512,
    tokenizer="gpt2"
)
# Returns: (batch_size, seq_len) tensor of token IDs
```

## Evaluation

| Dataset | Task | Usage |
|---------|------|-------|
| WikiText-103 test | Language modeling PPL | PPL @ 5 lengths |
| PG-19 test | Language modeling PPL (long-form) | PPL @ 5 lengths |
| Synthetic | Needle-in-a-Haystack | Retrieval accuracy |
| Synthetic | Multi-hop reasoning | Cross-doc accuracy |
| Synthetic | N-digit addition | Arithmetic acc |

## Synthetic Generators

All in `src/data/synthetic.py`:

### Needle-in-a-Haystack

```
Parameters: seq_len (L), needle_position (p), context_tokens
Output: [tokens[0..p], needle_tokens, tokens[p..L], query_tokens]
Label: needle_value
```

Needle format: `"The secret code is <VALUE>."`
Query: `"What is the secret code?"`
Context: random Wikipedia sentences or repeated filler tokens.

### Multi-hop

Two facts inserted at different depths:
- Fact 1: `"<NAME> lives in <CITY>."` at position p1
- Fact 2: `"<CITY> is in <COUNTRY>."` at position p2 > p1
- Query: `"Which country does <NAME> live in?"`

### Arithmetic

```
Input:  "12345 + 67890 = "
Output: "80235"
```
Digit counts: 1-10 digits. Zero-shot evaluation (no scratchpad).
