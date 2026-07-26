import torch
import random


PROMPT_TEMPLATES = {
    "needle": (
        "The following is a long text about various topics. "
        "{context_text} "
        "{needle_sentence} "
        "{context_text_after} "
        "Question: What is the secret code? Answer: The secret code is"
    ),
    "multi_hop": (
        "Here is some information. "
        "{fact1} "
        "{fact2} "
        "Question: {question} Answer:"
    ),
    "arithmetic": (
        "{num1} + {num2} ="
    ),
}


class SyntheticNeedleHaystack:
    def __init__(self, tokenizer, vocab_size=50257):
        self.tokenizer = tokenizer
        self.vocab_size = vocab_size
        self.filler_tokens = list(range(100, 1000))

    def generate(self, seq_len, needle_depth=0.5):
        needle_value = f"{random.randint(10000, 99999)}"
        needle_sentence = f"The secret code is {needle_value}."

        needle_pos = int(seq_len * needle_depth)
        context_after_len = seq_len - needle_pos - len(self.tokenizer.encode(needle_sentence))

        filler = " ".join([f"token_{random.randint(0, 1000)}" for _ in range(seq_len // 5)])
        context_text = filler * (needle_pos // len(filler.split()) + 1)
        context_text = " ".join(context_text.split()[:needle_pos])

        context_text_after = filler[:context_after_len] if context_after_len > 0 else ""

        prompt = PROMPT_TEMPLATES["needle"].format(
            context_text=context_text[:needle_pos],
            needle_sentence=needle_sentence,
            context_text_after=context_text_after,
        )

        tokens = self.tokenizer.encode(prompt)
        input_ids = torch.tensor(tokens[:seq_len], dtype=torch.long)

        # Answer: the needle value as token IDs (digits)
        answer_tokens = self.tokenizer.encode(f" {needle_value}")
        answer = torch.tensor(answer_tokens, dtype=torch.long)

        return input_ids, answer, needle_value


class SyntheticMultiHop:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.names = ["Alice", "Bob", "Charlie", "Diana", "Eve", "Frank"]
        self.cities = ["London", "Tokyo", "Paris", "Berlin", "Sydney", "Seoul"]
        self.countries = ["England", "Japan", "France", "Germany", "Australia", "South Korea"]

    def generate(self, seq_len):
        name = random.choice(self.names)
        city = random.choice(self.cities)
        country = self.countries[self.cities.index(city)]

        fact1 = f"{name} lives in {city}."
        fact2 = f"{city} is in {country}."
        question = f"Which country does {name} live in?"

        prompt = PROMPT_TEMPLATES["multi_hop"].format(
            fact1=fact1,
            fact2=fact2,
            question=question,
        )

        tokens = self.tokenizer.encode(prompt)
        input_ids = torch.tensor(tokens[:seq_len], dtype=torch.long)
        answer_ids = self.tokenizer.encode(f" {country}")
        answer = torch.tensor(answer_ids, dtype=torch.long)

        return input_ids, answer, country


class SyntheticArithmetic:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def generate(self, max_digits=5):
        d1 = random.randint(1, max_digits)
        d2 = random.randint(1, max_digits)
        num1 = random.randint(10 ** (d1 - 1), 10 ** d1 - 1)
        num2 = random.randint(10 ** (d2 - 1), 10 ** d2 - 1)
        result = num1 + num2

        prompt = PROMPT_TEMPLATES["arithmetic"].format(num1=num1, num2=num2)
        tokens = self.tokenizer.encode(prompt)
        input_ids = torch.tensor(tokens, dtype=torch.long)
        answer_ids = self.tokenizer.encode(f" {result}")
        answer = torch.tensor(answer_ids, dtype=torch.long)

        return input_ids, answer, result
