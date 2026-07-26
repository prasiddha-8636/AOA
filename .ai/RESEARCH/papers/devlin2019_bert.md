# Devlin et al. (2019) — "BERT: Pre-training of Deep Bidirectional Transformers"

## Bibliography Key
devlin2019

## Venue
NAACL-HLT 2019, pp. 4171–4186

## Methodology
- Introduced BERT: Bidirectional Encoder Representations from Transformers
- Uses learned absolute positional embeddings
- Pre-trained with Masked Language Modeling (MLM) and Next Sentence Prediction (NSP)
- Transformer encoder-only architecture

## Strengths
- Bidirectional pre-training enables richer contextual representations than unidirectional models
- Learned absolute PE adapts to the task during pretraining
- State-of-the-art on eleven NLP tasks at the time of publication
- Conceptually simple and empirically powerful

## Weaknesses
- Learned absolute PE cannot extrapolate beyond the maximum position seen during training (typically 512 tokens)
- No length extrapolation capability
- BERT is an encoder-only model; not directly comparable to decoder-only models regarding NoPE findings

## Zero-Shot Capability
- Poor: learned embeddings have no representation for positions beyond the training maximum

## Fine-Tuning Requirements
- Fine-tuning is standard for BERT adaptation to downstream tasks
- Position embeddings cannot be extended without retraining or adding new embeddings

## Compute Cost
- Moderate: learned PE adds parameters proportional to max context length, but only table lookup at runtime

## Implementation Complexity
- Low: learned embedding table lookup

## Long-Context Retrieval
- Not evaluated for long-context retrieval due to fixed context window

## Recency Bias
- None: learned absolute PE has no explicit recency mechanism

## Limitations
- Fixed context length (512 tokens) prevents processing of longer documents
- Learned embeddings are out-of-distribution for unseen positions

## Research Gaps
- BERT's fixed context window motivated subsequent work on length generalization and extrapolation methods