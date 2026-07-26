# Zhao et al. (2024) — "Length Extrapolation of Transformers: A Survey from the Perspective of Positional Encoding"

## Bibliography Key
zhao2024survey

## Venue
Findings of ACL: EMNLP 2024, pp. 9959–9977

## Methodology
- Comprehensive survey of length extrapolation methods from the perspective of positional encoding
- Unified notation across all reviewed methods
- Categorizes PEs into absolute, relative, rotary, and bias-based paradigms
- Covers PE-based extrapolation methods (position interpolation, randomized methods)
- Highlights challenges and future directions

## Strengths
- Most comprehensive and up-to-date survey on length extrapolation (as of 2024)
- Unified notation enables direct comparison across methods
- Covers both PE-based methods and their combinations
- Well-structured taxonomy of methods
- Identifies challenges and future directions systematically
- Widely cited (foundational reference for this research area)

## Weaknesses
- Survey papers cannot provide new empirical results
- The survey's analysis depends on the quality and selection of the primary sources
- Some methods may be missing or under-represented
- The "perspective of positional encoding" framing may miss non-PE approaches (e.g., architectural changes)

## Zero-Shot Capability
-综述 paper; does not propose a specific method with its own zero-shot capability

## Fine-Tuning Requirements
-综述 paper; does not propose a specific method with its own fine-tuning requirements

## Compute Cost
-综述 paper; does not propose a specific method with its own compute cost

## Implementation Complexity
-综述 paper; does not propose a specific method with its own implementation complexity

## Long-Context Retrieval
-综述 paper; discusses retrieval challenges across methods

## Recency Bias
-综述 paper; surveys recency bias properties of different methods

## Limitations
- Survey cannot provide new experimental validation
- Depends on the selected primary sources for accuracy
- The framing around PE may miss architectural or training-based approaches

## Research Gaps
- Identified by the survey itself: unified benchmarks, non-causal attention models, content-conditioned methods