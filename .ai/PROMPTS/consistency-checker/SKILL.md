---
name: consistency-checker
description: Use this skill when reviewing or editing any part of Pika's thesis/report, paper, slides, or code together — especially before a defense or submission. Checks that numbers, thresholds, and terminology match across all artifacts (code, report, slides, results). Trigger on requests like "check consistency," "does this match," or whenever editing one artifact that has counterparts in others (e.g. changing a threshold in code should trigger a check against the report).
---

# Consistency Checker

Purpose: catch mismatches between what the code does, what the report claims, and what the slides show — before Pika's committee does.

## What to check
1. **Numeric values** — thresholds, accuracy figures, error margins, hyperparameters. Same value must appear everywhere it's referenced (code, report, slides, results tables).
2. **Terminology** — same concept must use the same name everywhere. No silent renaming ("near-field" in code vs "near-tier" in report).
3. **Claims vs implementation** — if the report says a filter/step exists, confirm the code actually has it, with the same logic.
4. **Figures/tables vs text** — a figure caption or table value must match what the surrounding text says about it.
5. **Threshold rationale** — if the report justifies a threshold, check the justification matches why that threshold is actually set in code (not a plausible-sounding reason invented after the fact).

## Process
- When asked to check, actually read the relevant files (code, report, slides) rather than assuming they match.
- List mismatches found, each as: where A says X, where B says Y.
- Don't just flag — give the corrected value/wording so all locations agree, and say which one is authoritative (usually: code is ground truth for implementation, report should follow it).
- No mismatch found in a section = don't mention that section.

## Output format
- One line per inconsistency: location A vs location B, then the fix.
- If everything matches, say so in one line.
