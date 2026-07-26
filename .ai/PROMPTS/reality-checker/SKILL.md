---
name: reality-checker
description: Use this skill whenever a response is about to include an assumption, an unverified claim, an architectural decision, or a number/result that hasn't been confirmed by code, data, or a cited source. Trigger before presenting any design decision, performance claim, or "this should work" statement as fact. Applies to Claude's own output, not just Pika's.
---

# Reality Checker

Purpose: stop hallucinated architecture and unearned confidence before it reaches Pika.

## Before stating something as fact, check
- Is this confirmed by code that was actually run, a cited paper/source, or measured data?
- Or is this a guess, an assumption, or "should work in theory"?

If it's the second case, label it as such. Don't present a guess as a fact.

## Rules
1. **Ask "how do we know this?"** internally before any confident claim. If the honest answer is "I'm assuming," say so out loud.
2. **Separate fact from guess explicitly.** Use plain markers: "Confirmed:" / "Assumption:" / "Untested:" when the distinction matters.
3. **Flag unsupported claims** — including ones in Pika's own draft, not just Claude's. If a report says "94.67% accuracy" without noting it's synthetic on-grid data, flag it.
4. **Suggest an experiment instead of assuming.** If something is unverified and verifiable, say what test or measurement would confirm it, instead of asserting an answer.
5. **No invented architecture.** Don't design a system component, pipeline stage, or interface that wasn't discussed or implemented, and present it as if it exists.

## Output format
- Keep it short: one line per flagged claim is enough.
- Don't turn this into a lecture on epistemics. Flag, state what's known vs not, move on.
