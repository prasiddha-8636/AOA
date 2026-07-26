---
name: performance-optimizer
description: Use this skill when Pika asks to optimize, speed up, or reduce memory/resource use of code, or when reviewing code that runs on large data (image batches, DEM tiles, panorama scraping, training loops). Not for micro-style preferences that don't change measured performance.
---

# Performance Optimizer

## Look for
- Unnecessary allocations (objects/arrays created in a loop that could be reused or hoisted out).
- Slow loops (Python-level loops over data that a library can vectorize).
- Repeated I/O (re-reading the same file, re-downloading, re-opening a connection per iteration).
- Repeated/duplicate DB or API calls that could be batched or cached.
- Missing vectorization (NumPy/PyTorch ops written as manual loops).
- Missing caching for expensive, repeatable computation (e.g. same DEM tile loaded per query).
- Memory usage — loading full datasets into memory instead of streaming, unbounded buffers, holding references longer than needed.
- Algorithmic complexity — O(n²) where O(n log n) is achievable, wasted passes over data.

## Rules
- Only suggest a change if it has a measurable benefit. If the code runs once on small input, leave it alone — say so instead of suggesting a rewrite.
- State the expected improvement concretely: complexity class change, fewer passes, fewer I/O calls, memory saved. Don't say "this should be faster" with no reason.
- Give the rewritten code directly, not a description of the idea.
- If a fix trades correctness or readability for speed, say that tradeoff plainly — don't hide it.

## Output format
- One item per issue: what's slow, why, the fix.
- No categories with nothing to report — skip them silently.
- If the code is already fine, say so in one line.
