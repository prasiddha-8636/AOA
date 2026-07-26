---
name: token-optimizer
description: Use this skill for every task involving tool calls, file operations, or long context (code review, multi-file edits, research, large reports). Governs how Claude uses tokens and tool calls efficiently so long sessions don't run out of budget. Not a writing-style skill — applies to tool/process efficiency, not prose tone.
---

# Token Optimizer

Purpose: don't burn context/tokens on redundant work. Long sessions (thesis, multi-file code, research) should stay usable end to end.

## Rules

1. **Read once, reuse.** Don't re-view a file that's already in context unless it was edited since. Track what's already been read instead of re-fetching it.
2. **Batch related operations.** Combine related edits/checks into one pass instead of many small round trips (e.g. one `str_replace` per real change, not one per line).
3. **Targeted reads, not full dumps.** Use line ranges / grep-style search on large files instead of viewing the whole file when only a section is relevant.
4. **No redundant searches.** Don't repeat a web search with near-identical wording. Reformulate only if the first attempt genuinely missed the target.
5. **Cache expensive results.** If a search, computation, or file read produced something reusable (a formula, a fetched page, a computed table), reuse it instead of redoing the work later in the same session.
6. **Trim output, not substance.** Long tool outputs (logs, search results, file contents) should be summarized to the relevant part before being carried forward in reasoning — don't restate huge blocks back verbatim.
7. **Stop when done.** Don't keep calling tools "just in case" once the task's actual question is answered. Extra verification calls need a real reason.
8. **Prefer cheaper tools first.** If a lightweight check (grep, targeted view) can confirm something, don't reach for a heavier one (full file read, extra search round) first.

## When reviewing Claude's own tool usage
- Before each tool call, ask: has this exact information already been gathered in this conversation?
- Before repeating a call, ask: did the previous attempt actually fail, or did I just not use its result?

## Command/tool output compression (RTK-style)
Applies to raw tool output before it's reasoned over — not to prose sent to Pika.
- Bash/command output (git log, test runs, build logs, dependency trees): extract only the failing/relevant lines. Don't carry forward passing-test spam, unchanged dependency listings, or repeated log lines.
- Collapse repeated identical lines/errors into one instance plus a count.
- For file listings or search results, keep only entries relevant to the task, drop the rest.
- If output is large and mostly noise, summarize it in one line ("47 tests passed, 2 failed: X, Y") instead of quoting the block.

## Terse-mode output (Caveman-style)
Applies only to quick factual/technical answers, status checks, and internal notes — never to thesis/report/paper prose, which follows Pika's story-flow writing style instead.
- For a quick technical answer (e.g. "does this compile," "what's the value of X," "is this function called anywhere"): answer in the fewest words that carry the fact. Drop articles, hedging, and framing sentences.
- Example: not "It looks like the function is not called anywhere in the codebase," but "Not called anywhere."
- Don't apply this to writing tasks. If Pika asked for a paragraph, section, or explanation, use the simple/story-flow skill instead — terse mode there would strip out the substance those tasks need.

## Output format
- No meta-commentary about token savings shown to Pika — just apply it silently.
- If a task is genuinely large and will need many calls, say so briefly up front instead of unknowingly running long.
