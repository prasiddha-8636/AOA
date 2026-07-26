---
name: engineering-reviewer
description: Use this skill right after writing or editing any significant piece of code — a new function, a bug fix, a refactor, a new module. Also use when Pika asks for a code review, or before saying code is "done." Not for trivial one-line changes (typo fixes, renaming) or non-code writing.
---

# Engineering Reviewer

Highest priority skill. Run this check silently before handing code back as finished. Don't wait to be asked.

## Checklist

1. **Correctness vs requirement** — Does the code actually do what was asked? Re-read the request, then the code. Flag any gap between them.
2. **Bugs and edge cases** — empty input, zero/negative values, off-by-one, null/None, wrong types, boundary conditions.
3. **Security** — injection risk, unsafe deserialization, path traversal, secrets in code, unvalidated input reaching a sink.
4. **Performance regressions** — did this change make something slower or more memory-hungry than the previous version?
5. **Simplicity** — is there a shorter or more direct way to write this? Flag unnecessary abstraction, unneeded parameters, over-engineering.
6. **Dead code** — unused variables, functions, imports, unreachable branches.
7. **Better algorithm/data structure available** — only flag if it changes complexity class or is clearly simpler, not style preference.

## Output format
- State pass/fail on requirement match first, one line.
- List only real issues found, most severe first. No issue found in a category = don't mention that category.
- For every issue: give the fix directly (a diff or corrected snippet), not a description of what to change.
- No praise padding, no "great job overall" filler.
- If nothing is wrong, say so in one line and stop.
