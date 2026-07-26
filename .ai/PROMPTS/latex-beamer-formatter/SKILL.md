---
name: latex-beamer-formatter
description: Use this skill whenever writing, editing, or fixing LaTeX or Beamer files for Pika (thesis report, IEEE paper, presentation slides, .cls files). Trigger on any .tex/.cls edit, any request to format a document, fix a compile error, or add packages/figures to a LaTeX source. Also trigger proactively after editing any .tex file to catch errors before Pika compiles it.
---

# LaTeX/Beamer Formatter

Purpose: prevent LaTeX errors before they happen, and fix them immediately if they occur. A broken build the night before a defense is the failure mode this skill exists to avoid.

## Before making changes
- Read the actual .cls file and preamble in use. Don't assume a generic IEEE/Beamer template — check what Pika's file actually defines.
- Check which packages are already loaded before adding a new one. Don't load a package twice, and don't load one that conflicts with something in the .cls (e.g. font packages, geometry, hyperref order issues).
- Note the document class options in use (paper size, columns, theme) before changing layout.

## Package rules
- Only add a package if it's actually needed for the requested feature.
- Check for known conflicts before adding (e.g. `hyperref` should load near-last; `subcaption` vs `subfig` shouldn't coexist; `microtype` needs specific engines).
- If a package requires a specific compiler (XeLaTeX/LuaLaTeX vs pdfLaTeX), say so — don't add a package that silently breaks the current build.

## .cls safety
- Never edit the .cls file unless explicitly asked to. Prefer preamble-level fixes.
- If a .cls edit is required, explain exactly what changed and why, since .cls changes affect the whole document.
- Don't override class-defined commands/environments unless necessary — redefining something the class already provides is a common source of silent breakage.

## Figure/layout rules (carried from Pika's general style)
- Figures only when relevant, never for filler.
- No overlap with page bounds, text, or other boxes. Check placement specifiers (`[H]`, `[htbp]`) and float behavior, not just visual guess.
- Keep spacing and float placement clean — no orphaned floats, no text squeezed around a figure awkwardly.

## Error handling
- If a compile error is reported, don't guess at a generic fix. Identify the exact error (missing package, undefined control sequence, mismatched braces, float too large, encoding issue) from the log/message given.
- Give the corrected code directly, not a description of what's wrong.
- After fixing, check for the same error pattern elsewhere in the file so it doesn't recur.
- If unsure what caused the error without seeing the actual log, ask for it rather than guessing.

## Output format
- State what was changed and why, briefly.
- Show the corrected block of code, not just the diff description.
- Flag any package/version dependency the fix introduces.
