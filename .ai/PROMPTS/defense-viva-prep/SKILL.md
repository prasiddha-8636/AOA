---
name: defense-viva-prep
description: Use this skill when Pika is preparing for a defense, viva, or presentation Q&A on the geo-localization project or any academic work. Covers generating practice questions, algorithm walkthroughs, formula recall drills, and romanized Nepali translations of technical answers. Trigger on requests like "quiz me," "practice questions," "what might they ask," or "translate this to Nepali for defense."
---

# Defense/Viva Prep

## What this produces
- Practice questions drawn from the actual report/code/slides — not generic viva questions unrelated to the project.
- Algorithm walkthroughs: step-by-step explanation of a method (e.g. FFT+DTW matching), matching the actual implementation, not a textbook version of the algorithm.
- Formula recall drills: formula, what each symbol means, why it's used here.
- Romanized Nepali translations of technical answers, on request.

## Rules
1. **Base questions on real weak points.** Prioritize things flagged elsewhere as fragile: synthetic-only accuracy numbers, threshold justifications, unaddressed camera roll, undefined image counts, anything a committee would push on.
2. **Answers must be defendable.** Don't give an answer that oversells a result. If the honest answer is "we didn't test that," the practice answer should say that, not dodge it.
3. **Match implementation, not generic theory.** If the algorithm walkthrough diverges from what the code does, flag the gap instead of describing the textbook version.
4. **Romanized Nepali** — natural phrasing, not literal word-for-word translation. Keep technical terms in English if that's how Pika already says them (e.g. keep "threshold," "accuracy" as-is rather than forcing a translated term).
5. **No filler questions.** Every question should target something a real examiner would probe, not padding to hit a count.

## Output format
- Question, then expected strong answer, then the honest caveat if one exists.
- Keep answers in Pika's short, direct prose style — no lecture-length answers.
