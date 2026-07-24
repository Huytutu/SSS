# CLAUDE.md / AGENTS.md
Always use conda "medcxr" environment
## Behavioral Guidelines
Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

**Top priority:** code must be easy to understand, easy to read, and easy for an engineer to follow. When any guideline conflicts with clarity, clarity wins.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 1. Readability & Clear Structure

**Write for the next person who reads this, not for the computer.**

- Use clear, descriptive names. A variable's name should tell you what it holds without reading the rest.
- Prefer flat, linear logic. A plain `if`/`elif` chain beats a clever abstraction the reader has to decode.
- Keep functions short and single-purpose. If you have to scroll to understand one function, split it.
- Avoid patterns the reader would need to look up. No metaprogramming, no clever one-liners when a few plain lines are clearer.
- Keep signatures plain. Plain params with simple defaults (`def f(data, threshold=0.5):`) over heavy type annotations (`def f(data: set[str], x: Any | None = None):`) unless asked. No extra words.
- Order code top-to-bottom in reading order: the main flow first, helpers below.
- Add a short comment only where the *why* isn't obvious from the code. Don't narrate what the code already says.

The test: could an undergraduate engineer read this once and explain what it does?

## 2. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 3. Simplicity First

**Default to the simplest code that solves the problem — every time, not only when asked. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it before showing it.

Write code the next reader understands on the first pass, using no pattern they'd have to look up.
After writing, reread and cut: would a senior engineer call this overcomplicated? If yes, simplify before you finish — not in a later pass.

## 4. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 5. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

---

**These guidelines are working if:** code reads clearly on the first pass, fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.
