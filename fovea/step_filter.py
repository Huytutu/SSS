from __future__ import annotations

import re

__all__ = ["is_structural_step"]

# Rule 3: fixed set of generic transition/meta-commentary phrases. Matched as
# a whole-line prefix (after normalizing whitespace/case), not a substring
# search anywhere in the text -- a real claim that happens to mention e.g.
# "given the options" partway through a sentence should not be flagged.
_TRANSITION_PHRASES = [
    "let's analyze",
    "let's break down",
    "let's break this down",
    "given the options",
    "given this analysis",
    "to determine",
    "adding these together",
    "final answer",
    "therefore, the correct answer is",
]

# Rule 1: step is nothing but markdown/LaTeX delimiters or punctuation --
# e.g. a lone "\[" or "\]" (the RH-Bench id=21 rollback target).
_PURE_FORMATTING_RE = re.compile(r"^[\\\*\#\-\:\.\[\]\(\)\s]*$")

# Rule 2: a numbered, optionally bolded header with no content of its own --
# e.g. "1. **Identify Corresponding Angles**:" (the RH-Bench id=28 rollback
# target, picked twice in a row instead of the step with the actual error).
# Trailing "[:\*\s]*" (not just "\s*:?\s*") to tolerate a stray "**" after the
# colon (e.g. "**Identify...**:**", a markdown glitch some generations
# produce) -- TreeBench index=400 rolled back into exactly such a header
# because the extra "**" fell outside the old pattern's trailing "\s*:?\s*$".
_NUMBERED_HEADER_RE = re.compile(r"^\d+\.\s*(\*\*[^*]+\*\*)?[:\*\s]*$")


def _normalize(text: str) -> str:
    return " ".join(text.strip().lower().split())


def is_structural_step(text: str) -> bool:
    """True if `text` (one LeCo reasoning step) carries no reasoning content
    of its own -- a formatting/LaTeX delimiter, a bare numbered/bolded
    header, or a generic transition phrase -- and should therefore be
    excluded from LeCo rollback candidacy (see `exclude` in leco.py's
    find_lowest_scoring_step).

    Confirmed empirically as the cause of two independent LeCo rollback
    failures: rollback repeatedly landing on a bare "\\[" LaTeX delimiter
    instead of the step with the real reasoning error (RH-Bench id=21), and
    on "1. **Identify Corresponding Angles**:"-style header lines instead of
    the step that actually misapplied a geometry rule (RH-Bench id=28).
    These structural lines score lower under avg/dev than substantive
    content regardless of whether that content is correct, so without this
    filter they systematically hijack rollback away from the real error.

    Deliberately does NOT try to catch a step that echoes a given
    multiple-choice option verbatim (e.g. "Given the options: - A. ... -
    B. ..."). That failure mode needs the opposite treatment -- LeCo should
    be *more* willing to roll back into such a step to cut it out, not
    excluded from candidacy like the cases above -- and empirically, the
    rollback that would need to target it usually happens before the echo
    is even generated, so exclusion here wouldn't have helped anyway (see
    TreeBench index=344). Use one-shot prompting to address that pattern
    instead (fixes the model's underlying tendency to restate options,
    rather than trying to steer rollback after the fact).
    """
    stripped = text.strip()
    if not stripped:
        return True

    if _PURE_FORMATTING_RE.match(stripped):
        return True

    if _NUMBERED_HEADER_RE.match(stripped):
        return True

    normalized = _normalize(stripped)
    if any(normalized.startswith(phrase) for phrase in _TRANSITION_PHRASES):
        return True

    return False
