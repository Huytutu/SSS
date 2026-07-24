from __future__ import annotations
from dataclasses import dataclass
import re
from typing import List, Optional
import torch
from transformers import PreTrainedTokenizerBase

_CRITICAL_KEYWORDS = {
    "red", "green", "blue", "yellow", "black", "white", "orange", "purple", "brown", "pink", "gray", "grey",
    "left", "right", "above", "below", "behind", "front", "between", "next", "near", "far", "closest", "farthest",
    "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "many", "few",
}


def _is_word_start(tok: str) -> bool:
    return tok.startswith("Ġ") or tok.startswith("▁") or tok.startswith(" ")


# Signs that the last stretch of generated text is a math derivation rather than a
# description of the image: an equals sign, or common LaTeX math commands/delimiters.
_MATH_CONTEXT_RE = re.compile(r"=|\\frac|\\times|\\div|\\sqrt|\\boxed|\\cdot|\\\(|\\\)|\\\[|\\\]|\$")


def _in_math_context(prefix_tail: str, window: int = 40) -> bool:
    return bool(_MATH_CONTEXT_RE.search(prefix_tail[-window:]))


def _looks_critical(tok: str, prefix_tail: str = "") -> bool:
    text = tok.lower().lstrip("Ġ▁ ").strip()
    if not text:
        return False
    if re.fullmatch(r"[0-9]+", text):
        # A bare number is only "critical" (worth asking a visual decider about) if it
        # isn't a value computed mid-derivation -- GRIT can look at the image again, but
        # it can't check whether an algebra step is correct.
        return not _in_math_context(prefix_tail)
    if text in _CRITICAL_KEYWORDS:
        return True
    return any(text.startswith(p) for p in ["color", "colou", "num", "count", "id"])


def _passes_word_gate(cand_ids: List[int], input_ids_row: torch.LongTensor, tokenizer: Optional[PreTrainedTokenizerBase]) -> bool:
    if tokenizer is None or not cand_ids:
        return False
    try:
        tok = tokenizer.convert_ids_to_tokens(cand_ids[0])
    except Exception:
        return False
    if _is_word_start(tok):
        return True
    # Only decode a short tail window when we actually need math-context info, since
    # decoding is the expensive part of this check.
    try:
        prefix_tail = tokenizer.decode(input_ids_row[0, -30:].tolist(), skip_special_tokens=True)
    except Exception:
        prefix_tail = ""
    return _looks_critical(tok, prefix_tail)


@dataclass
class MixedGapTrigger:
    """Trigger GRIT when the mixed top-1/top-2 gap remains small at a non-singleton knee."""
    gap_thresh: float = 0.08
    min_k: int = 2
    cooldown: int = 5
    last_fire_step: int = -999999

    def __call__(
        self,
        last_info: dict,
        input_ids: torch.LongTensor,
        tokenizer: Optional[PreTrainedTokenizerBase],
    ) -> bool:
        step = int(last_info.get("step", -1))
        k = int(last_info.get("k", 1))
        gap = float(last_info.get("gap", 1.0))
        if k < self.min_k or gap > self.gap_thresh:
            return False
        if step - self.last_fire_step < self.cooldown:
            return False
        cand_ids: List[int] = list(map(int, list(last_info.get("cand_ids", [])[:1])))
        if not _passes_word_gate(cand_ids, input_ids, tokenizer):
            return False
        self.last_fire_step = step
        return True