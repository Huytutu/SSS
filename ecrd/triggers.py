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


def _looks_critical(tok: str) -> bool:
    text = tok.lower().lstrip("Ġ▁ ").strip()
    if not text:
        return False
    if re.fullmatch(r"[0-9]+", text):
        return True
    if text in _CRITICAL_KEYWORDS:
        return True
    return any(text.startswith(p) for p in ["color", "colou", "num", "count", "id"])


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
        if tokenizer is None or not cand_ids:
            return False
        try:
            tok = tokenizer.convert_ids_to_tokens(cand_ids[0])
        except Exception:
            return False
        if not _is_word_start(tok) and not _looks_critical(tok):
            return False
        self.last_fire_step = step
        return True


@dataclass
class ConformalTrigger:
    """Trigger GRIT when the APS conformal prediction set built from p_mix is not a
    singleton. Replaces MixedGapTrigger's gap_thresh heuristic with a threshold q_hat
    calibrated offline (see scripts/calib/) so that the set covers the true token with
    probability >= 1 - alpha on held-out data. Same gate conditions (min_k, cooldown,
    word-start/critical-keyword) as MixedGapTrigger; only the confidence check differs.
    """
    q_hat: float
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
        if k < self.min_k:
            return False
        if step - self.last_fire_step < self.cooldown:
            return False
        cand_ids: List[int] = list(map(int, list(last_info.get("cand_ids", [])[:1])))
        if tokenizer is None or not cand_ids:
            return False
        try:
            tok = tokenizer.convert_ids_to_tokens(cand_ids[0])
        except Exception:
            return False
        if not _is_word_start(tok) and not _looks_critical(tok):
            return False

        cand_probs = last_info.get("cand_mix_probs")
        if cand_probs is None or len(cand_probs) == 0:
            return False
        sorted_probs, _ = torch.sort(torch.as_tensor(cand_probs), descending=True)
        cum = 0.0
        set_size = 0
        for p in sorted_probs.tolist():
            cum += p
            set_size += 1
            if cum >= self.q_hat:
                break
        if set_size <= 1:
            return False

        self.last_fire_step = step
        return True
