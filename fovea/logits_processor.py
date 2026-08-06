from __future__ import annotations
from typing import Any, Dict, List, Optional, Union

import os

import torch
from transformers import LogitsProcessor, PreTrainedTokenizerBase

from .scorer import PromptDeviationScorer

__all__ = ["VDGDLogitsProcessor", "knee_topk"]

EPS = 1e-12


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    return float(value) if value is not None else float(default)


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return int(value) if value is not None else int(default)


def knee_topk(p_sorted: torch.Tensor, min_k: int, max_k: int) -> int:
    """Maximum-gap knee: k* = argmax_k (p_(k) - p_(k+1)) + 1 on an already
    sorted (descending) probability vector, clamped to [min_k, max_k]."""
    L = p_sorted.shape[0]
    if L <= 1:
        return max(1, min_k)
    diffs = p_sorted[:-1] - p_sorted[1:]
    k = int(torch.argmax(diffs).item()) + 1
    return max(min_k, min(k, max_k))


class VDGDLogitsProcessor(LogitsProcessor):
    """VDGD decoder (knee-truncated candidate set + eq. 3 deviation re-scoring)."""

    def __init__(
        self,
        scorer: Union[PromptDeviationScorer, List[PromptDeviationScorer]],
        *,
        min_k: Optional[int] = None,
        max_k: Optional[int] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        neg_inf_val: float = -1e9,
        collect_step_log: bool = False,
    ):
        super().__init__()
        self.scorer = scorer
        self.min_k = _env_int("VDGD_MIN_K", 1) if min_k is None else int(min_k)
        self.max_k = _env_int("VDGD_MAX_K", 64) if max_k is None else int(max_k)
        self.tokenizer = tokenizer
        self.neg_inf_val = float(neg_inf_val)

        # Optional: log the post-rescore candidate-set distribution at every decoding
        # step, so LeCo (fovea/leco.py) can compute per-step confidence scores from the
        # same distribution VDGD actually decodes from, without a second forward pass.
        self.collect_step_log = bool(collect_step_log)
        self.step_log: List[Dict[str, Any]] = []
        self._step = 0

    def _get_scorer(self, i: int) -> PromptDeviationScorer:
        return self.scorer[i] if isinstance(self.scorer, (list, tuple)) else self.scorer

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        B, V = scores.shape
        device = scores.device
        dtype = scores.dtype
        probs = torch.softmax(scores.to(torch.float32), dim=-1)

        for i in range(B):
            probs_i = probs[i]
            p_sorted, idx_sorted = torch.sort(probs_i, descending=True)
            k_i = knee_topk(p_sorted, self.min_k, min(self.max_k, V))  # knee truncation
            cand_idx = idx_sorted[:k_i]

            scorer_i = self._get_scorer(i)
            assert scorer_i is not None, "VDGDLogitsProcessor requires a scorer."
            KL_i = scorer_i.score_token_ids(cand_idx)  # eq. 3: prompt-deviation cost

            new_logits_i = torch.full((V,), self.neg_inf_val, dtype=torch.float32, device=device)
            new_logits_i[cand_idx.to(device)] = -KL_i.to(device=device, dtype=torch.float32)
            scores[i, :] = new_logits_i.to(dtype=dtype)

            if self.collect_step_log:
                cand_probs = torch.softmax(new_logits_i[cand_idx], dim=-1)
                self.step_log.append({
                    "step": self._step,
                    "top1_prob": float(cand_probs.max().item()),
                    "cand_probs": cand_probs.detach().cpu(),
                    "cand_ids": cand_idx.detach().cpu(),
                })

        self._step += 1
        return scores
