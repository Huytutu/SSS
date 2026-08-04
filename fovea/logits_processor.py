from __future__ import annotations
from typing import List, Optional, Union

import os

import torch
import torch.nn.functional as F
from transformers import LogitsProcessor, PreTrainedTokenizerBase

from .scorer import TextScorer
from .vision_scorer import VisionScorer

__all__ = ["VDGDLogitsProcessor", "knee_topk"]

EPS = 1e-30


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    return float(value) if value is not None else float(default)


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return int(value) if value is not None else int(default)


def _env_flag(name: str, default: int = 0) -> int:
    value = os.getenv(name)
    if value is None:
        return int(bool(default))
    return 1 if value.strip().lower() in ("1", "true", "yes", "y", "on") else 0


def _jsd(base_logp: torch.Tensor, cand_logp: torch.Tensor) -> torch.Tensor:
    """JSD-style divergence matching the ReVisiT reference implementation's
    `_revisit_decoding` (mean, not sum, over the constrained candidate set).
    base_logp: [k]. cand_logp: [..., k] (e.g. [J, |v|, k]). Returns [...]."""
    base_p = base_logp.exp()
    cand_p = cand_logp.exp()
    avg_p = 0.5 * (base_p + cand_p)
    log_avg = avg_p.clamp_min(EPS).log()
    kl1 = (avg_p * (log_avg - base_logp)).mean(dim=-1)
    kl2 = (avg_p * (log_avg - cand_logp)).mean(dim=-1)
    return 0.5 * (kl1 + kl2)


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
        scorer: Optional[Union[TextScorer, List[TextScorer]]] = None,
        *,
        min_k: Optional[int] = None,
        max_k: Optional[int] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        neg_inf_val: float = -1e9,
        vision_scorer: Optional[Union[VisionScorer, List[VisionScorer]]] = None,
        gate_min_k: Optional[int] = None,
    ):
        super().__init__()
        if scorer is None and vision_scorer is None:
            raise ValueError("VDGDLogitsProcessor requires a TextScorer, a VisionScorer, or both.")
        self.scorer = scorer
        self.min_k = _env_int("VDGD_MIN_K", 1) if min_k is None else int(min_k)
        self.max_k = _env_int("VDGD_MAX_K", 64) if max_k is None else int(max_k)
        self.tokenizer = tokenizer
        self.neg_inf_val = float(neg_inf_val)
        self.vision_scorer = vision_scorer
        self.debug = _env_flag("VDGD_DEBUG", 0)
        # "jsd" = ReVisiT's constrained-divergence selection. "random" is the
        # control arm: if it performs the same, the selection is decorative.
        self.vision_select = os.getenv("VDGD_VISION_SELECT", "jsd").strip().lower()
        self._step = 0
        self.vision_trace: List[dict] = []

        # Entry gate (Stage 0 of the look-again/think-again pipeline): reuses
        # knee_topk's own k -- no separate gap/keyword machinery. k_i is
        # already "how many tokens are plausible right now"; a wide knee IS
        # the ambiguity signal, nothing new to compute. Fires only on k_i >=
        # gate_min_k; join gate_trace with vision_trace on "step" to also see
        # that step's jsd_min/median/max (Stage 1 needs both).
        self.gate_min_k = _env_int("VDGD_GATE_MIN_K", 4) if gate_min_k is None else int(gate_min_k)
        self.gate_trace: List[dict] = []

    def _record_gate(self, cand_idx: torch.Tensor, cand_probs: torch.Tensor) -> None:
        if int(cand_idx.numel()) < self.gate_min_k:
            return
        cand_ids = cand_idx.detach().cpu().tolist()
        event = {
            "step": self._step,
            "k": len(cand_ids),
            "cand_ids": cand_ids,
            "cand_probs": cand_probs.detach().cpu().tolist(),
        }
        if self.tokenizer is not None:
            try:
                event["cand_tokens"] = self.tokenizer.convert_ids_to_tokens(cand_ids)
            except Exception:
                pass
        self.gate_trace.append(event)

    def _get_scorer(self, i: int) -> Optional[TextScorer]:
        if self.scorer is None:
            return None
        return self.scorer[i] if isinstance(self.scorer, (list, tuple)) else self.scorer

    def _get_vision_scorer(self, i: int) -> Optional[VisionScorer]:
        if self.vision_scorer is None:
            return None
        return self.vision_scorer[i] if isinstance(self.vision_scorer, (list, tuple)) else self.vision_scorer

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
            self._record_gate(cand_idx, p_sorted[:k_i])

            scorer_i = self._get_scorer(i)
            vision_scorer_i = self._get_vision_scorer(i)
            new_logits_i = torch.full((V,), self.neg_inf_val, dtype=torch.float32, device=device)

            if vision_scorer_i is None:
                # Text-only: original VDGD full override, no base/vision terms.
                assert scorer_i is not None, "VDGDLogitsProcessor requires a TextScorer when vision_scorer is absent."
                KL_i = scorer_i.score_token_ids(cand_idx)  # eq. 3: prompt-deviation cost
                new_logits_i[cand_idx.to(device)] = -KL_i.to(device=device, dtype=torch.float32)
            else:
                refined_c = self._refine(scores[i], cand_idx, scorer_i, vision_scorer_i)
                new_logits_i[cand_idx.to(device)] = refined_c

            scores[i, :] = new_logits_i.to(dtype=dtype)
            self._step += 1

        return scores

    def _refine(
        self,
        scores_i: torch.Tensor,
        cand_idx: torch.Tensor,
        scorer_i: Optional[TextScorer],
        vision_scorer_i: VisionScorer,
    ) -> torch.Tensor:
        """Log-space refinement (ReVisiT Sec. 4.3): base + vision, plus text
        when a TextScorer is attached (scorer_i is None -> plain ReVisiT,
        base+vision only, matching the reference implementation). Every term
        is log_softmax'd over `cand_idx` (the paper's V_cons^t) before
        summing -- see the plan doc for why this matters."""
        device = scores_i.device
        base_logp_c = F.log_softmax(scores_i[cand_idx].to(torch.float32), dim=-1)

        text_logp_c = 0.0
        if scorer_i is not None:
            KL_i = scorer_i.score_token_ids(cand_idx)  # eq. 3: prompt-deviation cost
            text_logp_c = F.log_softmax((-KL_i).to(device=device, dtype=torch.float32), dim=-1)

        vis_logp_c = vision_scorer_i.score_token_ids(cand_idx).to(device)  # [J, |v|, k]
        jsd = _jsd(base_logp_c, vis_logp_c)  # [J, |v|]
        jsd_flat = jsd.reshape(-1)
        if self.vision_select == "random":
            flat_idx = int(torch.randint(jsd_flat.numel(), (1,)).item())
        else:
            flat_idx = int(jsd_flat.argmin().item())
        j_star, i_star = divmod(flat_idx, vis_logp_c.shape[1])
        vis_logp_star = vis_logp_c[j_star, i_star]  # [k]
        layer = vision_scorer_i.layers[j_star]

        self.vision_trace.append({
            "step": self._step,
            "k": int(cand_idx.numel()),
            "layer": layer,
            "vision_idx": i_star,
            "jsd": float(jsd[j_star, i_star].item()),
            # Spread over all (layer, vision_token) pairs: without real range
            # here, the argmin above is decided by float noise, not evidence.
            "jsd_min": float(jsd_flat.min().item()),
            "jsd_median": float(jsd_flat.median().item()),
            "jsd_max": float(jsd_flat.max().item()),
        })
        if self.debug and self._step < 50:
            print(f"[VDGD-DEBUG] step={self._step} chosen vision token: layer={layer} idx={i_star} jsd={jsd[j_star, i_star]:.4f}")

        return base_logp_c + text_logp_c + vis_logp_star
