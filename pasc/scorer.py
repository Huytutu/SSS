from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn.functional as F

__all__ = ["Evidence", "EvidenceScorer"]


@dataclass
class Evidence:
    """One short visual fact, produced by the corrector after a crop."""

    id: str
    text: str
    step: Optional[int] = None
    bbox: Optional[tuple] = None


class EvidenceScorer:
    """Scores candidate tokens against a growing bank of evidence sentences.

    The question each evidence sentence answers is: "reading you, how surprised
    would I be to see this token next?" A token a sentence supports is cheap; a
    token nothing supports is expensive. That cost then reweights the model's
    own distribution (see logits_processor.py).

    Each sentence is encoded once, into per-position next-token log-probs over
    the whole vocabulary. Scoring is then a lookup with no forward pass, which
    is what makes this affordable inside a decoding loop.

    Two aggregations, both deliberate:
      * over positions within a sentence -- mean of the probabilities, so a
        token the sentence anticipates *anywhere* counts, not only at the end;
      * over sentences -- soft-min, so one supporting sentence is enough to
        clear a token, rather than being averaged away by unrelated ones.
    """

    def __init__(self, model, tokenizer, cfg):
        self.model = model
        self.tokenizer = tokenizer
        self.cfg = cfg
        self.device = next(model.parameters()).device
        self._nll: List[torch.Tensor] = []      # per evidence: [positions, vocab] on CPU
        self._items: List[Evidence] = []
        self._vocab_size: Optional[int] = None

    def __len__(self) -> int:
        return len(self._items)

    @property
    def items(self) -> List[Evidence]:
        return list(self._items)

    @torch.inference_mode()
    def add(self, ev: Evidence) -> None:
        text = (ev.text or "").strip()
        if not text or text.lower() == "none":
            return
        enc = self.tokenizer([text], return_tensors="pt", add_special_tokens=True)
        input_ids = enc["input_ids"].to(self.device)
        logits = self.model(input_ids=input_ids, use_cache=False).logits

        keep = min(logits.shape[1], self.cfg.max_evidence_prefix)
        logp = F.log_softmax(logits[0, :keep, :].float(), dim=-1)
        self._vocab_size = int(logp.shape[-1])
        self._nll.append((-logp).to(torch.float16).cpu())
        self._items.append(ev)

    @torch.no_grad()
    def cost(self, cand_ids: torch.Tensor) -> torch.Tensor:
        """One consistency cost per candidate. Zeros when the bank is empty, so
        an un-primed scorer leaves the model's distribution untouched."""
        n = int(cand_ids.numel())
        if n == 0 or not self._nll:
            return torch.zeros(n, dtype=torch.float32)

        ids = cand_ids.detach().to("cpu", torch.long)
        # Smoothing floor: without it, a token no evidence anticipates gets an
        # unbounded cost and would be excluded outright rather than penalised.
        eps = 1e-6 / max(self._vocab_size or 1, 1)

        per_evidence = []
        for nll in self._nll:
            log_q = -nll[:, ids].float()                       # [positions, n]
            log_mean_q = torch.logsumexp(log_q, dim=0) - math.log(log_q.shape[0])
            smoothed = torch.logaddexp(
                log_mean_q + math.log1p(-eps),
                torch.full_like(log_mean_q, math.log(eps)),
            )
            per_evidence.append(-smoothed)

        stacked = torch.stack(per_evidence, dim=0)
        tau = max(self.cfg.evidence_agg_tau, 1e-6)
        return -tau * (torch.logsumexp(-stacked / tau, dim=0) - math.log(stacked.shape[0]))
