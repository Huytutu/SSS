from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
from transformers import LogitsProcessor

from .scorer import Evidence
from .trigger import is_groundable

__all__ = ["PASCLogitsProcessor", "nucleus_k"]

EPS = 1e-30


def _is_content_token(token):
    """A token that could plausibly replace a visual claim: a real word start,
    not punctuation or whitespace. Keeps the corrector from swapping a word for
    a newline while still allowing "one" -> "two" or "left" -> "right"."""
    if not token or not token.startswith(("Ġ", "▁")):
        return False
    word = token.lstrip("Ġ▁")
    return bool(word) and any(c.isalnum() for c in word)


def nucleus_k(sorted_probs: torch.Tensor, top_p: float, min_k: int, max_k: int) -> int:
    """How many candidates the model is still taking seriously: the smallest set
    whose probabilities sum to `top_p`.

    This replaces a largest-drop ("knee") rule, which turned out to be
    degenerate here. In a steeply decaying distribution the biggest single drop
    is nearly always the one right after the top token, so the knee returned
    k=1 for 87% of tokens (11143 of 12804 measured) -- including tokens where
    the model was genuinely torn, e.g. p_top=0.33. That silently starved the
    corrector, which can only choose among candidates it is given.
    """
    if sorted_probs.numel() <= 1:
        return max(1, min_k)
    cumulative = torch.cumsum(sorted_probs, dim=0)
    k = int(torch.searchsorted(cumulative, torch.tensor(top_p, device=cumulative.device))) + 1
    return max(min_k, min(k, max_k))


class PASCLogitsProcessor(LogitsProcessor):
    """Runs the whole loop for one decoding step.

    Order matters, and it is:
      1. read the attention signals for the step that just finished;
      2. narrow the next-token distribution to a candidate set (nucleus cut);
      3. reweight those candidates by how well the evidence bank supports them;
      4. if the trigger flagged the token, crop the image, re-ask, apply the answer,
         and keep the evidence for every later step.

    Step 4 is what costs money -- one extra generate() per firing -- so the
    trigger's job is to be stingy. Steps 1-3 are free.

    One deliberate departure from PAS. PAS scores token y_k using the attention
    row *at* y_k, which only exists once y_k has been fed back in -- by then the
    token is already emitted and only a rollback could change it. With a KV
    cache, the row available while choosing y_k is the one from y_{k-1}. So this
    reads the signal as "how grounded was the model's state going into this
    choice", one position earlier than the paper's definition. That is what
    makes intervention possible at all without rewinding, and it is the same
    offset the AUROC measurements in PASConfig were taken under.

    Everything the run did is recorded in `correction_log`, one entry per
    firing; that is what the demo prints and what the TreeBench eval writes out.
    """

    def __init__(self, model, processor, cfg, probe, scorer, trigger, corrector=None):
        self.model = model
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.cfg = cfg
        self.probe = probe
        self.scorer = scorer
        self.trigger = trigger
        self.corrector = corrector

        self.step = 0
        self.step_log: List[Dict[str, Any]] = []
        self.correction_log: List[Dict[str, Any]] = []

    @property
    def n_corrections(self) -> int:
        return len(self.correction_log)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        signals = self.probe.pop_step()
        probs = torch.softmax(scores[0].float(), dim=-1)
        sorted_probs, sorted_ids = torch.sort(probs, descending=True)

        k = nucleus_k(sorted_probs, self.cfg.top_p, self.cfg.min_k, min(self.cfg.max_k, probs.numel()))
        cand_ids = sorted_ids[:k]
        top_token = self.tokenizer.convert_ids_to_tokens(int(cand_ids[0]))

        # Evidence only arbitrates tokens that could carry a visual claim --
        # reweighting punctuation and function words by visual-consistency cost
        # is meaningless.
        #
        # Known limitation: this does NOT protect the output format. A token
        # like "Answer" is word-initial, alphabetic and not a stopword, so it
        # passes is_groundable and still gets reweighted. On TreeBench index=8
        # that is enough to make the model write "Answer: A</think>" instead of
        # "<answer>A</answer>", losing the answer to the parser even though no
        # token was forced. Excluding format vocabulary would need the caller to
        # declare it, which nothing here does yet.
        prefix_tail = self.tokenizer.decode(input_ids[0, -8:].tolist(), skip_special_tokens=False)
        groundable = is_groundable(top_token, self.cfg, prefix_tail)
        if groundable:
            scores = self._apply_evidence(scores, probs, cand_ids, float(sorted_probs[0]))

        gap = float(sorted_probs[0] - sorted_probs[1]) if sorted_probs.numel() > 1 else 1.0
        fire = self.corrector is not None and self.trigger.should_fire(
            signals, top_token, self.step, gap, k, prefix_tail)

        self.step_log.append({
            "step": self.step,
            "k": k,
            "top_token": top_token,
            "top_prob": float(sorted_probs[0]),
            "pas_raw": signals.pas_raw,
            "pas_share": signals.pas_share,
            "peak_diff": signals.peak_diff,
            "local_ratio": signals.local_ratio,
            "z": self.trigger.last_z,
            "gap": gap,
            "groundable": groundable,
            "fired": False,
        })

        if fire:
            # Show the corrector top-N by rank, not by probability mass: on a
            # confidently wrong token the right word carries ~1e-4 and never
            # enters the nucleus, but does sit in the top few by rank.
            n = max(k, self.cfg.correct_top_n)
            scores = self._correct(input_ids, scores, signals, sorted_ids[:n], cand_ids, top_token)

        self.step += 1
        return scores

    def _apply_evidence(self, scores, probs, cand_ids, top_prob):
        """Blend the model's distribution with one shaped by the evidence bank.

        The blend weight is the model's own top probability: where it is sure,
        it keeps its distribution; where it is unsure, evidence gets a say.
        With an empty bank the costs are all zero and this is a no-op.
        """
        if len(self.scorer) == 0:
            return scores

        cost = self.scorer.cost(cand_ids).to(scores.device)
        evidence_logits = torch.full_like(probs, -1e9)
        evidence_logits[cand_ids] = -cost / max(self.cfg.rescore_tau, 1e-6)
        evidence_probs = torch.softmax(evidence_logits, dim=-1)

        # Put the evidence distribution on the same footing as the candidates'
        # share of the original mass, so blending can reorder candidates without
        # inflating their total probability.
        evidence_probs = evidence_probs * (probs[cand_ids].sum() / (evidence_probs[cand_ids].sum() + EPS))

        alpha = top_prob if self.cfg.mix_alpha is None else self.cfg.mix_alpha
        alpha = min(alpha, self.cfg.max_mix_alpha)
        mixed = alpha * probs + (1.0 - alpha) * evidence_probs
        scores[0] = torch.log(mixed + EPS).to(scores.dtype)
        return scores

    def _correct(self, input_ids, scores, signals, shown_ids, knee_ids, top_token):
        """Crop, re-ask, and apply what the crop says.

        The correction is deliberately allowed to pick a token far outside the
        nucleus. That is the whole point: a confident hallucination puts ~1e-4
        on the right word. Measured over 113 count/side/colour tokens, the
        correct alternative sat at median rank 3-8 -- reachable by rank, never
        by probability (nucleus k was 1 for 90% of left/right tokens at
        p_top=0.999). An earlier version only applied candidates inside the
        nucleus, which made it structurally unable to fix exactly the errors it
        was built for.

        What keeps that from producing wreckage -- an earlier run turned "Look"
        into a newline -- is filtering the candidate list to tokens that could
        be a visual claim in the first place, rather than trusting probability
        to do it.
        """
        candidates = [
            {"id": int(i), "text": self.tokenizer.convert_ids_to_tokens(int(i))}
            for i in shown_ids.tolist()
            if _is_content_token(self.tokenizer.convert_ids_to_tokens(int(i)))
        ]
        if len(candidates) < 2:
            return scores
        prefix_text = self.tokenizer.decode(input_ids[0].tolist(), skip_special_tokens=True)

        result = self.corrector.correct(signals.img_row, prefix_text, candidates)
        if result is None:
            return scores

        chosen = result["choice_id"]
        applied = chosen != int(knee_ids[0])
        if applied:
            scores[0, chosen] = float(scores[0].max()) + 50.0

        if result["evidence"]:
            self.scorer.add(Evidence(id=f"crop-{self.step}", text=result["evidence"],
                                     step=self.step, bbox=result["bbox"]))

        self.step_log[-1]["fired"] = True
        self.correction_log.append({
            "step": self.step,
            "pas_raw": signals.pas_raw,
            "z": self.trigger.last_z,
            "gap": self.trigger.last_gap,
            "original_token": top_token,
            "chosen_token": result["choice_text"],
            "applied": applied,
            "bbox": result["bbox"],
            "evidence": result["evidence"],
            "n_candidates": len(candidates),
            "n_knee": int(knee_ids.numel()),
        })
        return scores
