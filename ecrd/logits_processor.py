from __future__ import annotations
from typing import Optional, List, Dict, Any
import os
import traceback
import torch
from transformers import LogitsProcessor, PreTrainedTokenizerBase
from .evidence import Evidence
from .triggers import _is_word_start, _looks_critical

EPS = 1e-30


def _env_flag(name: str, default: int = 0) -> int:
    value = os.getenv(name)
    if value is None:
        return int(bool(default))
    return 1 if value.strip().lower() in ("1", "true", "yes", "y", "on") else 0


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    return float(value) if value is not None else float(default)


def _vis_token(t: str) -> str:
    if t is None:
        return "<NULL>"
    return t.replace("\n", "\\n").replace("\t", "\\t").replace("Ġ", "␠").replace("▁", "␠").replace(" ", "␠")


def _fmt_top(ids: torch.Tensor, probs: torch.Tensor, topk: int, tokenizer: Optional[PreTrainedTokenizerBase] = None) -> str:
    topk = min(topk, ids.numel())
    items = []
    for j in range(topk):
        tid = int(ids[j].item())
        pv = float(probs[ids[j]].item())
        if tokenizer is not None:
            try:
                tok = tokenizer.convert_ids_to_tokens(tid)
            except Exception:
                traceback.print_exc()
                tok = None
            tok_s = _vis_token(tok) if isinstance(tok, str) else "<UNK>"
            items.append(f"{tok_s}|<{tid}>({pv:.4f})")
        else:
            items.append(f"<{tid}>({pv:.4f})")
    return " | ".join(items)


def knee_topk(p_sorted: torch.Tensor, min_k: int, max_k: int) -> int:
    """Maximum-gap knee on an already sorted probability vector."""
    L = p_sorted.shape[0]
    if L <= 1:
        return max(1, min_k)
    diffs = p_sorted[:-1] - p_sorted[1:]
    k = int(torch.argmax(diffs).item()) + 1
    return max(min_k, min(k, max_k))


class ECRDLogitsProcessor(LogitsProcessor):
    """ECRD supervisor implemented as a HuggingFace LogitsProcessor."""

    def __init__(
        self,
        scorer: Optional[object] = None,
        scorers: Optional[List[object]] = None,
        *,
        min_k: Optional[int] = None,
        max_k: Optional[int] = None,
        tau: Optional[float] = None,
        strict: Optional[bool] = None,
        mix_alpha: Optional[float] = None,
        mix_reweight: Optional[int] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        neg_inf_val: float = -1e9,
        collect_calibration_log: bool = False,
        calib_min_k: int = 2,
        calib_cooldown: int = 5,
        collect_branch_checkpoints: bool = False,
    ):
        super().__init__()
        self.scorer = scorer
        self.scorers = scorers
        self.min_k = int(os.getenv("ECRD_MIN_K", "1")) if min_k is None else int(min_k)
        self.max_k = int(os.getenv("ECRD_MAX_K", "64")) if max_k is None else int(max_k)
        self.tau = _env_float("ECRD_REWEIGHT_TAU", 1.0) if tau is None else float(tau)
        self.strict = bool(_env_flag("ECRD_STRICT", 0)) if strict is None else bool(strict)
        self.tokenizer = tokenizer
        if mix_alpha is None:
            env_alpha = os.getenv("ECRD_LOGMIX_ALPHA", "0.6")
            self.mix_alpha = float(env_alpha) if env_alpha is not None else None
        else:
            self.mix_alpha = float(mix_alpha)
        self.mix_reweight = _env_flag("ECRD_MIX_REWEIGHT", 1) if mix_reweight is None else int(mix_reweight)
        self.neg_inf_val = float(neg_inf_val)

        # ECRD paper default: alpha_eff = p_top when adaptation is enabled.
        self.alpha_adapt = _env_flag("ECRD_ALPHA_ADAPT", 1)
        self.alpha_ref_top = _env_float("ECRD_ALPHA_REF_TOP", 0.6)
        self.alpha_slope = _env_float("ECRD_ALPHA_SLOPE", 0.8)

        self.debug = _env_flag("ECRD_DEBUG", 0)
        self.debug_top = int(os.getenv("ECRD_DEBUG_TOP", "10"))
        self.debug_max = int(os.getenv("ECRD_DEBUG_MAX", "50"))
        self._step = 0

        self._grit_hook = None
        self._trigger = None
        self._ev_pool = None
        self._question = None
        self._image = None
        self.last_info: List[Dict[str, Any]] = []

        # Optional: collect gate-eligible (step, cand_mix_probs, chosen_id) records for
        # building a Conformal-ECRD calibration set. Independent of the GRIT trigger's
        # own cooldown, since we want every step a CP-trigger *could* fire on, not only
        # the ones the currently-active trigger happened to fire on.
        self.collect_calibration_log = bool(collect_calibration_log)
        self.calib_log: List[Dict[str, Any]] = []
        self._calib_min_k = int(calib_min_k)
        self._calib_cooldown = int(calib_cooldown)
        self._calib_last_fire_step = -999999

        # Counts every time the trigger fires and the GRIT decider is actually called,
        # for comparing invocation rate across trigger implementations (gap vs conformal).
        self.grit_invocations = 0

        # Optional: record near-tie steps so branch_probe can later re-run each one
        # with the runner-up token forced and see whether the final answer changes.
        # Records the top-2 ids of the *effective* (mixed) distribution, i.e. what
        # decoding actually sampled from. Assumes batch size 1, like the GRIT hook.
        self.collect_branch_checkpoints = bool(collect_branch_checkpoints)
        self.branch_checkpoints: List[Dict[str, Any]] = []
        self._branch_gap_thresh = _env_float("ECRD_BRANCH_GAP_THRESH", 0.08)

    def _calib_gate_eligible(self, info: Dict[str, Any]) -> bool:
        if info["k"] < self._calib_min_k:
            return False
        if info["step"] - self._calib_last_fire_step < self._calib_cooldown:
            return False
        cand_ids = info["cand_ids"]
        if self.tokenizer is None or len(cand_ids) == 0:
            return False
        try:
            tok = self.tokenizer.convert_ids_to_tokens(int(cand_ids[0]))
        except Exception:
            return False
        return _is_word_start(tok) or _looks_critical(tok)

    def set_grit_runtime(self, *, hook, trigger, evidence_pool, question: str = "", image: Any = None):
        self._grit_hook = hook
        self._trigger = trigger
        self._ev_pool = evidence_pool
        self._question = question
        self._image = image

    def _get_scorer(self, i: int):
        return self.scorers[i] if self.scorers is not None else self.scorer

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        B, V = scores.shape
        device = scores.device
        dtype = scores.dtype
        probs = torch.softmax(scores.to(torch.float32), dim=-1)
        self.last_info = []

        for i in range(B):
            probs_i = probs[i]
            p_sorted, idx_sorted = torch.sort(probs_i, descending=True)
            k_i = knee_topk(p_sorted, self.min_k, min(self.max_k, V))
            cand_idx = idx_sorted[:k_i]

            scorer_i = self._get_scorer(i)
            assert scorer_i is not None, "ECRDLogitsProcessor requires a scorer."
            S_i = scorer_i.score_token_ids(input_ids[i:i + 1], cand_idx)

            vdgd_logits_i = torch.full((V,), self.neg_inf_val, dtype=torch.float32, device=device)
            vdgd_logits_i[cand_idx] = (-S_i / max(self.tau, 1e-6)).to(device=device, dtype=torch.float32)
            vdgd_probs_i = torch.softmax(vdgd_logits_i, dim=-1)

            mix_probs_i = None
            alpha_eff = None
            if self.strict or (self.mix_alpha is None):
                scores[i, :].fill_(self.neg_inf_val)
                scores[i, :] = vdgd_logits_i.to(dtype=dtype)
            else:
                mass_base_c = probs_i[cand_idx].sum().to(torch.float32)
                if self.mix_reweight:
                    mass_vdgd_c = vdgd_probs_i[cand_idx].sum() + EPS
                    scale = (mass_base_c / mass_vdgd_c).clamp(min=0.0, max=1e6)
                    vdgd_probs_i = vdgd_probs_i * scale

                p_top = float(p_sorted[0].item())
                alpha_eff = float(self.mix_alpha)
                if self.alpha_adapt:
                    alpha_eff = p_top

                mix_probs_i = alpha_eff * probs_i.to(torch.float32) + (1.0 - alpha_eff) * vdgd_probs_i
                scores[i, :] = torch.log(mix_probs_i + EPS).to(dtype=dtype)

            eff_probs = mix_probs_i if mix_probs_i is not None else vdgd_probs_i
            eff_sorted, eff_idx = torch.sort(eff_probs, descending=True)
            p1 = float(eff_sorted[0].item())
            p2 = float(eff_sorted[1].item()) if eff_sorted.numel() > 1 else 0.0
            gap = p1 - p2
            info = dict(
                step=self._step,
                k=k_i,
                cand_ids=cand_idx.detach().cpu(),
                cand_mix_probs=eff_probs[cand_idx].detach().cpu(),
                base_top=float(p_sorted[0].item()),
                p1=p1,
                p2=p2,
                gap=gap,
                top_ids=eff_idx[:10].detach().cpu(),
                knee_mode="knee",
                alpha_eff=alpha_eff if alpha_eff is not None else None,
            )
            self.last_info.append(info)

            if self.collect_branch_checkpoints and k_i >= 2 and gap <= self._branch_gap_thresh:
                top2 = info["top_ids"]
                if top2.numel() >= 2:
                    self.branch_checkpoints.append({
                        "step": info["step"],
                        "gap": gap,
                        "rank0_id": int(top2[0].item()),
                        "rank1_id": int(top2[1].item()),
                    })

            if (self._grit_hook is not None) and (self._trigger is not None):
                try:
                    if self._trigger(info, input_ids[i:i + 1], self.tokenizer):
                        self.grit_invocations += 1
                        candidates = []
                        for tid in cand_idx.tolist():
                            try:
                                tok = self.tokenizer.convert_ids_to_tokens(int(tid)) if self.tokenizer else str(tid)
                            except Exception:
                                traceback.print_exc()
                                tok = str(tid)
                            candidates.append({"id": int(tid), "text": tok})

                        prefix_text = ""
                        if self.tokenizer is not None:
                            try:
                                prefix_text = self.tokenizer.decode(input_ids[i].tolist(), skip_special_tokens=True)
                            except Exception:
                                traceback.print_exc()

                        result = self._grit_hook(
                            image=self._image,
                            question=self._question or "",
                            prefix_text=prefix_text,
                            candidates=candidates,
                        )
                        forced_id = None
                        if result is not None:
                            if result.get("choice_id") is not None:
                                forced_id = int(result["choice_id"])
                            elif result.get("choice_text"):
                                txt = str(result["choice_text"]).strip()
                                for c in candidates:
                                    if c["text"] == txt:
                                        forced_id = int(c["id"])
                                        break

                        if forced_id is not None:
                            base_scores = scores[i, :].clone()
                            bias = base_scores.max().detach() + 50.0
                            scores[i, :] = base_scores
                            scores[i, forced_id] = bias

                            ev_text = str(result.get("unified_evidence") or "").strip()
                            if self._ev_pool is not None:
                                try:
                                    if ev_text and hasattr(self._ev_pool, "add_evidence"):
                                        self._ev_pool.add_evidence(Evidence(
                                            id=f"grit-{self._step}",
                                            text=ev_text,
                                            source="grit",
                                            time_step=self._step,
                                            bbox=None,
                                        ))
                                except Exception:
                                    traceback.print_exc()
                            if _env_flag("ECRD_DEBUG_GRIT", 1) and self._step < self.debug_max:
                                print(f"[ECRD-GRIT] step={self._step} FORCED id={forced_id} ev='{ev_text}'")
                except Exception as ex:
                    traceback.print_exc()
                    if self.debug and self._step < self.debug_max:
                        print(f"[ECRD-GRIT] exception: {ex}")

            if self.collect_calibration_log and self._calib_gate_eligible(info):
                self._calib_last_fire_step = info["step"]
                chosen_id = int(scores[i, :].argmax().item())
                self.calib_log.append({
                    "step": info["step"],
                    "cand_ids": info["cand_ids"].tolist(),
                    "cand_mix_probs": info["cand_mix_probs"].tolist(),
                    "chosen_id": chosen_id,
                    "gap": info["gap"],
                    "p1": info["p1"],
                })

            if self.debug and self._step < self.debug_max:
                mb_c = float(probs_i[cand_idx].sum().item())
                sv = float(vdgd_probs_i.sum().item())
                sm = float((mix_probs_i if mix_probs_i is not None else vdgd_probs_i).sum().item())
                print(f"[ECRD-DEBUG] step={self._step} k={k_i} mass_base(C)={mb_c:.6f} sum_vdgd={sv:.6f} sum_mix={sm:.6f}")
                print(f"[ECRD-DEBUG] top_base: {_fmt_top(idx_sorted, probs_i, self.debug_top, self.tokenizer)}")
                print(f"[ECRD-DEBUG] top_vdgd: {_fmt_top(torch.argsort(vdgd_probs_i, descending=True), vdgd_probs_i, self.debug_top, self.tokenizer)}")
                print(f"[ECRD-DEBUG] top_mix:  {_fmt_top(torch.argsort(eff_probs, descending=True), eff_probs, self.debug_top, self.tokenizer)}")

            self._step += 1
        return scores
