from __future__ import annotations
from typing import List, Optional, Any, Dict
import math
import torch
import torch.nn.functional as F
from .evidence import Evidence

__all__ = ["EvidenceScorer"]


def _ensure_left_pad(tokenizer):
    if hasattr(tokenizer, "padding_side") and tokenizer.padding_side != "left":
        tokenizer.padding_side = "left"
    if getattr(tokenizer, "pad_token_id", None) is None and getattr(tokenizer, "eos_token_id", None) is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id


class EvidenceScorer:
    """VDGD-style textual evidence scorer used by ECRD.

    Each evidence sentence is precomputed into prefix next-token log-probabilities.
    At inference, score_token_ids returns one consistency cost per candidate token.
    The implementation follows the paper logic: mean-over-prefix support, smoothed
    token probability, and soft-min aggregation over evidence items.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: Any,
        agg_tau: float = 1.0,
        max_prefix_len: int = 128,
        device: Optional[str] = None,
        backend_model: Optional[torch.nn.Module] = None,
    ) -> None:
        self.model = model
        self.backend_model = backend_model or model
        self.tokenizer = tokenizer if (hasattr(tokenizer, "encode") or hasattr(tokenizer, "__call__")) else tokenizer.tokenizer
        _ensure_left_pad(self.tokenizer)
        self.agg_tau = float(agg_tau)
        self.max_prefix_len = int(max_prefix_len)
        if device is None:
            try:
                self.device = str(next(self.backend_model.parameters()).device)
            except Exception:
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        try:
            self.model_dtype = next(self.backend_model.parameters()).dtype
        except Exception:
            self.model_dtype = torch.bfloat16
        self._bank: List[Dict[str, torch.Tensor]] = []
        self._vocab_size: Optional[int] = None
        self._evidences: List[Evidence] = []

    @torch.inference_mode()
    def add_evidence(self, ev: Evidence) -> None:
        text = (ev.text or "").strip()
        if not text:
            return
        tok = self.tokenizer(text=[text], return_tensors="pt", padding=False, add_special_tokens=True)
        input_ids = tok["input_ids"].to(self.device)
        attn_mask = tok.get("attention_mask")
        if attn_mask is not None:
            attn_mask = attn_mask.to(self.device)

        model = self.backend_model
        orig_use_cache = getattr(model.generation_config, "use_cache", None)
        try:
            if orig_use_cache is not None:
                model.generation_config.use_cache = False
            out = model(input_ids=input_ids, attention_mask=attn_mask, use_cache=False)
        finally:
            if orig_use_cache is not None:
                model.generation_config.use_cache = orig_use_cache

        logits = out.logits
        L = int(logits.shape[1])
        V = int(logits.shape[-1])
        if self._vocab_size is None:
            self._vocab_size = V
        L_eff = min(L, self.max_prefix_len)
        logits = logits[:, :L_eff, :]
        logp = F.log_softmax(logits.to(dtype=torch.float32), dim=-1)
        nll = (-logp).squeeze(0).to("cpu").to(torch.float16)
        self._bank.append({"id": ev.id, "nll": nll})
        self._evidences.append(ev)

    def get_evidences(self) -> List[Evidence]:
        return list(self._evidences)

    def num_evidences(self) -> int:
        return len(self._bank)

    @torch.no_grad()
    def score_token_ids(self, prefix_ids: torch.Tensor, cand_idx: torch.Tensor) -> torch.Tensor:
        if cand_idx.dim() == 0:
            cand_idx = cand_idx.unsqueeze(0)
        cand_idx = cand_idx.to(dtype=torch.long)
        k = int(cand_idx.numel())
        if k == 0:
            return torch.empty(0, device=prefix_ids.device, dtype=torch.float32)
        if not self._bank:
            return torch.zeros(k, device=prefix_ids.device, dtype=torch.float32)

        V = int(self._vocab_size or 0)
        eps = 1e-6 if V <= 0 else (1e-6 / V)
        cand_cpu = cand_idx.detach().to("cpu")
        per_e_scores: List[torch.Tensor] = []
        for entry in self._bank:
            nll = entry["nll"]
            L_eff = int(nll.shape[0])
            nll_Lk = nll[:, cand_cpu].to(torch.float32)
            logq_Lk = -nll_Lk
            log_mean_q = torch.logsumexp(logq_Lk, dim=0) - math.log(max(L_eff, 1))
            a = log_mean_q + math.log(1.0 - eps)
            b = math.log(eps) if V > 0 else -1e9
            log_q_smooth = torch.logsumexp(torch.stack([a, torch.full_like(a, b)]), dim=0)
            per_e_scores.append(-log_q_smooth)

        E = len(per_e_scores)
        S = torch.stack(per_e_scores, dim=0)
        tau = max(self.agg_tau, 1e-6)
        log_mean = torch.logsumexp(-S / tau, dim=0) - math.log(E)
        return (-tau * log_mean).to(device=prefix_ids.device, dtype=torch.float32)

    @torch.no_grad()
    def score_token_strs(self, token_strs: List[str]) -> torch.Tensor:
        enc = self.tokenizer(token_strs, add_special_tokens=False)["input_ids"]
        ids = []
        for item in enc:
            if isinstance(item, list) and len(item) == 1:
                ids.append(item[0])
        if not ids:
            return torch.tensor([], dtype=torch.float32)
        fake_prefix = torch.zeros(1, 1, dtype=torch.long, device=self.device)
        return self.score_token_ids(fake_prefix, torch.tensor(ids, dtype=torch.long))
