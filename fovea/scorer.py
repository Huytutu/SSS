from __future__ import annotations
from typing import Any, Optional

import torch
import torch.nn.functional as F

__all__ = ["TextScorer"]


class TextScorer:
    """VDGD prompt-deviation scorer (eq. 3).

    KL(one_hot(w_k) || p_VLM(.|x_<j)) reduces exactly to -log p_VLM(w_k|x_<j)
    (only the w_k term of the KL sum is nonzero for a one-hot reference), so:

        KL_{x_i,w_k} = min_j KL(one_hot(w_k) || p_VLM(.|x_<j))
                     = -max_j log p_VLM(w_k | x_<j)

    set_prompt() runs one forward pass over the (description-prefixed,
    image-conditioned) prompt and caches the per-position NLL over the full
    vocabulary; score_token_ids() is then a min-reduce lookup, no further
    forward passes needed during decoding.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: Any,
        device: Optional[str] = None,
        backend_model: Optional[torch.nn.Module] = None,
    ) -> None:
        self.model = model
        self.backend_model = backend_model or model
        self.tokenizer = tokenizer
        if device is None:
            try:
                self.device = str(next(self.backend_model.parameters()).device)
            except Exception:
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        # cached [n, V] per-position NLL over the prompt, fp16 on CPU. If this
        # becomes a memory problem on very long image-token prompts, the fix
        # is to truncate the cached window rather than lower precision further.
        self._neg_logp: Optional[torch.Tensor] = None
        self._vocab_size: Optional[int] = None

    @torch.inference_mode()
    def set_prompt(self, **model_inputs: Any) -> None:
        """model_inputs = full processor output (input_ids, pixel_values,
        image_grid_thw, attention_mask, ...) for the description-prefixed
        prompt. Runs one teacher-forced forward pass and caches p_VLM(.|x_<j)
        for every prompt position j via _cache_from_logits (see there for why
        logits[:, :-1, :] is exactly what eq. 3 needs)."""
        out = self.backend_model(**model_inputs, use_cache=False)
        self._cache_from_logits(out.logits)

    def _cache_from_logits(self, logits: torch.Tensor) -> None:
        """logits[:, t, :] predicts the token at position t+1, so
        logits[:, :-1, :] gives exactly the "predict the next prompt token"
        distributions eq. 3 needs (all j except j=1, the empty-prefix case,
        which is a single negligible row out of a prompt that is typically
        hundreds to thousands of tokens). Split out so a shared forward pass
        (e.g. also feeding VisionScorer) can call this directly instead of
        set_prompt running its own forward."""
        logits = logits[0, :-1, :]
        self._vocab_size = int(logits.shape[-1])
        logp = F.log_softmax(logits.to(torch.float32), dim=-1)
        self._neg_logp = (-logp).to(torch.float16).cpu()

    @torch.no_grad()
    def score_token_ids(self, cand_idx: torch.Tensor) -> torch.Tensor:
        """cost[k] = min_j NLL[j, cand_idx[k]]. Lower = some point in the
        prompt confidently anticipated this token (grounded); higher = no
        prompt position supports it (deviation / hallucination risk)."""
        if self._neg_logp is None:
            raise RuntimeError("TextScorer.set_prompt() must be called before scoring.")
        if cand_idx.dim() == 0:
            cand_idx = cand_idx.unsqueeze(0)
        cand_idx = cand_idx.to(dtype=torch.long, device="cpu")
        if cand_idx.numel() == 0:
            return torch.empty(0, dtype=torch.float32)
        return self._neg_logp[:, cand_idx].float().min(dim=0).values
