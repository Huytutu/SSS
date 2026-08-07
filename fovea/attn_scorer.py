from __future__ import annotations
from typing import Optional, Sequence, Tuple

import torch

__all__ = ["find_image_token_span", "token_attn_cost"]


def find_image_token_span(input_ids: torch.Tensor, image_token_id: int) -> Tuple[int, int]:
    """Locate the contiguous run of `image_token_id` in `input_ids[0]`.

    Returns (start, end) such that input_ids[0, start:end] are all image
    tokens. Qwen2.5-VL places every image's pad tokens in one contiguous run
    between <|vision_start|>/<|vision_end|>, so non-contiguity here means a
    multi-image prompt -- not supported by this helper."""
    ids = input_ids[0]
    matches = (ids == image_token_id).nonzero(as_tuple=True)[0]
    if matches.numel() == 0:
        raise ValueError(f"No image_token_id={image_token_id} found in input_ids.")
    start, end = int(matches[0].item()), int(matches[-1].item()) + 1
    if end - start != matches.numel():
        raise ValueError("Image tokens are not contiguous -- multi-image prompts aren't supported.")
    return start, end


def token_attn_cost(
    attn_step: Sequence[torch.Tensor],
    img_span: Tuple[int, int],
    layer_range: Optional[Tuple[int, int]] = None,
    eps: float = 1e-8,
) -> float:
    """attn_cost_t = D_kl_t (AFIP eq.3, arXiv:2605.24602) for one decode step:
    mean cross-head KL divergence between each head's image-token attention
    distribution and the collective (head-averaged) distribution -- "how much
    do the heads disagree about where in the image to look".

    Superseded VAR (eq.2, "how much attention landed on the image") as the
    signal used here: VAR is reported by the paper to decline monotonically
    over decode position across model backbones, which is a *when* effect
    that swamps within-generation z-scoring (see fovea/leco.py's
    `_detrend_step_costs`, and the discussion in score_step's `attn_score`
    docstring) and is disconnected from *correctness* -- a legitimate late
    step (e.g. a closing "therefore the answer is X") has low VAR without
    hallucinating. D_kl has no such reported temporal trend, and being a
    per-position head-disagreement measure it is closer in kind to `dev_cost`
    (a spiky, token-localized signal) than VAR (a slow, position-correlated
    one) -- a better match for the quantile-dominance aggregation this file
    already uses for dev_cost (one bad token should dominate a step of good
    ones).

    `attn_step` is one entry of `out.attentions` from
    `model.generate(output_attentions=True, return_dict_in_generate=True)`: a
    per-layer tuple of [batch, heads, q_len, kv_len] post-softmax attention
    tensors. Only the last query position (the one that produced this step's
    token) and the image-token columns `img_span` are used. Averaged over all
    heads and all layers in `layer_range` (default: every layer, matching
    AFIP's own default full-stack range).

    Per layer: each head's raw attention row over the image span is
    Laplace-smoothed into a proper distribution `p_head`; the collective
    distribution `p_collective` is the *normalized head-averaged raw row*
    (matching eq.1's Ā_t^l, not a plain average of already-normalized
    per-head distributions -- so a head that barely attends to the image
    contributes little to the collective, same as it contributes little raw
    mass to Ā_t^l)."""
    s, e = img_span
    n = e - s
    lo, hi = layer_range if layer_range is not None else (0, len(attn_step) - 1)
    layer_kls = []
    for l in range(lo, hi + 1):
        row = attn_step[l][0, :, -1, s:e]  # [heads, n_image_tokens], raw post-softmax attention
        p_head = (row + eps) / (row.sum(dim=-1, keepdim=True) + n * eps)  # [heads, n]
        row_avg = row.mean(dim=0)  # [n] -- Ā_t^l, head-averaged raw attention (eq.1's definition)
        p_collective = (row_avg + eps) / (row_avg.sum() + n * eps)  # [n]
        kl_per_head = (p_head * (p_head.log() - p_collective.log())).sum(dim=-1)  # [heads], eq.3's KL(P_h || P_l)
        layer_kls.append(kl_per_head.mean())  # (1/H) sum_h KL(...)
    return float(torch.stack(layer_kls).mean().item())
