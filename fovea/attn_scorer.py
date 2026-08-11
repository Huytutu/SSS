from __future__ import annotations
from typing import Tuple

import torch

__all__ = ["find_image_token_span"]


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
