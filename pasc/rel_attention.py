from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
from qwen_vl_utils import process_vision_info

from .prompts import BASELINE_DESCRIPTION_PROMPT

__all__ = ["compute_baseline_row", "attention_map", "crop_bbox", "bbox_from_att_image_adaptive"]


@torch.inference_mode()
def compute_baseline_row(model, processor, image, probe, max_pixels=None):
    """Attention over image tokens under a question-agnostic prompt.

    This is the denominator of MLLMs Know Where to Look's relative attention
    (arXiv:2502.17422): dividing the real question's image attention by this
    cancels the question-independent bias -- border patches, high-frequency
    texture -- that otherwise dominates a raw attention map.

    One forward pass per image, not per token. Runs through `probe`, so the
    attention comes back without output_attentions=True.
    """
    image_content = {"type": "image", "image": image}
    if max_pixels is not None:
        image_content["max_pixels"] = max_pixels
    messages = [{"role": "user", "content": [image_content, {"type": "text", "text": BASELINE_DESCRIPTION_PROMPT}]}]
    chat = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    imgs, vids = process_vision_info(messages)
    inputs = processor(text=[chat], images=imgs, videos=vids, padding=True, return_tensors="pt").to(model.device)

    from .attn_probe import find_image_span
    span = find_image_span(inputs["input_ids"], model.config.image_token_id)
    probe.set_context(prompt_len=int(inputs["input_ids"].shape[1]), img_span=span)
    model(**inputs, use_cache=True)
    return probe.pop_step().img_row


def attention_map(img_row, baseline_row, image_grid_thw, floor_frac=0.25):
    """Relative attention as a 2D patch grid.

    `image_grid_thw` counts raw 14px patches; Qwen2.5-VL merges each 2x2 block
    into one image token, so the token grid is half that in each spatial
    dimension -- the same `/ 2` mllms_know's rel_attention_qwen2_5 applies.

    `floor_frac` clamps the denominator from below, at that fraction of its own
    mean. mllms_know divides straight through, which works in their setting
    because numerator and denominator are read at the same query position. We
    read the numerator mid-generation, and the mismatch lets patches with almost
    no baseline attention -- sky, blank wall -- produce enormous ratios and win
    the crop. See PASConfig.baseline_floor_frac for the measurements.
    """
    base = baseline_row.float()
    base = torch.clamp(base, min=float(base.mean()) * floor_frac)
    ratio = (img_row.float() / base).cpu().numpy()
    h, w = (int(image_grid_thw[0, 1]) // 2, int(image_grid_thw[0, 2]) // 2)
    if ratio.size != h * w:
        raise ValueError(f"attention row has {ratio.size} entries but grid is {h}x{w}={h*w}.")
    return ratio.reshape(h, w)


def crop_bbox(img_row, baseline_row, image_grid_thw, image_size, bbox_size, floor_frac=0.25):
    """(x1, y1, x2, y2) in original-image pixels for the region to zoom into."""
    att_map = attention_map(img_row, baseline_row, image_grid_thw, floor_frac)
    return bbox_from_att_image_adaptive(att_map, image_size, bbox_size)


# ---------------------------------------------------------------------------
# Vendored verbatim (modulo formatting) from MLLMs Know Where to Look,
# https://github.com/saccharomycetes/mllms_know, utils.py, MIT licensed.
# Kept as-is rather than rewritten: the ratio-by-sharpness selection below is
# subtle enough that a reimplementation would silently diverge from the
# published method.
# ---------------------------------------------------------------------------
def bbox_from_att_image_adaptive(att_map, image_size, bbox_size=336):
    """Pick a crop around the attention peak, choosing the crop scale adaptively.

    Slides a window over `att_map` at several scales, takes the position with
    the most attention, then keeps the scale whose peak stands out most sharply
    from its neighbours -- so a tight, confident blob yields a tight crop and a
    diffuse one yields a wide crop.
    """
    ratios = [1, 1.2, 1.4, 1.6, 1.8, 2]

    max_att_poses = []
    differences = []
    block_nums = []

    for ratio in ratios:
        block_size = image_size[0] / att_map.shape[1], image_size[1] / att_map.shape[0]
        block_num = (min(int(bbox_size * ratio / block_size[0]), att_map.shape[1]),
                     min(int(bbox_size * ratio / block_size[1]), att_map.shape[0]))
        if att_map.shape[1] - block_num[0] < 1 and att_map.shape[0] - block_num[1] < 1:
            if ratio == 1:
                return 0, 0, image_size[0], image_size[1]
            continue
        block_nums.append((block_num[0], block_num[1]))

        sliding_att = np.zeros((att_map.shape[0] - block_num[1] + 1, att_map.shape[1] - block_num[0] + 1))
        max_att = -np.inf
        max_att_pos = (0, 0)
        for x in range(att_map.shape[1] - block_num[0] + 1):
            for y in range(att_map.shape[0] - block_num[1] + 1):
                att = att_map[y:y + block_num[1], x:x + block_num[0]].sum()
                sliding_att[y, x] = att
                if att > max_att:
                    max_att = att
                    max_att_pos = (x, y)

        adjcent_atts = []
        if max_att_pos[0] > 0:
            adjcent_atts.append(sliding_att[max_att_pos[1], max_att_pos[0] - 1])
        if max_att_pos[0] < sliding_att.shape[1] - 1:
            adjcent_atts.append(sliding_att[max_att_pos[1], max_att_pos[0] + 1])
        if max_att_pos[1] > 0:
            adjcent_atts.append(sliding_att[max_att_pos[1] - 1, max_att_pos[0]])
        if max_att_pos[1] < sliding_att.shape[0] - 1:
            adjcent_atts.append(sliding_att[max_att_pos[1] + 1, max_att_pos[0]])
        differences.append((max_att - np.mean(adjcent_atts)) / (block_num[0] * block_num[1]))
        max_att_poses.append(max_att_pos)

    best = int(np.argmax(differences))
    max_att_pos = max_att_poses[best]
    block_num = block_nums[best]
    selected_bbox_size = bbox_size * ratios[best]
    block_size = image_size[0] / att_map.shape[1], image_size[1] / att_map.shape[0]

    x_center = int(max_att_pos[0] * block_size[0] + block_size[0] * block_num[0] / 2)
    y_center = int(max_att_pos[1] * block_size[1] + block_size[1] * block_num[1] / 2)
    x_center = max(x_center, selected_bbox_size // 2)
    y_center = max(y_center, selected_bbox_size // 2)
    x_center = min(x_center, image_size[0] - selected_bbox_size // 2)
    y_center = min(y_center, image_size[1] - selected_bbox_size // 2)

    x1 = max(0, x_center - selected_bbox_size // 2)
    y1 = max(0, y_center - selected_bbox_size // 2)
    x2 = min(image_size[0], x_center + selected_bbox_size // 2)
    y2 = min(image_size[1], y_center + selected_bbox_size // 2)
    return x1, y1, x2, y2
