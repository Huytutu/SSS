from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

import torch
from qwen_vl_utils import process_vision_info

from .prompts import CORRECT_PROMPT_TEMPLATE
from .rel_attention import crop_bbox

__all__ = ["SelfCorrector"]

_ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.S | re.I)
_EVIDENCE_RE = re.compile(r"<evidence>\s*(.*?)\s*</evidence>", re.S | re.I)


class SelfCorrector:
    """Re-asks the base model about one uncertain token, zoomed in.

    When PAS flags a token, the attention map says which part of the image the
    model was (weakly) looking at. This crops that region, shows the model the
    full image and the crop together, and asks it to pick among the candidates
    it was already considering and to state one visual fact backing the choice.

    Two things this is not:
      * not a second model -- it reuses the loaded base model, unlike the GRIT
        client this replaces, which loaded a separate fine-tuned 3B;
      * not a free-form rewrite -- the answer is constrained to the candidate
        set, so a correction can only reorder what the model already proposed.

    The evidence sentence is the durable part: it goes into the evidence bank
    and keeps influencing later tokens, so one crop can pay off more than once.
    """

    def __init__(self, model, processor, cfg, image, baseline_row, image_grid_thw, max_pixels=None):
        self.model = model
        self.processor = processor
        self.cfg = cfg
        self.image = image
        self.baseline_row = baseline_row
        self.image_grid_thw = image_grid_thw
        self.max_pixels = max_pixels

    @torch.inference_mode()
    def correct(self, img_row, prefix_text: str, candidates: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Returns {choice_id, choice_text, evidence, bbox}, or None if the model
        didn't answer in the requested format."""
        bbox = crop_bbox(img_row, self.baseline_row, self.image_grid_thw, self.image.size,
                         self.cfg.bbox_size, self.cfg.baseline_floor_frac)
        crop = self.image.crop(bbox)

        lines = "\n".join(f'{i}: "{c["text"]}"' for i, c in enumerate(candidates))
        prompt = CORRECT_PROMPT_TEMPLATE.format(prefix=prefix_text[-400:], candidates=lines)

        content = [self._image_part(self.image), self._image_part(crop), {"type": "text", "text": prompt}]
        messages = [{"role": "user", "content": content}]
        chat = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        imgs, vids = process_vision_info(messages)
        inputs = self.processor(text=[chat], images=imgs, videos=vids, padding=True,
                                return_tensors="pt").to(self.model.device)

        out = self.model.generate(**inputs, do_sample=False, use_cache=True,
                                  max_new_tokens=self.cfg.correct_max_new_tokens)
        text = self.processor.batch_decode(out[:, inputs["input_ids"].shape[1]:],
                                           skip_special_tokens=True)[0]

        index = self._parse_index(text, len(candidates))
        if index is None:
            return None
        return {
            "choice_id": int(candidates[index]["id"]),
            "choice_text": str(candidates[index]["text"]),
            "evidence": self._parse_evidence(text),
            "bbox": tuple(int(v) for v in bbox),
        }

    def _image_part(self, image):
        part = {"type": "image", "image": image}
        if self.max_pixels is not None:
            part["max_pixels"] = self.max_pixels
        return part

    @staticmethod
    def _parse_index(text: str, n_candidates: int) -> Optional[int]:
        match = _ANSWER_RE.search(text)
        if not match:
            return None
        digits = re.search(r"\d+", match.group(1))
        if not digits:
            return None
        index = int(digits.group())
        return index if 0 <= index < n_candidates else None

    @staticmethod
    def _parse_evidence(text: str) -> str:
        match = _EVIDENCE_RE.search(text)
        if not match:
            return ""
        evidence = " ".join(match.group(1).split())
        return "" if evidence.lower() == "none" else evidence
