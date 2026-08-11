from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
from qwen_vl_utils import process_vision_info

from .attn_probe import AttentionProbe, find_image_span
from .config import PASConfig
from .logits_processor import PASCLogitsProcessor
from .rel_attention import compute_baseline_row
from .scorer import EvidenceScorer
from .self_correct import SelfCorrector
from .trigger import PASTrigger

__all__ = ["pasc_generate"]


@torch.inference_mode()
def pasc_generate(
    model,
    processor,
    image,
    messages: List[Dict[str, Any]],
    cfg: Optional[PASConfig] = None,
    max_new_tokens: int = 512,
    prefill: str = "",
    max_pixels: Optional[int] = None,
    correct: bool = True,
) -> Tuple[str, PASCLogitsProcessor]:
    """Generate an answer with PAS-gated self-correction.

    `messages` is built by the caller so benchmarks can keep their own system
    prompt and phrasing; `prefill` is appended to the chat template to force the
    assistant to open a given way (TreeBench prefills "<think>").

    Returns the decoded answer and the processor, whose `correction_log` and
    `step_log` hold everything that happened.

    Set `correct=False` for the ablation that measures and logs but never crops.
    """
    cfg = cfg or PASConfig()

    chat = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) + prefill
    imgs, vids = process_vision_info(messages)
    inputs = processor(text=[chat], images=imgs, videos=vids, padding=True,
                       return_tensors="pt").to(model.device)

    probe = AttentionProbe(model, cfg)
    try:
        # Baseline attention first: it needs the probe on a different prompt, and
        # set_context below then switches the probe over to the real one.
        baseline_row = compute_baseline_row(model, processor, image, probe, max_pixels=max_pixels)

        prompt_len = int(inputs["input_ids"].shape[1])
        probe.set_context(prompt_len, find_image_span(inputs["input_ids"], model.config.image_token_id))

        scorer = EvidenceScorer(model, processor.tokenizer, cfg)
        corrector = None
        if correct:
            corrector = SelfCorrector(model, processor, cfg, image, baseline_row,
                                      inputs["image_grid_thw"], max_pixels=max_pixels)
        proc = PASCLogitsProcessor(model, processor, cfg, probe, scorer, PASTrigger(cfg), corrector)

        out = model.generate(**inputs, do_sample=False, use_cache=True,
                             max_new_tokens=max_new_tokens, logits_processor=[proc])
        text = processor.batch_decode(out[:, prompt_len:], skip_special_tokens=True,
                                      clean_up_tokenization_spaces=False)[0].strip()
        return text, proc
    finally:
        probe.restore()
