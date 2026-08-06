from __future__ import annotations
from typing import Any, Optional

from qwen_vl_utils import process_vision_info

from SSS.fovea.leco import leco_loop
from .qwen_vdgd import build_vdgd_processor
from .qwen_base import DEFAULT_MODEL_PATH, build_messages, load_qwen

__all__ = ["qwen_leco"]


def qwen_leco(
    image: Any,
    question: str,
    model=None,
    processor=None,
    model_path: str = DEFAULT_MODEL_PATH,
    min_pixels: Optional[int] = None,
    max_pixels: Optional[int] = None,
    min_k: Optional[int] = None,
    max_k: Optional[int] = None,
    max_new_tokens: int = 1024,
    max_iters: int = 3,
    one_shot: bool = False,
):
    """LeCo (arXiv:2403.19094) applied on top of VDGD: same describe -> prefix ->
    prime-scorer setup as qwen_vdgd, then an iterative loop that rolls back to the
    lowest-confidence reasoning step and regenerates -- still through the same
    VDGDLogitsProcessor each round -- until two consecutive answers agree or
    `max_iters` is reached. Returns (answer, description, n_iters).

    `one_shot`, if True, primes the answer-generation prompt with
    ONE_SHOT_REASONING_EXAMPLE -- see build_vdgd_processor's docstring."""
    if model is None or processor is None:
        model, processor = load_qwen(model_path)

    proc, gen_config, augmented_question, description = build_vdgd_processor(
        model, processor, image, question, min_pixels, max_pixels, min_k, max_k, one_shot=one_shot
    )
    proc.collect_step_log = True

    messages = build_messages(image, augmented_question, min_pixels, max_pixels)
    chat = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    img_inputs, vid_inputs = process_vision_info(messages)
    inputs = processor(text=[chat], images=img_inputs, videos=vid_inputs, padding=True, return_tensors="pt").to(model.device)

    answer, n_iters, _history = leco_loop(
        model, processor, proc, gen_config, inputs, max_new_tokens=max_new_tokens, max_iters=max_iters
    )
    return answer, description, n_iters
