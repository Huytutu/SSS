from __future__ import annotations
from typing import Any, Optional

import copy

import torch
from qwen_vl_utils import process_vision_info
from transformers.generation import LogitsProcessorList

from SSS.fovea import PromptDeviationScorer, VDGDLogitsProcessor, GLOBAL_DESCRIPTION_PROMPT, ONE_SHOT_REASONING_EXAMPLE
from .qwen_base import DEFAULT_MODEL_PATH, load_qwen, build_messages

__all__ = ["build_messages", "generate_description", "build_vdgd_processor", "qwen_vdgd"]


def _build_description_messages(image: Any, text: str, min_pixels: Optional[int], max_pixels: Optional[int]):
    # Plain image+text, no "Think step by step / <answer>" suffix -- that
    # suffix belongs to the final-answer prompt (qwen_base.build_messages),
    # not to the "describe the image" preamble.
    image_content = {"type": "image", "image": image}
    if min_pixels is not None:
        image_content["min_pixels"] = min_pixels
    if max_pixels is not None:
        image_content["max_pixels"] = max_pixels
    return [{"role": "user", "content": [image_content, {"type": "text", "text": text}]}]


@torch.inference_mode()
def generate_description(
    model,
    processor,
    image: Any,
    question: str,
    min_pixels: Optional[int] = None,
    max_pixels: Optional[int] = None,
    max_new_tokens: int = 256,
) -> str:
    """VDGD preamble, step 1: describe the image."""
    prompt = GLOBAL_DESCRIPTION_PROMPT.format(instruction=question)
    messages = _build_description_messages(image, prompt, min_pixels, max_pixels)
    chat = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    img_inputs, vid_inputs = process_vision_info(messages)
    inputs = processor(text=[chat], images=img_inputs, videos=vid_inputs, padding=True, return_tensors="pt").to(model.device)
    out = model.generate(**inputs, do_sample=False, max_new_tokens=max_new_tokens, use_cache=True)
    return processor.batch_decode(
        out[:, inputs.input_ids.shape[1]:], skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0].strip()


def build_vdgd_processor(
    model,
    processor,
    image: Any,
    question: str,
    min_pixels: Optional[int] = None,
    max_pixels: Optional[int] = None,
    min_k: Optional[int] = None,
    max_k: Optional[int] = None,
    one_shot: bool = False,
):
    """VDGD end-to-end for Qwen2.5-VL: generate the image description,
    concatenate it as a prefix to the prompt, prime the scorer on that
    description-prefixed prompt, and return a ready-to-use VDGDLogitsProcessor.

    Returns (proc, gen_config, augmented_question). The caller must generate
    against `augmented_question`, not the original `question` -- that's the
    "concatenate the generated description as a prefix" step from the paper.

    `one_shot`, if True, inserts ONE_SHOT_REASONING_EXAMPLE between the
    description and the real question -- see that constant's docstring in
    fovea/prompts.py. Only affects the final answer-generation prompt; the
    description step above still runs on the plain `question`, so the
    "describe the image" instruction isn't polluted by the example. Off by
    default: validated on a single example so far, not a general default.
    """
    description = generate_description(model, processor, image, question, min_pixels, max_pixels)
    one_shot_prefix = ONE_SHOT_REASONING_EXAMPLE if one_shot else ""
    augmented_question = f"{description}\n\n{one_shot_prefix}{question}"

    messages = build_messages(image, augmented_question, min_pixels, max_pixels)
    chat = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    img_inputs, vid_inputs = process_vision_info(messages)
    inputs = processor(text=[chat], images=img_inputs, videos=vid_inputs, padding=True, return_tensors="pt").to(model.device)

    scorer = PromptDeviationScorer(model=model, tokenizer=processor.tokenizer)
    scorer.set_prompt(**inputs)

    proc = VDGDLogitsProcessor(scorer=scorer, tokenizer=processor.tokenizer, min_k=min_k, max_k=max_k)

    gen_config = copy.copy(model.generation_config)
    gen_config.do_sample = False
    gen_config.top_p = None
    gen_config.top_k = None
    gen_config.max_new_token = 512
    return proc, gen_config, augmented_question, description


def qwen_vdgd(
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
    one_shot: bool = False,
):
    """Same shape as qwen_base.qwen_base: load-if-needed, generate, decode.
    Pass an already-loaded model/processor to reuse them for a fair,
    single-load comparison against qwen_base. Returns (answer, augmented_question)."""
    if model is None or processor is None:
        model, processor = load_qwen(model_path)

    proc, gen_config, augmented_question, description = build_vdgd_processor(
        model, processor, image, question, min_pixels, max_pixels, min_k, max_k, one_shot=one_shot
    )

    messages = build_messages(image, augmented_question, min_pixels, max_pixels)
    chat = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    img_inputs, vid_inputs = process_vision_info(messages)
    inputs = processor(text=[chat], images=img_inputs, videos=vid_inputs, padding=True, return_tensors="pt").to(model.device)

    out = model.generate(
        **inputs,
        generation_config=gen_config,
        max_new_tokens=max_new_tokens,
        logits_processor=LogitsProcessorList([proc]),
    )
    answer = processor.batch_decode(
        out[:, inputs.input_ids.shape[1]:], skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0].strip()
    return answer, description
