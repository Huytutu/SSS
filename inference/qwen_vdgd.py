from __future__ import annotations
from typing import Any, Optional

import copy

import torch
from qwen_vl_utils import process_vision_info
from transformers.generation import LogitsProcessorList

from SSS.fovea import TextScorer, VisionScorer, VDGDLogitsProcessor, GLOBAL_DESCRIPTION_PROMPT
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
    use_vision_grounding: bool = False,
    use_text_grounding: bool = True,
):
    """VDGD end-to-end for Qwen2.5-VL: generate the image description,
    concatenate it as a prefix to the prompt, prime the scorer on that
    description-prefixed prompt, and return a ready-to-use VDGDLogitsProcessor.

    Returns (proc, gen_config, augmented_question). The caller must generate
    against `augmented_question`, not the original `question` -- that's the
    "concatenate the generated description as a prefix" step from the paper.

    use_vision_grounding=True adds a ReVisiT-style vision-token grounding
    source (see fovea.VisionScorer) on top of VDGD's textual description
    grounding, sharing a single forward pass with the TextScorer.

    use_text_grounding=False skips the description step entirely (no
    generate_description call, no prompt prefix, no TextScorer) -- combined
    with use_vision_grounding=True this reduces to plain ReVisiT (base +
    vision only), matching the reference implementation's own algorithm
    with no VDGD text layer on top. At least one of use_text_grounding /
    use_vision_grounding must be True.
    """
    assert use_text_grounding or use_vision_grounding, (
        "build_vdgd_processor needs at least one grounding source (text and/or vision)."
    )

    if use_text_grounding:
        description = generate_description(model, processor, image, question, min_pixels, max_pixels)
        augmented_question = f"{description}\n\n{question}"
    else:
        description = ""
        augmented_question = question

    messages = build_messages(image, augmented_question, min_pixels, max_pixels)
    chat = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    img_inputs, vid_inputs = process_vision_info(messages)
    inputs = processor(text=[chat], images=img_inputs, videos=vid_inputs, padding=True, return_tensors="pt").to(model.device)

    scorer = TextScorer(model=model, tokenizer=processor.tokenizer) if use_text_grounding else None
    vision_scorer = None
    if use_vision_grounding:
        image_token_id = int(model.config.image_token_id)
        merge_size = int(model.config.vision_config.spatial_merge_size)
        expected_vis_tokens = int((inputs["image_grid_thw"].prod(dim=-1) / (merge_size ** 2)).sum().item())
        actual_vis_tokens = int((inputs["input_ids"][0] == image_token_id).sum().item())
        assert actual_vis_tokens == expected_vis_tokens, (
            f"image_token_id={image_token_id} matched {actual_vis_tokens} positions in input_ids, "
            f"but image_grid_thw implies {expected_vis_tokens} vision tokens -- image_token_id is "
            f"likely wrong for this model."
        )

        with torch.inference_mode():
            out = model(**inputs, use_cache=False, output_hidden_states=True)
        if scorer is not None:
            scorer._cache_from_logits(out.logits)

        vision_scorer = VisionScorer(model=model, image_token_id=image_token_id)
        vision_scorer.set_prompt_from_hidden_states(out.hidden_states, inputs["input_ids"], inputs["image_grid_thw"])
    else:
        scorer.set_prompt(**inputs)

    proc = VDGDLogitsProcessor(
        scorer=scorer, tokenizer=processor.tokenizer, min_k=min_k, max_k=max_k, vision_scorer=vision_scorer
    )

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
    use_vision_grounding: bool = False,
    use_text_grounding: bool = True,
    return_processor: bool = False,
):
    """Same shape as qwen_base.qwen_base: load-if-needed, generate, decode.
    Pass an already-loaded model/processor to reuse them for a fair,
    single-load comparison against qwen_base. Returns (answer, augmented_question).

    use_text_grounding=False with use_vision_grounding=True is plain ReVisiT
    (no description, no prompt prefix) -- see build_vdgd_processor.

    return_processor=True additionally returns (proc, gen_ids) -- the
    VDGDLogitsProcessor instance (whose `.vision_trace` records each
    decoding step's JSD-selected vision token when use_vision_grounding is
    on) and the raw generated token ids, for callers that want to inspect
    the decoding process rather than just its text output."""
    if model is None or processor is None:
        model, processor = load_qwen(model_path)

    proc, gen_config, augmented_question, description = build_vdgd_processor(
        model, processor, image, question, min_pixels, max_pixels, min_k, max_k,
        use_vision_grounding=use_vision_grounding, use_text_grounding=use_text_grounding,
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
    gen_ids = out[:, inputs.input_ids.shape[1]:]
    answer = processor.batch_decode(
        gen_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0].strip()
    if return_processor:
        return answer, description, proc, gen_ids[0]
    return answer, description
