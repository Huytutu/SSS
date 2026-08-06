from __future__ import annotations
from typing import Any, Callable, Dict, List, Optional

import os
import re

import torch

__all__ = ["find_earliest_divergence", "default_extract_answer"]

_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", flags=re.S)


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return int(value) if value is not None else int(default)


def default_extract_answer(text: str) -> str:
    """Leading option letter out of <answer>...</answer>, matching the
    convention eval/eval_treebench_qwen.py already uses."""
    match = _ANSWER_RE.search(text or "")
    answer_text = match.group(1).strip() if match else (text or "").strip()
    letter = re.match(r"([A-Za-z])\b", answer_text)
    return letter.group(1).upper() if letter else answer_text.upper()[:1]


@torch.inference_mode()
def find_earliest_divergence(
    model,
    processor,
    out_ids: torch.Tensor,
    prompt_len: int,
    checkpoints: List[Dict[str, Any]],
    model_inputs: Dict[str, Any],
    max_tests: Optional[int] = None,
    cont_max_new_tokens: Optional[int] = None,
    extract_answer: Optional[Callable[[str], str]] = None,
) -> Optional[Dict[str, Any]]:
    """Find the earliest near-tie step whose runner-up token leads to a
    different final answer -- the rollback anchor for a rethink.

    Greedy decoding is deterministic given a prefix, so branch A (the original
    continuation) is already contained in `out_ids` and costs nothing; only
    branch B needs generating. Checkpoints are tried oldest-first and the
    search stops at the first disagreement, so a response whose reasoning went
    wrong early usually costs one continuation rather than `max_tests`.

    `model_inputs` must be the processor output that produced `out_ids`: its
    pixel_values/image_grid_thw are re-passed so the continuation still sees
    the image. Without them Qwen2.5-VL skips the vision tower entirely and the
    image placeholder tokens fall back to plain text embeddings, i.e. branch B
    would be reasoning blind.

    Returns the anchor dict, or None if no tested checkpoint diverged.
    """
    extract = extract_answer or default_extract_answer
    max_tests = _env_int("ECRD_BRANCH_MAX_TESTS", 3) if max_tests is None else int(max_tests)
    cont_tokens = (
        _env_int("ECRD_BRANCH_CONT_TOKENS", 250) if cont_max_new_tokens is None else int(cont_max_new_tokens)
    )
    if not checkpoints or max_tests <= 0:
        return None

    orig_text = processor.batch_decode(
        out_ids[:, prompt_len:], skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0].strip()
    orig_pred = extract(orig_text)

    vision_kwargs = {
        key: model_inputs[key]
        for key in ("pixel_values", "image_grid_thw")
        if model_inputs.get(key) is not None
    }

    tested = 0
    for checkpoint in sorted(checkpoints, key=lambda c: c["step"]):
        if tested >= max_tests:
            break
        step = int(checkpoint["step"])
        prefix = out_ids[:, : prompt_len + step]
        runner_up = torch.tensor([[int(checkpoint["rank1_id"])]], device=out_ids.device)
        forced = torch.cat([prefix, runner_up], dim=1)

        branch = model.generate(
            input_ids=forced,
            attention_mask=torch.ones_like(forced),
            do_sample=False,
            max_new_tokens=cont_tokens,
            use_cache=True,
            **vision_kwargs,
        )
        branch_text = processor.batch_decode(
            branch[:, forced.shape[1]:], skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()
        branch_pred = extract(branch_text)
        tested += 1

        if branch_pred != orig_pred:
            score_key = "entropy" if "entropy" in checkpoint else "gap"
            return {
                "step": step,
                score_key: float(checkpoint[score_key]),
                "orig_pred": orig_pred,
                "branch_pred": branch_pred,
                "n_tested": tested,
            }

    return None
