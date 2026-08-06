from __future__ import annotations
from typing import Any, Callable, Dict, Optional

import torch
from transformers.generation import LogitsProcessorList

from .branch_probe import default_extract_answer, find_earliest_divergence
from .evidence import Evidence

__all__ = ["ecrd_generate_with_rethink"]

# Bridges GRIT's rethink back into the main model's own voice. The main model
# never saw GRIT's <rethink> tags in training, so the note is plain prose.
RETHINK_SPLICE = "\n\nWait -- reconsidering: {rethink}\n\n"


@torch.inference_mode()
def ecrd_generate_with_rethink(
    model,
    processor,
    image: Any,
    question: str,
    proc,
    inputs: Dict[str, Any],
    grit_client=None,
    max_new_tokens: int = 512,
    use_reasoning_rethink: bool = False,
    evidence_pool=None,
    extract_answer: Optional[Callable[[str], str]] = None,
) -> Dict[str, Any]:
    """Run ECRD, then optionally repair one reasoning segment via GRIT.

    Pass 1 is the existing ECRD pipeline, untouched: `proc`'s token-level GRIT
    hook still fires throughout. The addition is what happens afterwards --
    a near-tie step whose runner-up flips the final answer marks a genuine
    fork in the reasoning, so we roll back there and let GRIT reconsider the
    whole segment rather than a single token.

    `proc` must be built with collect_branch_checkpoints=True for the repair
    path to have anything to look at.
    """
    extract = extract_answer or default_extract_answer
    prompt_len = int(inputs["input_ids"].shape[1])

    out = model.generate(
        **inputs,
        do_sample=False,
        use_cache=True,
        max_new_tokens=max_new_tokens,
        logits_processor=LogitsProcessorList([proc]),
    )
    answer_text = processor.batch_decode(
        out[:, prompt_len:], skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0].strip()

    result = {
        "answer": extract(answer_text),
        "answer_text": answer_text,
        "rethink_applied": False,
        "anchor": None,
        "rethink_text": None,
        "original_answer": extract(answer_text),
        "original_answer_text": answer_text,
    }
    if not use_reasoning_rethink or grit_client is None:
        return result

    checkpoints = getattr(proc, "branch_checkpoints", None) or []
    anchor = find_earliest_divergence(
        model, processor, out, prompt_len, checkpoints, inputs, extract_answer=extract
    )
    result["anchor"] = anchor
    if anchor is None:
        return result

    prefix_text = processor.batch_decode(
        out[:, prompt_len: prompt_len + anchor["step"]],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    rethink = grit_client.rethink_segment(image=image, question=question, prefix_text=prefix_text)
    rethink_text = (rethink.get("rethink_text") or "").strip()
    result["rethink_text"] = rethink_text
    if not rethink_text:
        return result

    # GRIT's own <answer> phrasing is not guaranteed to match the caller's
    # option-letter convention, so splice only its reasoning and let the main
    # model finish in the format the rest of the pipeline already parses.
    note_ids = processor.tokenizer(
        RETHINK_SPLICE.format(rethink=rethink_text), add_special_tokens=False, return_tensors="pt"
    ).input_ids.to(out.device)
    spliced = torch.cat([out[:, : prompt_len + anchor["step"]], note_ids], dim=1)
    vision_kwargs = {
        key: inputs[key] for key in ("pixel_values", "image_grid_thw") if inputs.get(key) is not None
    }
    repaired = model.generate(
        input_ids=spliced,
        attention_mask=torch.ones_like(spliced),
        do_sample=False,
        use_cache=True,
        max_new_tokens=max_new_tokens,
        **vision_kwargs,
    )
    repaired_text = processor.batch_decode(
        repaired[:, prompt_len:], skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0].strip()

    result["rethink_applied"] = True
    result["answer"] = extract(repaired_text)
    result["answer_text"] = repaired_text

    if evidence_pool is not None and hasattr(evidence_pool, "add_evidence"):
        evidence_pool.add_evidence(Evidence(
            id=f"grit-rethink-{anchor['step']}",
            text=rethink_text,
            source="grit-rethink",
            time_step=anchor["step"],
            bbox=None,
        ))
    return result
