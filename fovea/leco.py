from __future__ import annotations
from typing import Any, Callable, Dict, List, Optional, Tuple

import re

import torch
from transformers import LogitsProcessor
from transformers.generation import LogitsProcessorList

__all__ = [
    "segment_steps", "score_step", "find_lowest_scoring_step", "leco_loop",
    "NegativeGuidanceLogitsProcessor", "MTI_PAPER_MARKER",
]

_ANSWER_RE = re.compile(r"<answer>", flags=re.I)

# MTI-style contrastive marker: frames the discarded continuation as what NOT
# to repeat. Capped length (see leco_loop) since it's re-embedded in the
# negative-branch forward pass on every triggered token.
NEGATIVE_MARKER_TEMPLATE = (
    "\n(The following continuation was flagged as unreliable and should not be "
    "repeated: {cut_text}\nReconsider instead.)\n"
)
_DEFAULT_NEGATIVE_MARKER = "\n(The reasoning above was flagged as unreliable.)\n"

# The MTI paper's own marker: a short, content-agnostic negative prompt (fixed
# regardless of what was discarded), rather than one built from `cut_text`.
# Pass negative_marker="OUTPUT ERROR" to leco_loop to use this instead.
MTI_PAPER_MARKER = "OUTPUT ERROR"


def _default_extract_answer(text: str) -> str:
    match = re.search(r"<answer>(.*?)</answer>", text or "", flags=re.S | re.I)
    answer_text = match.group(1).strip() if match else (text or "").strip()
    letter = re.match(r"([A-Za-z])\b", answer_text)
    return letter.group(1).upper() if letter else answer_text.upper()[:1]


def segment_steps(token_ids: List[int], tokenizer, answer_start_idx: Optional[int]) -> List[Tuple[int, int]]:
    """Split a continuation's token ids into reasoning-step ranges (start, end).

    There is no numbered "Step N:" format in this pipeline's free-form CoT output,
    so steps are cut at newline tokens instead -- a step ends at (and includes) the
    first token whose decoded text contains "\\n". Tokens from `answer_start_idx`
    onward (the "<answer>...</answer>" span) are excluded: rolling back into the
    final answer isn't a meaningful reasoning rollback.
    """
    limit = len(token_ids) if answer_start_idx is None else min(answer_start_idx, len(token_ids))
    ranges: List[Tuple[int, int]] = []
    start = 0
    for i in range(limit):
        tok_text = tokenizer.decode([token_ids[i]], skip_special_tokens=True)
        if "\n" in tok_text:
            ranges.append((start, i + 1))
            start = i + 1
    if start < limit:
        ranges.append((start, limit))
    return ranges


def _kl_to_uniform(probs: torch.Tensor, tau: float) -> float:
    """KL(softmax(logits/tau) || uniform) over the candidate set -- how far the
    (temperature-rescaled) distribution is from spread-evenly, i.e. how peaked it is."""
    n = probs.numel()
    if n <= 1:
        return 0.0
    logits = torch.log(probs.clamp_min(1e-12)) / max(tau, 1e-6)
    p = torch.softmax(logits, dim=-1)
    uniform = 1.0 / n
    kl = (p * (torch.log(p.clamp_min(1e-12)) - torch.log(torch.tensor(uniform)))).sum()
    return float(kl.item())


def score_step(
    step_log_slice: List[Dict[str, Any]],
    next_step_head: List[Dict[str, Any]],
    K: int = 3,
    tau: float = 0.3,
) -> float:
    """LeCo step confidence: avg_score + trans_score - diver_score.

    avg_score: mean top1 (chosen-token) probability across the step's tokens.
    diver_score: mean KL(rescaled candidate-set probs || uniform) -- high when the
        candidate distribution stays spread out (uncertain) rather than peaked.
    trans_score: mean top1 probability over the first K tokens of the *next* step
        (heading tokens). If there is no next step, defaults to this step's own
        avg_score (neutral) rather than 0 -- 0 would structurally under-score every
        terminal step regardless of its actual confidence, biasing rollback toward
        the last step (typically a trivial "Therefore, the answer is:" transition
        in this pipeline's free-form CoT) instead of a genuine low-confidence one.
    """
    if not step_log_slice:
        return 0.0
    avg_score = sum(r["top1_prob"] for r in step_log_slice) / len(step_log_slice)
    diver_score = sum(_kl_to_uniform(r["cand_probs"], tau) for r in step_log_slice) / len(step_log_slice)
    head = next_step_head[:K]
    trans_score = sum(r["top1_prob"] for r in head) / len(head) if head else avg_score
    return avg_score + trans_score - diver_score


def find_lowest_scoring_step(scores: List[float]) -> int:
    return int(min(range(len(scores)), key=lambda i: scores[i]))


class NegativeGuidanceLogitsProcessor(LogitsProcessor):
    """MTI-style classifier-free guidance, applied only at high-entropy tokens.

    LeCo's plain rollback-and-regenerate calls model.generate() again on the
    unchanged truncated prefix with greedy decoding -- deterministic given the
    same weights and input, so it mostly just re-derives the same continuation
    it already discarded, rather than actually reconsidering. This steers the
    (already VDGD-rescored) distribution away from a negative-context branch
    -- the same prefix plus a short marker naming the discarded reasoning as
    unreliable -- only where the model is genuinely uncertain, giving the
    regeneration step actual force instead of being a no-op.

    negative_prefix_ids: token ids for the real prefix + the negative marker
        (built once per rollback, see leco_loop). vision_kwargs are re-passed
        to the negative branch's forward pass so it still sees the image.
    """

    def __init__(
        self,
        model,
        negative_prefix_ids: torch.Tensor,
        prefix_len: int,
        vision_kwargs: Dict[str, Any],
        guidance_scale: float = 0.5,
        entropy_thresh: float = 0.5,
    ):
        super().__init__()
        self.model = model
        self.negative_prefix_ids = negative_prefix_ids
        self.prefix_len = prefix_len
        self.vision_kwargs = vision_kwargs
        self.guidance_scale = guidance_scale
        self.entropy_thresh = entropy_thresh
        self.n_triggered = 0

    @torch.inference_mode()
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        pos_probs = torch.softmax(scores.to(torch.float32), dim=-1)
        entropy = float(-(pos_probs * torch.log(pos_probs.clamp_min(1e-12))).sum(dim=-1)[0].item())
        if entropy <= self.entropy_thresh:
            return scores

        self.n_triggered += 1
        generated_so_far = input_ids[:, self.prefix_len:]
        neg_input = torch.cat([self.negative_prefix_ids, generated_so_far], dim=1)
        neg_out = self.model(
            input_ids=neg_input, attention_mask=torch.ones_like(neg_input), use_cache=False, **self.vision_kwargs
        )
        neg_logp = torch.log_softmax(neg_out.logits[:, -1, :].to(torch.float32), dim=-1)
        pos_logp = torch.log_softmax(scores.to(torch.float32), dim=-1)

        cfg_logp = pos_logp + self.guidance_scale * (pos_logp - neg_logp)
        return cfg_logp.to(scores.dtype)


@torch.inference_mode()
def leco_loop(
    model,
    processor,
    proc,
    gen_config,
    inputs: Dict[str, Any],
    max_new_tokens: int = 1024,
    max_iters: int = 3,
    extract_answer: Optional[Callable[[str], str]] = None,
    use_negative_guidance: bool = False,
    guidance_scale: float = 0.5,
    guidance_entropy_thresh: float = 0.5,
    max_negative_marker_chars: int = 400,
    negative_marker: Optional[str] = None,
) -> Tuple[str, int, List[Dict[str, Any]]]:
    """Iteratively regenerate through `proc` (a VDGDLogitsProcessor with
    collect_step_log=True), rolling back to the lowest-confidence reasoning step
    each round, until two consecutive iterations agree or `max_iters` is hit.

    `proc`'s scorer stays primed on the same prompt for the whole loop -- only the
    prefix fed to `model.generate` shrinks/regrows -- so every regeneration pass is
    still VDGD-decoded, not vanilla greedy.

    `use_negative_guidance` (opt-in, off by default): after a rollback, regenerate
    with NegativeGuidanceLogitsProcessor contrasting against a negative marker,
    instead of plain greedy decoding on the unchanged prefix -- see that class's
    docstring for why plain regeneration tends to just reproduce what was
    discarded. By default the marker is built from the previous round's
    discarded `cut_text` (NEGATIVE_MARKER_TEMPLATE); pass `negative_marker`
    (e.g. `MTI_PAPER_MARKER`, i.e. "OUTPUT ERROR") to use a fixed,
    content-agnostic marker instead, matching the MTI paper's own choice.
    """
    extract = extract_answer or _default_extract_answer
    tokenizer = processor.tokenizer
    prompt_len = int(inputs["input_ids"].shape[1])
    vision_kwargs = {
        key: inputs[key] for key in ("pixel_values", "image_grid_thw") if inputs.get(key) is not None
    }

    prefix_ids = inputs["input_ids"]
    prev_answer: Optional[str] = None
    text = ""
    history: List[Dict[str, Any]] = []
    negative_prefix_ids: Optional[torch.Tensor] = None

    for it in range(max_iters):
        remaining_budget = max_new_tokens - (prefix_ids.shape[1] - prompt_len)
        if remaining_budget <= 0:
            break

        proc.step_log = []
        processors = [proc]
        cfg_proc: Optional[NegativeGuidanceLogitsProcessor] = None
        if use_negative_guidance and negative_prefix_ids is not None:
            cfg_proc = NegativeGuidanceLogitsProcessor(
                model, negative_prefix_ids, prefix_ids.shape[1], vision_kwargs,
                guidance_scale=guidance_scale, entropy_thresh=guidance_entropy_thresh,
            )
            processors.append(cfg_proc)

        out = model.generate(
            input_ids=prefix_ids,
            attention_mask=torch.ones_like(prefix_ids),
            generation_config=gen_config,
            max_new_tokens=remaining_budget,
            logits_processor=LogitsProcessorList(processors),
            **vision_kwargs,
        )
        cont_ids = out[0, prefix_ids.shape[1]:].tolist()
        text = processor.batch_decode(
            out[:, prompt_len:], skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()
        cont_text = processor.batch_decode(
            [cont_ids], skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        answer = extract(text)
        # `new_text` is what THIS iteration generated from its (possibly truncated)
        # prefix -- for iter > 0 that's exactly the replacement for the previous
        # iteration's `cut_text`, so the two can be shown side by side.
        record: Dict[str, Any] = {"iter": it, "answer": answer, "n_tokens": len(cont_ids), "new_text": cont_text}
        if cfg_proc is not None:
            record["cfg_triggers"] = cfg_proc.n_triggered
        history.append(record)

        if answer == prev_answer:
            break

        answer_match = _ANSWER_RE.search(cont_text)
        answer_start_idx = None
        if answer_match:
            # Token index of the first token whose decode reaches the "<answer>" tag.
            running = ""
            for idx, tid in enumerate(cont_ids):
                running += tokenizer.decode([tid], skip_special_tokens=True)
                if len(running) >= answer_match.start():
                    answer_start_idx = idx
                    break

        ranges = segment_steps(cont_ids, tokenizer, answer_start_idx)
        record["step_ranges"] = ranges
        if len(ranges) <= 1:
            break

        step_scores = [
            score_step(proc.step_log[a:b], proc.step_log[ranges[j + 1][0]:ranges[j + 1][0] + 3])
            if j + 1 < len(ranges) else score_step(proc.step_log[a:b], [])
            for j, (a, b) in enumerate(ranges)
        ]
        # Exclude the last step from candidacy: it's the segment right before
        # <answer>, typically a trivial "Therefore, the answer is:" transition
        # rather than reasoning -- rolling back into it discards nothing
        # substantive and just reproduces the same continuation deterministically.
        bad = find_lowest_scoring_step(step_scores[:-1])
        record["step_scores"] = step_scores
        record["rollback_step"] = bad
        if bad == 0:
            break

        cut_start = ranges[bad][0]
        record["kept_text"] = processor.batch_decode(
            [cont_ids[:cut_start]], skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        record["cut_text"] = processor.batch_decode(
            [cont_ids[cut_start:]], skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        if use_negative_guidance:
            if negative_marker is not None:
                marker_text = negative_marker
            else:
                marker_cut_text = record["cut_text"][:max_negative_marker_chars]
                marker_text = NEGATIVE_MARKER_TEMPLATE.format(cut_text=marker_cut_text) if marker_cut_text else _DEFAULT_NEGATIVE_MARKER
            marker_ids = tokenizer(marker_text, add_special_tokens=False, return_tensors="pt").input_ids.to(prefix_ids.device)
            negative_prefix_ids = torch.cat([out[:, : prefix_ids.shape[1] + cut_start], marker_ids], dim=1)

        prefix_ids = out[:, : prefix_ids.shape[1] + cut_start]
        prev_answer = answer

    return text, len(history), history
