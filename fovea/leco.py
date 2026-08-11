from __future__ import annotations
from typing import Any, Callable, Dict, List, Optional, Tuple

import math
import re

import torch
from transformers import LogitsProcessor
from transformers.generation import LogitsProcessorList

from .step_filter import is_structural_step
from .attn_scorer import find_image_token_span
from .attn_hook import AttentionCostCollector, AttnCostRecorder

__all__ = [
    "segment_steps", "score_step", "step_dev_cost", "step_attn_cost",
    "find_lowest_scoring_step", "find_first_bad_step", "leco_loop",
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


def step_dev_cost(step_log_slice: List[Dict[str, Any]], dev_quantile: float = 0.9) -> float:
    """High quantile of a step's per-token deviation cost -- see score_step's
    `dev_score` docstring for why the quantile (not the mean) is used. Exposed
    separately so callers can compute all steps' costs first and build
    `dev_stats` for score_step (see leco_loop)."""
    dev_costs = torch.tensor([r["dev_cost"] for r in step_log_slice])
    return float(torch.quantile(dev_costs, dev_quantile))


def step_attn_cost(step_log_slice: List[Dict[str, Any]], attn_quantile: float = 0.9) -> float:
    """High quantile of a step's per-token attention-ungroundedness cost --
    mirrors step_dev_cost exactly (same one-bad-token-should-dominate
    rationale), reading "attn_cost" (see score_step's `attn_score` docstring
    and fovea/attn_hook.py's AttentionCostCollector, which populates it)
    instead of "dev_cost". Only present in step_log entries when leco_loop
    was run with attn_weight > 0."""
    attn_costs = torch.tensor([r["attn_cost"] for r in step_log_slice])
    return float(torch.quantile(attn_costs, attn_quantile))


def _detrend_step_costs(costs: torch.Tensor) -> torch.Tensor:
    """OLS-regress `costs` (one value per step, in step order) on step index
    and return the residuals. Guards against a *positional* confound in
    `attn_cost` dominating the within-generation z-score below -- the
    concern that originally motivated this (VAR, AFIP eq.2, reported to
    decline monotonically with decode position) is no longer the signal in
    use (attn_cost is now eq.3's cross-head KL, which the paper does not
    report any such trend for), so with the current signal this is mostly a
    cheap safety net rather than a load-bearing fix -- on a genuinely flat
    series it reduces to subtracting the mean, which the downstream z-score
    would do anyway (OLS residuals are
    always exactly mean-zero, regardless of the fitted slope, so this never
    corrupts the z-score even when there's no real trend to remove).

    Caveat if re-applied to a signal with a real trend: with the ~3-8 steps
    typical of one generation here, a per-generation linear fit is a weak
    estimator (high slope variance on so few points) and can't correct a
    non-linear decline -- a population-level baseline (expected cost vs.
    relative step position, fit across many generations) would be more
    robust than re-fitting a line from scratch every time, but is out of
    scope here.

    No-ops below 3 steps: a line fits any 2 points exactly, so residuals
    would be identically zero and carry no signal."""
    n = costs.numel()
    if n < 3:
        return costs
    idx = torch.arange(n, dtype=costs.dtype)
    idx_c = idx - idx.mean()
    slope = (idx_c * (costs - costs.mean())).sum() / (idx_c ** 2).sum()
    intercept = costs.mean() - slope * idx.mean()
    return costs - (intercept + slope * idx)


def score_step(
    step_log_slice: List[Dict[str, Any]],
    next_step_head: List[Dict[str, Any]],
    K: int = 3,
    dev_quantile: float = 0.9,
    dev_stats: Optional[Tuple[float, float]] = None,
    dev_temperature: float = 1.0,
    dev_weight: float = 0.5,
    attn_quantile: float = 0.9,
    attn_stats: Optional[Tuple[float, float]] = None,
    attn_temperature: float = 1.0,
    attn_weight: float = 0.0,
    attn_cost_override: Optional[float] = None,
) -> float:
    """LeCo step confidence: avg_score + trans_score - diver_score - dev_score - attn_score.

    All four terms read `true_prob` / `dev_cost` logged by VDGDLogitsProcessor
    from the model's pre-rescore distribution -- not the VDGD-rescored
    (prompt-deviation) distribution, which reflects grounding cost rather than
    the model's own confidence and would otherwise collapse all four terms
    onto a single upstream signal (how many candidates knee-truncation kept).

    avg_score: mean true probability of the chosen token across the step's tokens.
    diver_score: 1 - H(P)/log(n), where P is the normalized distribution of those
        same per-token true probabilities across the step's n positions. 0 when
        confidence is flat across the step (ideal), -> 1 as it spikes on a few
        tokens while the rest are near-zero. Entropy-normalized so it's bounded
        to [0, 1] and comparable across step lengths, unlike a raw KL average.
    trans_score: mean true probability over the first K tokens of the *next* step
        (heading tokens). If there is no next step, defaults to this step's own
        avg_score (neutral) rather than 0 -- 0 would structurally under-score every
        terminal step regardless of its actual confidence, biasing rollback toward
        the last step (typically a trivial "Therefore, the answer is:" transition
        in this pipeline's free-form CoT) instead of a genuine low-confidence one.
    dev_score: grounding term -- how ungrounded the step's tokens are in the
        image-conditioned prompt, from PromptDeviationScorer's cost on the
        actually-chosen token. Uses a high quantile rather than the mean, since
        one ungrounded token (e.g. a hallucinated object) should dominate over
        many well-grounded filler tokens.

        `dev_stats`, if given, is the (mean, std) of `step_dev_cost` across all
        steps in the same generation (see leco_loop). Deviation cost is only
        near zero for near-verbatim claims (e.g. a color copied from the
        earlier description) and otherwise spans a wide range that a fixed
        absolute squash (e.g. 1 - e^-x) saturates well before the top of, so
        genuinely fabricated steps end up scored the same as ordinary
        reasoning scaffolding -- hence z-scoring against the generation's own
        mean/std before squashing. A first attempt min-max'd against
        (min, max) instead, but that forces the single highest-cost step to
        dev_score=1.0 in *every* generation regardless of how large the
        actual spread is -- since real deviation cost varies far more than
        avg/trans/diver do step-to-step, that mechanically made dev_score the
        largest-variance term and let it override the other three whenever
        they disagreed (confirmed empirically: on one trace the argmin of
        avg+trans-diver alone picked a different step than the full sum).
        `dev_temperature` controls how sharply z-scores are pushed toward
        0/1 by the sigmoid -- higher softens (smaller dev_score variance,
        less influence on the sum), lower sharpens. Falls back to the fixed
        1 - e^-x squash when no stats are given (e.g. scoring a single step
        in isolation).
    attn_score: attention-inconsistency term -- how much the step's tokens'
        attention heads disagree with each other about where in the image to
        look (AFIP, arXiv:2605.24602, eq.3: D_kl_t, mean cross-head KL
        divergence between each head's image-token attention and the
        collective/head-averaged one). Off by default (attn_weight=0.0) --
        opt in via leco_loop's `attn_weight`/`image_token_id`. Computed by
        fovea/attn_hook.py's `AttentionCostCollector`, which monkey-patches
        a small range of decoder layers to compute this inline and discard
        the attention tensor immediately -- see that module's docstring for
        why (in short: `output_attentions=True` on the full model was tried
        first and OOMs on a 24GB GPU for image-heavy prompts).

        Chose eq.3 over eq.2 (VAR, "how much attention landed on the image
        at all") after review: VAR is reported to decline monotonically over
        decode position across model backbones, which is a *when* effect
        that would dominate the within-generation z-score below and is
        disconnected from *correctness* (a legitimate late step can
        legitimately need little visual grounding). D_kl has no such
        reported temporal trend, and as a per-position head-disagreement
        measure it is a spikier, more token-localized signal -- a better
        match for the quantile-dominance treatment below (designed for
        dev_cost's similarly spiky per-token NLL) than VAR's slow,
        position-correlated one would have been.

        Same quantile + z-score-against-this-generation's-own-stats + sigmoid
        treatment as dev_score, for the same reasons documented above (one
        badly-inconsistent token should dominate a step of consistent ones;
        z-scoring against `attn_stats`, the (mean, std) of `step_attn_cost`
        across this generation's own steps, rather than a fixed squash or
        min-max, keeps one step from mechanically saturating attn_score in
        every generation).

        `attn_cost_override`, if given, is used as the step's `attn_cost_q`
        directly instead of recomputing it from `step_log_slice` via
        step_attn_cost -- leco_loop passes the already-detrended value here
        (see `_detrend_step_costs`) so this function doesn't undo that by
        recomputing the raw quantile.
    """
    if not step_log_slice:
        return 0.0
    true_probs = torch.tensor([r["true_prob"] for r in step_log_slice])
    avg_score = float(true_probs.mean())

    n = true_probs.numel()
    if n <= 1:
        diver_score = 0.0
    else:
        p = true_probs / true_probs.sum()
        entropy = float(-(p * torch.log(p.clamp_min(1e-12))).sum())
        diver_score = 1.0 - entropy / math.log(n)

    head = next_step_head[:K]
    trans_score = sum(r["true_prob"] for r in head) / len(head) if head else avg_score

    dev_cost_q = step_dev_cost(step_log_slice, dev_quantile)
    if dev_stats is None:
        dev_score = 1.0 - math.exp(-dev_cost_q)
    else:
        dev_mean, dev_std = dev_stats
        z = (dev_cost_q - dev_mean) / (dev_std + 1e-6)
        dev_score = 1.0 / (1.0 + math.exp(-z / max(dev_temperature, 1e-6)))

    if attn_weight == 0.0:
        attn_score = 0.0
    else:
        attn_cost_q = attn_cost_override if attn_cost_override is not None else step_attn_cost(step_log_slice, attn_quantile)
        if attn_stats is None:
            attn_score = 1.0 - math.exp(-attn_cost_q)
        else:
            attn_mean, attn_std = attn_stats
            z = (attn_cost_q - attn_mean) / (attn_std + 1e-6)
            attn_score = 1.0 / (1.0 + math.exp(-z / max(attn_temperature, 1e-6)))

    return avg_score + trans_score - diver_score - dev_weight * dev_score - attn_weight * attn_score


def find_lowest_scoring_step(scores: List[float], exclude: Optional[set] = None) -> int:
    """argmin(scores), skipping any index in `exclude` (e.g. the terminal
    pre-<answer> step, or steps flagged by is_structural_step -- see
    leco_loop). Falls back to considering every index if `exclude` would
    otherwise leave nothing to choose from."""
    exclude = exclude or set()
    candidates = [i for i in range(len(scores)) if i not in exclude]
    if not candidates:
        candidates = list(range(len(scores)))
    return min(candidates, key=lambda i: scores[i])


def find_first_bad_step(scores: List[float], exclude: Optional[set] = None, z_thresh: float = 1.0) -> int:
    """First step (in generation order), among candidates, whose score falls
    more than `z_thresh` robust standard deviations below the median of the
    candidate scores -- rolling back to the step that *first* went wrong,
    not whichever step happens to score worst overall.

    find_lowest_scoring_step's plain argmin picks the single worst step
    regardless of where it falls in the chain: if step 1 has one real
    hallucinated token and step 2 has five, argmin always picks step 2, so
    the rollback keeps regenerating downstream of step 1's still-uncorrected
    error every round. That failure isn't specific to any one grounding term
    (dev_score/attn_score) -- it's the selection rule itself comparing
    steps' *severity* when what rollback actually needs is the first step
    that crossed some *is-this-acceptable* line.

    Uses median and MAD (median absolute deviation, scaled by 1.4826 to be
    comparable to a standard deviation), not mean/std: a first version used
    mean/std and failed on exactly the scenario above in testing -- step 2's
    large dip inflates the *mean's* std enough that step 1's smaller-but-real
    dip no longer clears a mean-based threshold, i.e. the big error masks
    the earlier small one, which is the opposite of the intended fix.
    Median/MAD is the standard robust alternative: a single severe outlier
    step barely moves the median or the median-of-absolute-deviations, so it
    doesn't raise the bar for detecting other, milder anomalies.

    `z_thresh` is an unavoidable free parameter -- how far below the pack
    counts as "wrong enough" isn't derivable from the scores alone and needs
    empirical tuning; 1.0 is a permissive starting point, not a validated
    value.

    Falls back to find_lowest_scoring_step's argmin if no candidate crosses
    the threshold (e.g. all steps score similarly -- nothing stands out
    enough to call "first bad", but LeCo still needs to pick something once
    segmentation has already found >1 step)."""
    exclude = exclude or set()
    candidates = [i for i in range(len(scores)) if i not in exclude]
    if not candidates:
        candidates = list(range(len(scores)))
    cand_scores = torch.tensor([scores[i] for i in candidates])
    median = torch.quantile(cand_scores, 0.5)
    mad = torch.quantile((cand_scores - median).abs(), 0.5)
    scale = float(1.4826 * mad) + 1e-6
    for i in candidates:  # `candidates` is already in chronological order
        z = (scores[i] - float(median)) / scale
        if z < -z_thresh:
            return i
    return find_lowest_scoring_step(scores, exclude=exclude)


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
    dev_weight: float = 0.5,
    attn_weight: float = 0.0,
    image_token_id: Optional[int] = None,
    attn_layer_range: Optional[Tuple[int, int]] = None,
    attn_temperature: float = 1.0,
    attn_quantile: float = 0.9,
    detrend_attn: bool = True,
    rollback_z_thresh: float = 1.0,
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

    `dev_weight` (default 0.5): weight of score_step's dev_score term, exposed
    here so callers can isolate/ablate it against `attn_weight` (e.g. dev_weight=0
    to test attn_score alone). See score_step's `dev_score` docstring.

    `attn_weight` (opt-in, off by default): weight of the attention-based
    grounding term in score_step (see its `attn_score` docstring). Requires
    `image_token_id` (e.g. `model.config.image_token_id`) to locate the image
    tokens in the prompt. Unlike an earlier version of this, does NOT require
    attn_implementation="eager" or output_attentions=True -- fovea/attn_hook.py's
    AttentionCostCollector monkey-patches just `attn_layer_range` layers
    (default (5, 18) -- NOT the full stack; capturing all 28 layers' attention
    for a whole generation is what OOM'd on a 24GB GPU with image-heavy
    prompts) to compute each step's Dkl inline and discard the attention
    tensor immediately, so the rest of the model keeps using its normal
    fast attention kernel. `detrend_attn` (on by default when attn_weight > 0)
    regresses each round's per-step attn_cost on step index and z-scores the
    residual instead of the raw value -- see `_detrend_step_costs` (mostly a
    safety net with the current eq.3-based attn_cost).

    `rollback_z_thresh` (default 1.0): passed to `find_first_bad_step`,
    which picks the first step (not the single worst one) whose score falls
    more than this many standard deviations below the round's own step
    scores -- see that function's docstring for why plain argmin biases
    rollback toward whichever step is most severe rather than whichever
    went wrong first.
    """
    extract = extract_answer or _default_extract_answer
    tokenizer = processor.tokenizer
    prompt_len = int(inputs["input_ids"].shape[1])
    vision_kwargs = {
        key: inputs[key] for key in ("pixel_values", "image_grid_thw") if inputs.get(key) is not None
    }

    img_span: Optional[Tuple[int, int]] = None
    collector: Optional[AttentionCostCollector] = None
    if attn_weight > 0:
        if image_token_id is None:
            raise ValueError("leco_loop: attn_weight > 0 requires image_token_id (e.g. model.config.image_token_id).")
        # Image tokens live in the fixed prompt prefix, which every iteration's
        # (possibly-rolled-back) prefix_ids still starts with -- computed once.
        img_span = find_image_token_span(inputs["input_ids"], image_token_id)
        # Default (5, 18), NOT the full 28-layer stack: patching (and thus
        # eager-computing) every layer for a whole generation is what OOM'd
        # on a 24GB GPU with image-heavy prompts -- see attn_hook.py.
        collector = AttentionCostCollector(model, layer_range=attn_layer_range or (5, 18), img_span=img_span)

    prefix_ids = inputs["input_ids"]
    prev_answer: Optional[str] = None
    text = ""
    history: List[Dict[str, Any]] = []
    negative_prefix_ids: Optional[torch.Tensor] = None

    try:
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
            if collector is not None:
                # Must come after `proc`: reads proc.step_log[-1], which proc's
                # own __call__ just appended for this same step (see
                # AttnCostRecorder's docstring).
                processors.append(AttnCostRecorder(collector, proc))

            gen_kwargs: Dict[str, Any] = dict(
                input_ids=prefix_ids,
                attention_mask=torch.ones_like(prefix_ids),
                generation_config=gen_config,
                max_new_tokens=remaining_budget,
                logits_processor=LogitsProcessorList(processors),
                **vision_kwargs,
            )
            out = model.generate(**gen_kwargs)
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

            # Mean/std of deviation cost across this generation's own steps -- see
            # score_step's `dev_stats` docstring for why z-scoring (not a fixed
            # absolute squash, and not min-max) is used.
            step_dev_costs = torch.tensor([step_dev_cost(proc.step_log[a:b]) for (a, b) in ranges])
            dev_stats = (float(step_dev_costs.mean()), float(step_dev_costs.std(unbiased=False)))

            # Same treatment for the attention-grounding cost -- see score_step's
            # `attn_score` docstring. Only computed when attn_weight > 0 (step_log
            # entries otherwise have no "attn_cost" key). Detrended against step
            # index first (see _detrend_step_costs) so attn_stats/attn_cost_override
            # are on the same (raw or residual) footing.
            attn_stats: Optional[Tuple[float, float]] = None
            step_attn_costs: Optional[torch.Tensor] = None
            if attn_weight > 0:
                step_attn_costs = torch.tensor(
                    [step_attn_cost(proc.step_log[a:b], attn_quantile) for (a, b) in ranges]
                )
                if detrend_attn:
                    step_attn_costs = _detrend_step_costs(step_attn_costs)
                attn_stats = (float(step_attn_costs.mean()), float(step_attn_costs.std(unbiased=False)))

            step_scores = [
                score_step(
                    proc.step_log[a:b],
                    proc.step_log[ranges[j + 1][0]:ranges[j + 1][0] + 3] if j + 1 < len(ranges) else [],
                    dev_stats=dev_stats,
                    dev_weight=dev_weight,
                    attn_quantile=attn_quantile,
                    attn_stats=attn_stats,
                    attn_temperature=attn_temperature,
                    attn_weight=attn_weight,
                    attn_cost_override=(float(step_attn_costs[j]) if step_attn_costs is not None else None),
                )
                for j, (a, b) in enumerate(ranges)
            ]
            # Exclude the last step (segment right before <answer>, typically a
            # trivial transition -- rolling back into it discards nothing
            # substantive) and any structural step (formatting/header/transition
            # line, or an option echo) from rollback candidacy: these score low
            # regardless of content and otherwise hijack rollback away from the
            # step that actually contains the error -- see step_filter.py.
            step_texts = [tokenizer.decode(cont_ids[a:b], skip_special_tokens=True) for (a, b) in ranges]
            exclude = {len(ranges) - 1}
            exclude |= {i for i, t in enumerate(step_texts) if is_structural_step(t)}
            bad = find_first_bad_step(step_scores, exclude=exclude, z_thresh=rollback_z_thresh)
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
    finally:
        if collector is not None:
            collector.restore()
