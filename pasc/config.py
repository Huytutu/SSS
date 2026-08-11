from dataclasses import dataclass

__all__ = ["PASConfig"]


@dataclass
class PASConfig:
    """Every tunable knob in pasc, in one place.

    Most of these are NOT calibrated -- they are starting points picked to be
    explainable, and each one is marked below. Tune them against
    scripts/test/eval_treebench_pasc.py before reporting numbers.
    """

    # --- which decoder layers to read attention from -------------------------
    # PAS (arXiv:2511.11502) ablated layer 0 on LLaVA-family models. That does
    # not transfer to Qwen2.5-VL: measured on 228 labelled RH-Bench
    # multiple-choice items, detection AUROC was 0.558 at layer 0 vs 0.614 at
    # layer 18 and 0.605 at layer 22. Layer 0 also puts only ~3% of its
    # attention on image tokens here, against ~8-23% at layers 14-26.
    # AFIP (arXiv:2605.24602) predicts exactly this: the visual signal only
    # appears after the modalities have fused, in intermediate/deep layers.
    pas_layer: int = 18
    # MLLMs Know Where to Look (arXiv:2502.17422) uses layer 22 for Qwen2.5-VL
    # (their qwen2_5_methods.py ATT_LAYER). Their own comment invites trying
    # other layers.
    localize_layer: int = 22

    # --- when to flag a token ------------------------------------------------
    # What decides that a token is worth a crop.
    #
    #   "either"      -- uncertain OR ungrounded. The default.
    #   "uncertainty" -- small top1-top2 gap. ECRD's premise.
    #   "pas"         -- unusually text-driven attention (PAS below).
    #   "both"        -- uncertain AND ungrounded.
    #
    # Why the union. The two signals catch disjoint failures, and neither alone
    # is enough:
    #
    #   * Uncertainty catches tokens the model is torn about -- but a
    #     hallucination is usually *confident*. Measured over 113 count, side
    #     and colour tokens, median p_top was 0.89-0.999 and a `gap < 0.2` rule
    #     fired on only 5-11% of them. Left/right sat at p_top = 0.999.
    #   * PAS catches confidently ungrounded tokens, which is exactly that
    #     population -- but says nothing about whether an alternative exists.
    #
    # The two are statistically indistinguishable as *detectors* on labelled
    # RH-Bench (PAS 0.614 vs gap 0.587, paired bootstrap +0.028, 95% CI
    # [-0.070, +0.120]), so this is not about which ranks better. It is about
    # covering both failure shapes.
    gate_mode: str = "either"
    # Fire when the top-1/top-2 probability gap falls below this. Measured over
    # 12.8k RH-Bench tokens, 0.2 flags ~14% of them before the stopword filter
    # and cooldown thin that out. ECRD's own default was 0.08 (~8%), which on a
    # fluent answer fires so rarely that the crop machinery barely runs.
    gap_thresh: float = 0.2
    # ...and when the knee kept at least this many candidates, i.e. the model
    # itself treats more than one as live.
    min_knee_k: int = 2

    # Which grounding signal the "pas"/"both" modes gate on. See attn_probe.StepSignals.
    #   "pas_raw"   -- PAS eq. 10 verbatim, and the only one that measured
    #                  above chance (AUROC 0.614 at layer 18). Caveat: it is a
    #                  sum over prelim tokens, so it grows with generation
    #                  length. It compares cleanly between generations at
    #                  similar positions -- PAS's own detection setting -- but
    #                  a fixed tau does get easier to cross late in a long answer.
    #   "pas_share" -- our attempt to remove that length trend by dividing by
    #                  prelim + image mass. It destroyed the signal
    #                  (AUROC 0.44-0.53 at every layer). Kept only for ablation.
    #   "peak_diff" -- AFIP's quantity, peak textual minus peak visual
    #                  attention. Length-invariant, untested against labels.
    #                  Note AFIP gates the opposite way (it damps correction
    #                  when this is high); we fire when it is high.
    gate_signal: str = "pas_raw"
    # How far above the answer's own running average the signal must sit, in
    # running standard deviations.
    #
    # Not an absolute threshold, and that is the point. pas_raw is a sum over
    # prelim tokens, so it climbs steadily as an answer gets longer: an absolute
    # cut of 0.5 (the 90th percentile over 12.8k RH-Bench tokens) fired almost
    # exclusively on tokens past step 90, whatever they said. Comparing each
    # token against a running mean and variance of the tokens before it removes
    # that drift while staying causal.
    #
    # Cross-example detection at a fixed position -- PAS's own setting, and what
    # the AUROC numbers above measure -- does *not* need this and is hurt by it.
    # Within one answer it is necessary.
    z_thresh: float = 1.5
    # Ignore the first few steps: a running mean over one or two samples is
    # noise, and prelim is nearly empty there anyway.
    warmup_steps: int = 8
    # Candidates shown to the corrector, by rank rather than probability mass.
    # 8, because that is where the right answer actually lives: over 113
    # measured count/side/colour tokens the correct alternative sat at median
    # rank 3-8 (85% within top-8 for left/right) while carrying ~1e-4
    # probability. Any probability-based cut misses it by construction.
    correct_top_n: int = 8
    # Minimum decoding steps between two corrections. Each correction costs a
    # full extra generate() call, so this bounds the worst case.
    cooldown: int = 5
    # Skip tokens that carry no visual claim (see trigger.is_groundable).
    # The original plan derived this from attention shape -- a function word
    # was expected to dump its prelim attention on the token right before it.
    # Measured across 16 layers, that separation is 0.01-0.04 and flips sign
    # between layers, so it is not usable; a stopword list does the job
    # reliably. `local_ratio` is still logged for anyone wanting to revisit it.
    skip_stopwords: bool = True
    # Only consider tokens that begin a new word. A mid-word continuation
    # piece is not an independent visual claim.
    require_word_start: bool = True

    # --- candidate set and evidence rescoring --------------------------------
    # Candidate set handed to the evidence scorer and the corrector: the
    # smallest group of tokens whose probabilities sum to `top_p`. See
    # logits_processor.nucleus_k for why this is not a largest-drop rule.
    top_p: float = 0.9
    min_k: int = 1
    max_k: int = 64
    # Temperature turning evidence cost into a distribution over candidates.
    rescore_tau: float = 1.0
    # Blend between the model's own distribution and the evidence-rescored one.
    # None means "use p_top", i.e. trust the model exactly as much as it trusts
    # itself -- confident steps stay untouched, unsure ones lean on evidence.
    mix_alpha: float = None
    # Ceiling on that weight. Without it, p_top sits at ~1.0 for most tokens of
    # a fluent answer, evidence gets ~0 weight, and the bank has no effect at
    # all -- which is precisely wrong for the confidently-hallucinating case the
    # method exists to catch. 0.9 leaves evidence a tenth of the say.
    max_mix_alpha: float = 0.9
    # Soft-min temperature when aggregating cost across evidence items: one
    # supporting sentence should be enough to clear a token.
    evidence_agg_tau: float = 1.0
    # Prefix positions kept per evidence sentence (see scorer.py).
    max_evidence_prefix: int = 128

    # --- cropping ------------------------------------------------------------
    # Base crop size in original-image pixels, before the adaptive ratio search
    # scales it by 1x..2x. mllms_know sets this to each model's native input
    # resolution (336 LLaVA, 224 BLIP); Qwen2.5-VL is dynamic-resolution so
    # there is no native value to copy. UNCALIBRATED.
    bbox_size: int = 448
    # Floor on the relative-attention denominator, as a fraction of its own
    # mean. Without it the crops land on empty sky: a third of the patches in a
    # typical photo (423 of 1247, measured) draw near-zero baseline attention,
    # and dividing by those blows the ratio up wherever the model looks least.
    # At 0.25, crops on a four-traffic-light scene moved from one-of-four
    # on empty sky to four-of-four on the lights themselves.
    baseline_floor_frac: float = 0.25

    # Tokens the corrector may generate when re-asked on the crop.
    correct_max_new_tokens: int = 64
