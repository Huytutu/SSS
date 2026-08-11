from __future__ import annotations

__all__ = ["PASTrigger", "is_groundable", "STOPWORDS"]

# Closed-class words plus the light verbs and quantifier-free fillers that make
# no visual claim on their own. Cropping the image to check "of" wastes a full
# extra generate() call and can only make things worse.
#
# This list replaces the attention-shape filter the design originally called
# for -- see PASConfig.skip_stopwords for the measurements that ruled that out.
STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "of", "to", "in", "on", "at", "by", "for",
    "with", "from", "as", "into", "onto", "over", "under", "about", "than", "then", "so",
    "is", "are", "was", "were", "be", "been", "being", "am", "do", "does", "did", "has",
    "have", "had", "can", "could", "will", "would", "shall", "should", "may", "might", "must",
    "it", "its", "this", "that", "these", "those", "there", "here", "which", "who", "whom",
    "what", "when", "where", "how", "why", "they", "them", "their", "we", "our", "you",
    "your", "he", "him", "his", "she", "her", "i", "me", "my", "not", "no", "also", "very",
    "both", "each", "such", "some", "any", "all", "more", "most", "other", "same", "while",
    "appears", "seems", "likely", "suggests", "indicating", "indicates", "showing", "shows",
}

_WORD_START_MARKS = ("Ġ", "▁", " ")

# Opening tags after which the very next token is the answer itself.
_ANSWER_TAGS = ("<answer>", "<ANSWER>")


def is_groundable(token: str, cfg, prefix_text: str = "") -> bool:
    """Could this token carry a visual claim worth checking against the image?

    False for word-continuation pieces, punctuation, and stopwords -- correcting
    those costs a generate() call and cannot improve grounding.

    Exception for the token right after an opening answer tag. It has no leading
    space, so the word-start rule would reject it -- yet it is the single most
    consequential token in the whole generation, and on a benchmark it is the
    only one that is scored. Measured on TreeBench, that rejection silently made
    the answer token ineligible even when the model was visibly torn about it
    (index=6: gap 0.183, six live candidates, never fired).
    """
    if token is None:
        return False
    if cfg.require_word_start and not token.startswith(_WORD_START_MARKS):
        if not prefix_text.rstrip().endswith(_ANSWER_TAGS):
            return False

    word = token.lstrip("Ġ▁ ").strip().lower()
    if not word or not any(c.isalnum() for c in word):
        return False
    if cfg.skip_stopwords and word in STOPWORDS:
        return False
    return True


class PASTrigger:
    """Decides whether the token about to be emitted deserves a look at the image.

    Always required: the token could carry a visual claim (see is_groundable),
    and enough steps have passed since the last correction.

    Then, by `cfg.gate_mode`:

    **uncertainty** -- the model is torn here: the top-1/top-2 gap is below
    `gap_thresh` and the nucleus kept at least `min_knee_k` candidates.

    **pas** -- the grounding signal stands out from this answer's own running
    average by `z_thresh` standard deviations: the model just got unusually
    text-driven relative to how it had been reading. The running comparison
    matters because PAS's raw score climbs with answer length, so an absolute
    threshold just fires on whatever comes last. Statistics update on every
    groundable step, fired or not, so a burst of corrections cannot drag the
    baseline up after itself.

    **both** -- uncertain and ungrounded. **either** (the default) -- one or the
    other, since the two catch disjoint failures: uncertainty finds tokens the
    model hesitates on, PAS finds the confident-but-ungrounded ones that make up
    most hallucinations. See PASConfig.gate_mode for the measurements.

    `last_z` and `last_gap` hold the most recent values whether or not they
    fired, so callers can log how close each token came.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.last_fired = -10**9
        self.last_z = 0.0
        self.last_gap = 1.0
        self._n = 0
        self._mean = 0.0
        self._m2 = 0.0     # sum of squared deviations, Welford's online variance

    def should_fire(self, signals, token: str, step: int, gap: float, knee_k: int,
                    prefix_text: str = "") -> bool:
        self.last_z = 0.0
        self.last_gap = gap
        if not is_groundable(token, self.cfg, prefix_text):
            return False

        # Always fold the sample in, even in uncertainty mode, so the running
        # baseline is comparable across modes and available for logging.
        self.last_z = self._update(signals.value(self.cfg.gate_signal))

        if step - self.last_fired < self.cfg.cooldown:
            return False

        mode = self.cfg.gate_mode
        uncertain = gap < self.cfg.gap_thresh and knee_k >= self.cfg.min_knee_k
        ungrounded = self._n > self.cfg.warmup_steps and self.last_z > self.cfg.z_thresh

        if mode == "either":
            fire = uncertain or ungrounded
        elif mode == "uncertainty":
            fire = uncertain
        elif mode == "pas":
            fire = ungrounded
        elif mode == "both":
            fire = uncertain and ungrounded
        else:
            raise ValueError(f"unknown gate_mode {mode!r}")

        if fire:
            self.last_fired = step
        return fire

    def _update(self, value: float) -> float:
        """z-score of `value` against the steps before it, then fold it in."""
        std = (self._m2 / self._n) ** 0.5 if self._n > 1 else 0.0
        z = (value - self._mean) / std if std > 1e-9 else 0.0

        self._n += 1
        delta = value - self._mean
        self._mean += delta / self._n
        self._m2 += delta * (value - self._mean)
        return z
