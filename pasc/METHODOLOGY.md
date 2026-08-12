# PASC: Uncertainty-Gated, Attention-Localized Self-Correction for VLM Chain-of-Thought

*Methodology*

This document is a paper-style write-up of the method implemented in this
directory. It states the problem, derives each component from the theory it
draws on, and gives the exact equations the code computes — including the
points at which the implementation departs from the papers it builds on, and
why. Code pointers are given in `` `file.py` `` form throughout; line-level
detail lives in code comments, not here. Empirical results are **not**
included — see [`README.md`](README.md), section *Honest status*, for what
has and has not been measured.

## 1. Problem statement

A vision-language model (VLM) answering a question about an image typically
writes a chain-of-thought: several sentences of reasoning before the final
answer. Long-form generation of this kind is known to drift from the visual
evidence as it proceeds — the model's own already-written text becomes an
increasingly strong prior on what comes next, and re-grounding in the image
becomes optional rather than load-bearing. This is one mechanism behind
object hallucination in VLMs (Rohrbach et al., 2018; Li et al., 2023).

Correcting this at inference time requires answering two separate questions
at every decoding step:

1. **When** should the model stop and re-examine the image, rather than
   continue from its own prior text? Re-examining at every step is
   prohibitively expensive (each check costs a full extra forward or generate
   pass); re-examining never forfeits the correction entirely.
2. **Where** in the image should it look? A step that decides "check the
   image" but has no way to select a region has no cheaper option than
   re-encoding the entire image at full resolution, which is both expensive
   and, for a large scene, not obviously more informative than the first
   pass.

This method answers (1) with the model's own output distribution — an
uncertainty signal that is free, already computed, and requires no auxiliary
model — and (2) with the model's own attention — a spatial signal that,
unlike a scalar confidence score, can name a location. Both signals are read
from the forward pass that is already happening; nothing here adds a second
model or a second full-resolution pass on the happy path.

## 2. Prior work this method builds on

Four papers motivate specific components. Each contributes one idea, and
each idea is used in a specific, narrower form than the paper's own framing
— the deviations are documented in Section 8.

**PAS — attention-based ungroundedness (arXiv:2511.11502, CVPR 2026).**
Proposes that a token generated with attention concentrated on the model's
own previously generated text, rather than on the image, is a token
generated *from language prior* rather than *from visual evidence*. Defines
a per-token score (their eq. 10) as the mean, across attention heads, of the
attention mass a token places on preceding text tokens. High score is read
as "text-driven, not image-driven."

**MLLMs Know Where to Look (arXiv:2502.17422, ICLR 2025).** Observes that a
VLM's raw attention map over image patches is dominated by
question-independent biases — attention sinks at image borders,
high-frequency texture drawing attention regardless of relevance — and that
subtracting out a *question-agnostic* attention map (obtained by prompting
with a generic "describe the image" instruction) isolates the
question-specific component. That relative map is then used to select a crop
for re-examination at higher effective resolution. This is a **contrastive
attribution** idea in the same family as contrastive/relevance saliency
methods (e.g., Selvaraju et al.'s Grad-CAM contrasts against a baseline
class; here the baseline is a generic prompt rather than a null class).

**AFIP (arXiv:2605.24602).** Studies *at which layer* a text-vs-image
attention imbalance is diagnostic. Its finding — that the signal only
becomes reliable in intermediate-to-deep layers, after cross-modal fusion
has happened, and is unreliable at layer 0 — is used here directly as a
layer-selection argument (Section 8.1). AFIP's own correction mechanism
(rewriting pre-softmax attention scores in place) is not implemented; see
Section 9.

**SAR (ACL 2024).** Proposes that self-augmented re-verification should
concentrate compute on *relevant* tokens rather than checking every token
sampled from a generation. The relevance criterion in the original paper
requires a finished sequence, an external cross-encoder, and multiple
sampled generations — machinery that is unavailable at the single token
about to be emitted. This method inherits SAR's *goal* (spend re-verification
budget selectively) but not its mechanism; Section 9 explains why an
attention-shape proxy for it was tried and abandoned.

## 3. System overview

At each decoding step `t`, four things happen, only one of them conditional:

```
attention row (2 layers) ──► grounding signal ──► trigger ─┐
next-token distribution  ──► candidate set                 │
        │                                                   ▼
        │                                  ┌── crop image, re-ask ───┐
        │                                  │  (only if trigger fires) │
        ▼                                  └────────────┬─────────────┘
   evidence bank ◄────────────────────────────────────────┘
        │
        └──► reweight candidate distribution ──► emit token
```

Reading attention and reweighting by the evidence bank happen on **every**
step and cost one extra small matmul and one cost lookup, respectively —
no extra model calls. Cropping and re-asking happen only when the trigger
fires, and are the only step that costs a full additional `generate()` call.

## 4. Notation

| Symbol | Meaning |
|---|---|
| `m` | prompt length, in tokens |
| `[a, b)` | image-token span in the prompt; `N = b − a` |
| `K` | length of the KV cache at the current step |
| `t` | decoding step index (0-based) |
| `H`, `V` | number of attention heads; vocabulary size |
| `l_pas`, `l_loc` | decoder layer read for the grounding signal / for localization |
| `ᾱ^(l)` | attention row of the newest query token at layer `l`, meaned over heads |
| `u_t` | `ᾱ^(l_loc)` restricted to the image span |
| `u_base` | same, computed under a question-agnostic prompt |
| `ρ` | relative attention map, reshaped to the image's patch grid |
| `p_t` | model's next-token distribution at step `t`, sorted descending |
| `C_t`, `k_t` | candidate token set and its size |
| `g_t` | top-1/top-2 probability gap |
| `z_t` | running z-score of the grounding signal |
| `E = {e_1, …, e_M}` | bank of evidence sentences accumulated so far |
| `c(v)` | evidence-implied cost of candidate token `v` |
| `α_t` | mixing weight between the model's own distribution and the evidence-reweighted one |

Configuration constants are introduced where first used and collected in
Section 10.

## 5. Reading attention without `output_attentions`

Computing this method's signals naively would call `model.generate(...,
output_attentions=True)`, which keeps every layer's full
`[heads, query_len, key_len]` attention tensor alive for the whole
generation. On an image-heavy VLM prompt (order 1,500 image tokens, tens of
decoder layers, hundreds of decoding steps) this exhausts a 24GB GPU; the
MLLMs Know Where to Look codebase hits the same limit and works around it by
loading weights in 4-bit.

The alternative used here exploits two facts about causal, KV-cached
decoding. First, only **one** new query token exists at each decoding step
after the prefill; every attention row for previously generated tokens was
already computed and discarded on a prior step, and is never needed again.
Second, that new query's attention row needs **no causal mask**: since it is
the most recent position, every key already in the cache is, by definition,
in its past.

This lets the implementation patch `forward` on exactly the two decoder
layers of interest (`l_pas`, `l_loc`). Each patched call first runs the
model's *real*, unmodified attention path — so the logits that reach the
sampler are bit-identical to an unpatched run — and only then performs one
additional matmul restricted to the newest query row:

```
q^(l,h) = RoPE( W_q^(l,h) x_last )                     one row only
α^(l,h) = softmax( q^(l,h) · (K^(l,h))ᵀ / √d_h )
ᾱ^(l)   = (1/H) Σ_h α^(l,h)                             mean over heads, per PAS eq. 10
```

Cost is `O(H · d_h · K)` per probed layer per step — a few hundred kilobytes
of extra compute — versus `O(L · H · q · K)` held live for the whole
generation under `output_attentions=True`.

## 6. Grounding signal extraction

The attention row `ᾱ^(l_pas)` is split at the current step into two spans:
*prelim* — everything generated by the model itself since the prompt ended,
excluding the position being scored — and *image* — the tokens spanning the
visual input.

```
prelim = ᾱ^(l_pas)_j ,  j = m, …, K−2
image  = ᾱ^(l_pas)_j ,  j = a, …, b−1

s_pas(t)   = Σ_j∈prelim ᾱ_j                                    (PAS eq. 10)
s_share(t) = s_pas / ( s_pas + Σ_j∈image ᾱ_j + ε )
s_peak(t)  = max_j∈prelim ᾱ_j − max_j∈image ᾱ_j                 (AFIP's peak-difference quantity)
```

`s_pas` is PAS's score, computed verbatim. It is a **sum**, not a mean or a
share, over an ever-growing prelim span — a property that matters for
Section 7. `s_share` was an attempt to remove that length dependence by
normalizing against total attended mass; empirically it destroys the signal
(see README), which is retained here as a cautionary note on why the
normalization is not applied by default. `s_peak` is AFIP's diagnostic,
included as a third option but not validated against labels in this
codebase.

**The causal offset.** PAS's own definition scores token `y_k` using the
attention row *at* `y_k` — i.e., the row computed when `y_k` is the query.
That row is only available once `y_k` has already been fed back into the
model, at which point emitting a *different* token requires a rollback.
Under KV-cached, single-token decoding, the row actually available while
*choosing* `y_k` is the one computed for the previous token, `y_{k−1}`:

```
signal used to gate step t  =  s_pas(t − 1)
```

This one-position lag is what makes intervention possible without
rewinding generation, at the cost of scoring "how grounded was the model's
state entering this choice" rather than PAS's literal "how grounded was this
token."

## 7. Uncertainty-gated triggering

### 7.1 Candidate set via nucleus sampling

The model's own distribution is not itself uncertain or certain; a criterion
is needed to decide how many alternatives it is still entertaining. This
method uses nucleus sampling (Holtzman et al., 2020) for that purpose — not
to *sample*, but to define the candidate set:

```
p_t = softmax(logits_t)                       sorted descending: p₍1₎ ≥ p₍2₎ ≥ …
k_t = clamp( min{ k : Σ_{i≤k} p₍i₎ ≥ P } , k_min, k_max )
C_t = { ids of the top-k_t tokens }
g_t = p₍1₎ − p₍2₎
```

An earlier version of this method used a largest-probability-drop ("knee")
rule instead of the cumulative-mass cutoff. In a distribution with the
typical steep, roughly geometric decay a language model produces, the single
largest drop is almost always the one immediately after the top token — so
the knee rule collapsed to `k=1` on the large majority of tokens measured,
including tokens where the top probability itself was under 0.4 and the
model was demonstrably still choosing between alternatives. Nucleus mass is
used instead because it is defined by *how much probability remains
unaccounted for*, which is the quantity that actually matters for "does an
alternative exist."

### 7.2 Why uncertainty, not grounding, decides *when* to correct

A first version of this method gated corrections on the grounding signal
`s_pas` alone, following PAS's own premise: correct when the model looks
ungrounded. This failed for a structural reason. `s_pas` is high exactly
when the model is attending to its own prior text rather than the image —
which is also, empirically, when the model is *confident*: a hallucinated
attribute (a wrong count, a wrong side, a wrong color) is usually stated
with very high probability, not tentatively. When the candidate set `C_t`
collapses to a single token because the model is sure, a correction step —
which can only *choose among candidates it is given* — has nothing to
change. Every correction under a pure-grounding gate was a no-op by
construction.

Gating (also) on uncertainty guarantees the candidate set is non-trivial
whenever a correction is attempted. The two signals are empirically close as
*detectors* of eventual error (see README, *Honest status*), so this is not
a claim that uncertainty ranks better than grounding — it is that they catch
different failure shapes, and a union is needed to catch both:

```
uncertain(t)  ⟺  g_t < γ  ∧  k_t ≥ k_min_knee
ungrounded(t) ⟺  n_samples > n_warmup  ∧  z_t > ζ
```

where `z_t` is defined next.

### 7.3 Length-invariant thresholding via a running z-score

`s_pas` is a **sum over the prelim span**, so on any fixed absolute
threshold, later tokens in a long answer are systematically more likely to
cross it regardless of what they say — the threshold measures answer length,
not grounding. This is corrected by comparing each step's signal to a
running estimate of the *mean and variance of the signal so far in this same
generation*, using Welford's numerically stable one-pass algorithm (Welford,
1962):

```
σ  = √( M₂ / n )                        population variance (n samples so far, not n−1)
z  = (s − μ) / σ                        if n > 1 and σ > 10⁻⁹, else 0

n  ← n + 1
δ  ← s − μ
μ  ← μ + δ/n
M₂ ← M₂ + δ · (s − μ_new)
```

`z` is computed against the samples that precede the current one, then the
current sample is folded into the running statistics — the update is
causal, and every groundable step contributes to the running baseline
*whether or not it triggered a correction*, so a burst of corrections cannot
drag the baseline upward and suppress subsequent detections.

This correction is explicitly a within-generation device: for *cross-example*
detection at a matched position — the setting PAS itself evaluates in — the
raw score without this correction is the more appropriate comparison, since
z-scoring against a single generation's own history has no meaning before
that generation exists.

### 7.4 Groundability filter

Not every token is worth checking against the image even when uncertain or
ungrounded — correcting a preposition or an article costs a full extra
generation and cannot improve grounding, since the word carries no visual
claim. A token `y` is *groundable* iff, writing `w` for the token with its
word-start marker stripped and lowercased:

```
groundable(y) ⟺ [ y starts a new word  ∨  the text so far ends with an opening <answer> tag ]
              ∧ w contains an alphanumeric character
              ∧ w is not a closed-class stopword
```

The disjunction with "ends with `<answer>`" exists because the token
immediately following an opening answer tag carries no leading whitespace —
so the word-start test alone would reject it — yet on a benchmark that
parses a tagged answer, that token is the *only* one that is ever scored.

### 7.5 The complete gate

```
fire(t) ⟺ groundable(y_t) ∧ (t − t_last_fired ≥ cooldown) ∧ MODE(t)

MODE(t) =  uncertain(t) ∨ ungrounded(t)     if gate_mode = "either"  (default)
           uncertain(t)                      if gate_mode = "uncertainty"
           ungrounded(t)                     if gate_mode = "pas"
           uncertain(t) ∧ ungrounded(t)      if gate_mode = "both"
```

`cooldown` exists purely for cost control: each firing is a full extra
`generate()` call, so a minimum spacing between corrections bounds the worst
case per answer.

## 8. Attention-based localization

### 8.1 Layer selection

MLLMs Know Where to Look and PAS were both developed and ablated primarily
on earlier VLM architectures (LLaVA-family, BLIP). Layer 0 was the default
grounding layer in the PAS paper. AFIP's account of *why* a text/image
attention imbalance becomes informative only after several layers — the
signal only exists once the two modalities have been fused by
self-attention, and layer 0 attention is essentially unimodal — predicts
that an early-layer default should transfer poorly to an architecture where
fusion happens at a different depth. This method uses a deep layer for
grounding and a separate deep layer (matching MLLMs Know Where to Look's own
published choice for Qwen2.5-VL) for localization, rather than reusing PAS's
layer-0 default; empirical support for this choice is in the README.

### 8.2 Contrastive normalization against a question-agnostic baseline

Raw attention over image patches in a VLM is known to be dominated by
biases that have nothing to do with the question being asked — attention
sinks at image borders, disproportionate mass on high-frequency or
high-contrast texture. MLLMs Know Where to Look's proposal is to isolate the
question-*specific* component by dividing the real question's image
attention by the same attention computed under a generic, question-agnostic
prompt (here, `"Write a general description of the image."`), on the theory
that the question-independent bias is present, and roughly equal, in both
passes and therefore cancels in the ratio. This is conceptually the same
move as contrastive explanation methods that attribute a prediction by
comparing against a reference/null input rather than reading raw activation
magnitude in isolation.

Concretely, with `u_base` the baseline row and `u_t` the real question's
image-attention row at the localization layer, both restricted to the image
span and merged into the model's native 2×2 patch-token grid:

```
ū_base   = mean_j u_base,j
ũ_base,j = max( u_base,j , φ · ū_base )              floor at fraction φ of the mean
ρ_j      = u_t,j / ũ_base,j
ρ        = reshape(ρ, H_p × W_p)
```

The floor `φ` is a deviation from the source method, which divides straight
through. That is safe when numerator and denominator are read at *the same*
query position, which is the source method's setting. Here the numerator is
read mid-generation — a different query position from the baseline pass —
and the two can disagree sharply on patches that draw almost no baseline
attention (a third of patches in a typical photo, by measurement): dividing
by a near-zero denominator produces an enormous ratio exactly on regions the
model is looking at *least*, e.g. open sky. Flooring the denominator at a
fraction of its own mean bounds this blow-up while leaving well-attended
patches' ratios essentially unchanged.

### 8.3 Adaptive multi-scale crop selection

Given the relative-attention map `ρ`, a bounding box is chosen by a
multi-scale sliding-window search, vendored from MLLMs Know Where to Look
(MIT-licensed) rather than reimplemented, since the scale-selection rule is
a specific enough numerical procedure that a rewrite risks silently
diverging from the published method.

For a base crop size `S` (in original-image pixels) and each candidate scale
`r ∈ {1, 1.2, …, 2}`:

```
n_x = min( ⌊S·r / b_x⌋, W_p ),   n_y = min( ⌊S·r / b_y⌋, H_p )        window size in patches
A_r(x, y) = Σ_{i<n_y} Σ_{j<n_x} ρ[y+i, x+j]                            windowed attention sum
(x*_r, y*_r) = argmax A_r
D_r = [ A_r(x*_r,y*_r) − mean of its 4 grid-neighbours ] / (n_x · n_y)  peak sharpness, per unit area
r* = argmax_r D_r
```

The final crop is centered on `(x*_{r*}, y*_{r*})` at scale `r*`, clamped to
stay inside the image. Choosing the scale by *peak sharpness relative to its
neighbours* rather than by raw magnitude is what makes the crop size
adaptive: a tight, well-localized attention blob produces a sharp peak at a
small window and is kept tight; a diffuse one only registers as sharp once
the window has widened enough to capture the whole blob, and is given a
wider crop accordingly.

## 9. Visual re-verification

When the trigger fires, the candidate set is widened by **rank** — the top
`N_corr` tokens by probability, filtered to tokens that are themselves
plausible content words — rather than left at the (typically much smaller)
nucleus set `C_t`. This is necessary because a confidently wrong token can
leave the correct alternative at very low probability while it is still
close in *rank*: on tokens measured for this method (count/side/colour
errors), the correct alternative sat at a median rank of 3–8 while carrying
on the order of `10⁻⁴` probability mass — reachable only by a rank-based
cut, never by a probability-mass cut, since the nucleus collapses to a
single candidate on almost all of these tokens.

The base VLM — not an auxiliary model — is then shown three things: the
full image, a crop of the attention-selected region, and a short prompt
listing the widened candidate list and asking it to (a) name one visual fact
it can see in the crop that supports a specific candidate, and (b) select
that candidate's index. Constraining the output to an index over a fixed
candidate list, rather than free-form regeneration, is what keeps a
correction from being able to introduce an unrelated token: the worst case
is picking the wrong candidate, not producing an off-distribution one.

If the model's answer differs from what would otherwise have been emitted,
the corresponding logit is given a large additive bonus, forcing it to be
selected:

```
if chosen ≠ argmax(scores):  scores[chosen] ← max(scores) + 50
```

This is a hard override, deliberately: since the chosen candidate can sit
far outside the nucleus, a soft reweighting proportional to the original
probability would be too weak to move the outcome.

### 9.1 On SAR's relevance criterion (not implemented)

SAR's own relevance measure — which tokens are worth re-verifying — is
defined over a *finished* sequence, using an external cross-encoder against
several independently sampled continuations. None of that machinery is
available at the single token about to be emitted, which is the setting
this method operates in. The originally planned substitute was an
attention-*shape* proxy: the hypothesis that function words (e.g. "of",
"and") dump their prelim attention onto the immediately preceding token,
distinguishing them structurally from content words without needing a
stopword list. Measured across many decoder layers, that separation turned
out to be small (on the order of a few hundredths) and to flip sign between
layers — not usable as a hard gate. A closed-class stopword list is used
instead, which is a cruder but empirically reliable substitute; the
attention-shape quantity is still logged per-step for anyone who wants to
revisit it with a different formulation.

## 10. Evidence accumulation as belief revision

A correction's most durable output is not the one token it may force — it
is the evidence sentence it produces, which is kept and used to influence
**every later token** in the answer, not only the one being corrected at
the moment of the crop. This is the primary reason a crop is judged worth
doing even on a token where the corrector's answer cannot change anything
(e.g. because the forced token is identical to what the model would have
emitted anyway): its value is downstream.

### 10.1 Scoring a candidate against one piece of evidence

Each evidence sentence `e_i` is encoded once, producing next-token
log-probabilities at every prefix position under the model's own language
head:

```
ℓ_i(π, v) = log p_θ(v | e_{i,1:π}),   π = 1, …, T_i,   T_i = min(|e_i|, T_max)
```

where `|e_i|` counts the tokenized sequence including special tokens. This
is computed once per evidence sentence — a single forward pass — and cached,
so scoring any later candidate token against it is a lookup, not a forward
pass; this is what keeps the evidence bank affordable to consult at every
decoding step rather than only at the step that produced it.

Aggregating within one sentence, over all positions rather than only the
final one, so that a token the sentence anticipates *anywhere* in its span
counts (a mean over positions, computed via log-sum-exp for numerical
stability):

```
q̄_i(v) = (1/T_i) Σ_π exp ℓ_i(π, v)
```

then smoothed with a small floor `ε_s = 10⁻⁶ / V` so a token that no
position of the sentence anticipates is penalized rather than assigned
literally infinite cost:

```
q̃_i(v) = (1 − ε_s)·q̄_i(v) + ε_s
c_i(v)  = − log q̃_i(v)
```

### 10.2 Aggregating across a growing bank of evidence

With multiple evidence sentences accumulated over the course of a
generation, the per-sentence costs are combined with a temperature-scaled
soft-minimum:

```
c(v) = − τ_a · [ log-sum-exp_i( −c_i(v) / τ_a ) − log M ]
```

This is not a soft-min in general — it is exactly a soft-min only as
`τ_a → 0`. At the default temperature `τ_a = 1`, the expression is exactly
the negative log of a **uniform mixture** over the bank:

```
c(v) |_{τ_a = 1} = − log[ (1/M) Σ_i q̃_i(v) ]
```

which is closer in spirit to a mixture-of-experts pooling of evidence than
to a "one supporting sentence is enough" rule; a single strongly supporting
sentence pulls the aggregate cost down but does not fully override
disagreement from the rest of the bank at this default. Lowering `τ_a`
moves the aggregation toward a true minimum, i.e. toward "one clear
supporting sentence suffices."

### 10.3 Blending evidence with the model's own belief

The evidence bank's implied preference over the candidate set is converted
to a distribution via a softmax over negative cost, then blended with the
model's own distribution:

```
q_t(v) = softmax( −c(v) / τ_r )   restricted to v ∈ C_t, −∞ elsewhere
q̂_t    = q_t · [ Σ_{v∈C_t} p_t(v) ] / [ Σ_{v∈C_t} q_t(v) ]
α_t    = min( p₍1₎ , α_max )
p'_t   = α_t · p_t + (1 − α_t) · q̂_t
```

The rescale of `q̂_t` before blending puts the evidence distribution on the
same total-mass footing as the candidates' share of the model's own
distribution, so that blending can *reorder* candidates without changing how
much total probability the candidate set holds relative to everything
outside it. Note `p'_t` is not itself a normalized distribution over the
full vocabulary — mass on `C_t` is preserved exactly by construction, while
mass outside `C_t` is scaled down by `α_t`, so `p'_t` sums to less than one.
Under the greedy decoding this pipeline uses, only the arg-max is consumed,
so this does not affect the emitted token; it would need to be renormalized
before use anywhere probabilities are read as probabilities (e.g. sampling,
perplexity).

`α_t = p₍1₎` (the default, when `mix_alpha` is unset) implements a simple
precision-weighting intuition from Bayesian belief combination: trust the
model's own distribution in direct proportion to how confident it already
is. A ceiling `α_max` is applied regardless of how `α_t` was obtained,
because for a token type where the model is nearly always highly confident
— which is exactly the profile of a *confident hallucination*, the failure
mode this method targets — `p₍1₎` alone would leave evidence essentially no
influence.

## 11. Cost accounting

| Stage | Frequency | Cost |
|---|---|---|
| Baseline attention row | once per image | 1 forward pass |
| Attention probe | every decoding step | 2 small matmuls, `O(H · d_h · K)` |
| Evidence cost lookup | every groundable step | gather over cached log-probs, no forward pass |
| Crop + re-ask | per triggered correction | up to `correct_max_new_tokens` decode steps, over a **two-image** prompt (full image and crop both go through the vision tower) |
| Evidence encoding | per evidence sentence added | 1 forward pass over a short sentence |

Only the last two rows are non-trivial in cost. All prior stages are
sub-linear additions to a forward pass that is already happening; only a
triggered correction adds a genuinely new `generate()` call, which is why
Section 7's gate is designed to be selective rather than to fire on every
uncertain or ungrounded step.

## 12. Limitations

This document describes what the method computes and the theoretical
motivation for each piece; it makes no claim that the method improves
answer accuracy. Detection quality, end-to-end accuracy, and known failure
modes (including a specific case where evidence reweighting corrupts the
expected output format) are reported, with numbers, in
[`README.md`](README.md) under *Honest status* — that section should be
read before this one is cited as evidence the method works.

## References

- H. Rohrbach, F. Hu, A. Vedantam, K. Saenko. *Object Hallucination in
  Image Captioning.* EMNLP 2018.
- Y. Li et al. *Evaluating Object Hallucination in Large Vision-Language
  Models.* EMNLP 2023.
- PAS. *Attention-based detection of ungrounded generation in VLMs.*
  arXiv:2511.11502, CVPR 2026.
- *MLLMs Know Where to Look: Training-free Perception of Small Visual
  Details with Multimodal LLMs.* arXiv:2502.17422, ICLR 2025.
  Code: https://github.com/saccharomycetes/mllms_know (MIT license;
  `bbox_from_att_image_adaptive` is vendored from this repository, see
  `rel_attention.py`).
- AFIP. arXiv:2605.24602.
- SAR. ACL 2024.
- A. Holtzman, J. Buys, L. Du, M. Forbes, Y. Choi. *The Curious Case of
  Neural Text Degeneration.* ICLR 2020. (Nucleus sampling, used here for
  candidate-set definition rather than generation.)
- B. P. Welford. *Note on a Method for Calculating Corrected Sums of
  Squares and Products.* Technometrics, 1962. (Online mean/variance
  update used by the running z-score gate.)
- R. R. Selvaraju et al. *Grad-CAM: Visual Explanations from Deep Networks
  via Gradient-based Localization.* ICCV 2017. (Cited for the general
  contrastive-attribution framing of Section 8.2; not otherwise used.)
