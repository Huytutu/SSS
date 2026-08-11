# PASC — uncertainty-gated, attention-cropped self-correction

During chain-of-thought, catch the token the model is genuinely torn about, zoom
in on the image region it was looking at, ask it again on the close-up, and keep
the resulting visual fact as evidence for the rest of the answer.

Built on ECRD's loop, with two things replaced:

| | ECRD | PASC |
|---|---|---|
| when to intervene | top1−top2 gap + hardcoded keyword list | top1−top2 gap + stopword filter |
| where to look | *nothing* — never crops (`bboxes` is hardcoded `[]`) | relative-attention crop |
| who answers | a second fine-tuned 3B model (GRIT) | the base model itself, on the crop |

```bash
bash scripts/run/run_pasc.sh path/to/image.jpg "How many traffic lights are there?"
```

---

## The problem

A VLM writing a long chain-of-thought hits moments where it is genuinely unsure
which word comes next — and at those moments it usually guesses from the text it
has already written rather than re-examining the image. The picture is right
there; the model just isn't looking closely enough.

Two things are needed to fix that mid-generation:

1. **Notice** the uncertain token, at the step it happens, without an extra pass.
2. **Act** on it — which requires knowing *where* in the image to look.

For (1) the model's own distribution is enough, and that is ECRD's premise. For
(2) a confidence number is useless — a scalar cannot point at anything — and
that is where attention comes in, via MLLMs Know Where to Look's relative
attention map.

### Why uncertainty, not PAS, decides *when*

The first version of this gated on PAS (attention-based ungroundedness). It
failed for a structural reason: PAS fires on tokens the model is *confident*
about, so the candidate set held a single token and the corrector — which can
only choose among candidates — had nothing to change. Every correction was a
no-op by construction.

Gating on uncertainty guarantees real alternatives exist. It also costs nothing
in detection quality: on 228 labelled RH-Bench items the two signals are
statistically indistinguishable (see *Honest status*). PAS's real contribution
is the attention map that places the crop, and that is kept.

`gate_mode` in `config.py` switches between `uncertainty` (default), `pas`, and
`both`, so the comparison is a one-line change.

## How it works

```
   ┌─ every decoding step ────────────────────────────────────────┐
   │                                                              │
   │  attention probe ──► PAS score ──► trigger ─┐                │
   │  (2 layers, free)                           │                │
   │                                             ▼                │
   │  next-token distribution ──► knee cut ──► candidates          │
   │            │                                │                │
   │            │                                ▼                │
   │            │                     ┌── crop the image ───┐     │
   │            │                     │  re-ask base model  │     │
   │            │                     └─────────┬───────────┘     │
   │            ▼                               ▼                 │
   │       evidence bank ◄────────────── evidence sentence        │
   │            │                                                 │
   │            └──► reweight candidates ──► emit token           │
   └──────────────────────────────────────────────────────────────┘
```

**1. Probe** (`attn_probe.py`). Reads one attention row per step from two
decoder layers. It does *not* replace the model's attention — it runs the real
kernel and adds one small matmul for the newest query row, so output is
bit-identical to an unpatched run. This matters: `output_attentions=True` keeps
every layer's full attention alive for the whole generation and exhausts a 24GB
GPU on image-heavy prompts.

**2. Trigger** (`trigger.py`). Fires when the token could carry a visual claim,
*and* the PAS score stands out from the answer's own running average, *and*
enough steps have passed since the last correction.

**3. Localize + correct** (`rel_attention.py`, `self_correct.py`). Divides the
step's image attention by a question-agnostic baseline, picks a crop around the
peak, then shows the base model the full image *and* the crop and asks it to
choose among the candidates and state one visual fact.

**4. Evidence** (`scorer.py`). That fact goes into a bank that reweights every
later token. This is why a crop is worth doing even when it cannot change the
current token — most of its value is downstream.

## Methodology

The equations below are what the code actually computes, not the papers'
originals — where the two differ, the difference is called out. This section
says how the method works, not that it works; for that see *Honest status*.

### 0. Notation

| Symbol | Meaning | In code |
|---|---|---|
| `m` | prompt length in tokens | `probe.prompt_len` |
| `[a, b)` | contiguous span of image tokens in the prompt, `N = b − a` | `probe.img_span` |
| `K` | KV-cache length at the current step | `row.shape[0]` |
| `t` | decoding step index, 0-based | `proc.step` |
| `H`, `V` | attention heads, vocabulary size | — |
| `l_pas`, `l_loc` | layer read for the PAS signal / for localization | `pas_layer=18`, `localize_layer=22` |
| `ᾱ^(l)` | attention row of the newest query at layer `l`, meaned over heads, length `K` | `probe._rows[l].mean(0)` |
| `u_t` | `ᾱ^(l_loc)` restricted to the image span, length `N` | `signals.img_row` |
| `u_base` | same, under the question-agnostic prompt | `baseline_row` |
| `ρ` | relative-attention map, `H_p × W_p` patch grid | `attention_map(...)` |
| `p_t` | model's next-token distribution, `p₍1₎ ≥ p₍2₎ ≥ …` sorted | `probs` |
| `C_t`, `k_t` | candidate set and its size | `cand_ids`, `k` |
| `g_t` | top-1/top-2 probability gap | `gap` |
| `z_t` | running z-score of the gate signal | `trigger.last_z` |
| `E = {e₁…e_M}` | evidence bank | `scorer._items` |
| `c(v)` | evidence cost of candidate `v` | `scorer.cost(...)` |
| `α_t` | blend weight, model vs evidence | `alpha` |
| `γ, ζ, Δ` | `gap_thresh=0.2`, `z_thresh=1.5`, `cooldown=5` | `config.py` |
| `N_corr` | candidates shown to the corrector, `correct_top_n=8` | `config.py` |
| `P, τ_r, τ_a` | `top_p=0.9`, `rescore_tau=1.0`, `evidence_agg_tau=1.0` | `config.py` |
| `φ, S` | `baseline_floor_frac=0.25`, `bbox_size=448` | `config.py` |

### 1. Per-step attention probe (`attn_probe.py`)

Two layers get their `forward` wrapped. The wrapper first calls the model's real
attention — so the emitted logits are bit-identical to an unpatched run — then
recomputes only the newest query row against the keys that call already wrote
into the cache:

```
q^(l,h)  = RoPE_mrope( W_q^(l,h) x_last )            one row, not the whole sequence
α^(l,h)  = softmax( q^(l,h) · K^(l,h)ᵀ · scaling )   scaling = 1/√d_h
ᾱ^(l)_j  = (1/H) Σ_h α^(l,h)_j                       mean over heads, as PAS does
```

No causal mask enters: the newest position has every cached key in its past, so
the full row is already valid. That is also why only the last row is readable.

Cost per step: one `[H, d_h] × [d_h, K]` matmul per probed layer, `O(H·d_h·K)`,
versus `output_attentions=True` materializing `O(L·H·q·K)` for every layer and
holding it for the whole generation.

### 2. Step signals (`attn_probe.pop_step`)

Split the row into *prelim* — previously generated text, excluding the current
position — and *image*:

```
prelim = ᾱ^(l_pas)_j ,  j = m … K−2          image = ᾱ^(l_pas)_j ,  j = a … b−1

s_pas(t)   = Σ_{j=m}^{K−2} ᾱ^(l_pas)_j                        PAS eq. 10
s_share(t) = s_pas / ( s_pas + Σ_{j∈image} ᾱ_j + ε )          length-normalization attempt
s_peak(t)  = max_{j∈prelim} ᾱ_j − max_{j∈image} ᾱ_j           AFIP's peak difference
r_local(t) = ᾱ_{K−2} / ( s_pas + ε )                          logged only
u_t        = ( ᾱ^(l_loc)_a , … , ᾱ^(l_loc)_{b−1} )            localization row
```

`s_pas` is a **sum**, so it grows with generation length — the reason §7 gates on
a running z-score rather than an absolute threshold. `s_share` was the attempt to
remove that trend by construction and destroyed the signal (AUROC 0.44–0.53);
`gate_signal` selects among the three.

### 3. The one-step causal offset

PAS scores token `y_k` with the attention row *at* `y_k`. That row only exists
once `y_k` has been fed back in — too late to change it without a rollback. With
a KV cache, the row available while choosing `y_k` is the one computed for
`y_{k−1}`:

```
signal used at step t  =  s_pas(t−1)      (PAS's definition is s_pas(t))
```

This is what makes intervention possible without rewinding, and it is the offset
under which every AUROC number in this README was measured.

### 4. Candidate set (`logits_processor.nucleus_k`)

```
p_t = softmax(z_t)                                     z_t = raw logits
k_t = clamp( min{ k : Σ_{i≤k} p₍i₎ ≥ P }, k_min, k_max )
C_t = { top-k_t token ids }
g_t = p₍1₎ − p₍2₎
```

Nucleus, not a largest-drop knee: in a steeply decaying distribution the biggest
drop is nearly always right after the top token, which returned `k=1` for 87% of
tokens (11143 of 12804 measured) and starved the corrector. The config key is
still named `min_knee_k` for the size floor used by the trigger.

### 5. Evidence cost (`scorer.py`)

Each evidence sentence `e_i` is encoded **once**, into next-token log-probs at
every prefix position:

```
ℓ_i(π, v) = log p_θ( v | e_{i,1:π} ) ,   π = 1 … T_i
T_i = min( |e_i| , max_evidence_prefix )
```

`e_i` here is the *tokenized* sequence, encoded with `add_special_tokens=True`,
so `|e_i|` counts the special tokens and the conditioning prefix at `π = 1` is
the leading special token rather than the first word of the sentence.

Stored as `−ℓ_i` in fp16 on CPU, so scoring a candidate later is a gather, not a
forward pass — which is what makes this affordable inside the decoding loop.

Aggregate over positions within a sentence (mean of probabilities, so a token the
sentence anticipates *anywhere* counts), smooth, then aggregate over sentences:

```
q̄_i(v) = (1/T_i) Σ_π exp ℓ_i(π, v)        computed as logsumexp_π ℓ_i(π,v) − log T_i
q̃_i(v) = (1 − ε_s)·q̄_i(v) + ε_s ,          ε_s = 1e−6 / V
c_i(v) = − log q̃_i(v)
c(v)   = − τ_a · [ logsumexp_i( −c_i(v)/τ_a ) − log M ]
```

The smoothing floor bounds the cost of a token no evidence anticipates, so it is
penalised rather than excluded outright.

The last line is a soft-min only in the `τ_a → 0` limit. At the shipped default
`τ_a = 1` it collapses to the negative log of a **uniform mixture** over the
bank:

```
c(v) |_{τ_a=1} = − log[ (1/M) Σ_i q̃_i(v) ]
```

so one strongly supporting sentence pulls the cost down but does not fully
override the rest. Lower `evidence_agg_tau` to approach a true min. With an empty
bank `c ≡ 0` and §6 is a no-op.

### 6. Evidence reweighting (`_apply_evidence`)

Applied only when the top token passes `is_groundable` (§7) — reweighting
punctuation by visual-consistency cost is meaningless.

```
q_t = softmax( ξ ) ,   ξ(v) = −c(v)/τ_r for v ∈ C_t,   −1e9 otherwise
q̂_t = q_t · ( Σ_{v∈C_t} p_t(v) ) / ( Σ_{v∈C_t} q_t(v) )
α_t = min( p₍1₎ , α_max )        α_max = max_mix_alpha = 0.9
p'_t = α_t · p_t + (1 − α_t) · q̂_t
scores ← log( p'_t + ε )
```

The rescale puts `q̂_t` on the same footing as the candidates' share of the
original mass, so the blend can reorder candidates without inflating their total.
It is not a normalized mixture: mass on `C_t` is preserved exactly, mass outside
`C_t` is scaled to `α_t·p_t(¬C_t)`, so `p'_t` sums to less than 1. Under greedy
decoding this is irrelevant — only the argmax matters.

`α_t = p₍1₎` means "trust the model exactly as much as it trusts itself." The cap
exists because `p₍1₎ ≈ 1.0` for most tokens of a fluent answer, which would give
evidence no weight at all — precisely wrong for the confidently-hallucinating
case the method targets. The cap applies to an explicitly set `mix_alpha` too.

### 7. Trigger (`trigger.py`)

**Groundability.** With `w` the token stripped of its word-start mark and
lowercased, and `x` the decoded tail of the sequence so far (last 8 tokens,
special tokens kept):

```
groundable(y) ⟺  [ y starts with Ġ/▁/space  ∨  x.rstrip() ends with <answer> ]
               ∧  w contains an alphanumeric character
               ∧  w ∉ STOPWORDS
```

Both bracketed conjuncts are conditional on their config flags — the first line
only applies when `require_word_start`, the last only when `skip_stopwords`; the
form above is the default configuration.

The `<answer>` disjunct exists because the answer token has no leading space, so
the word-start rule alone rejects the single most consequential token in the
generation — and on a benchmark, the only one that is scored. Measured on
TreeBench index=6, that rejection silently made the answer token ineligible even
with `gap = 0.183` and six live candidates.

**Running statistics.** Welford's online update, over every groundable step
whether or not it fired — so a burst of corrections cannot drag its own baseline
up. With `n` the count *before* the current sample `s`:

```
σ  = √( M₂ / n )                     population variance, n not n−1
z  = (s − μ) / σ        if n > 1 and σ > 1e−9, else 0

n  ← n + 1
δ  ← s − μ
μ  ← μ + δ/n
M₂ ← M₂ + δ·(s − μ_new)
```

`z` is computed against the preceding samples only, then the sample is folded in
— causal by construction.

**Gate.**

```
uncertain(t)  ⟺  g_t < γ  ∧  k_t ≥ min_knee_k
ungrounded(t) ⟺  n > warmup_steps  ∧  z_t > ζ

fire(t) ⟺ groundable(y_t) ∧ (t − t_last ≥ Δ) ∧ MODE
          MODE = uncertain ∨ ungrounded     ("either", default)
                 uncertain                  ("uncertainty")
                 ungrounded                 ("pas")
                 uncertain ∧ ungrounded     ("both")
```

The union is not about which signal ranks better — on labelled RH-Bench the two
are statistically indistinguishable (§*Honest status*). It is about covering two
disjoint failure shapes: uncertainty finds tokens with live alternatives but
misses confident hallucinations (median `p_top` 0.89–0.999 over 113 count/side/
colour tokens); PAS finds those but says nothing about whether an alternative
exists.

Note the loop order: `g_t` and `y_t` come from the **pre-evidence** distribution,
since §6 may have already rewritten `scores` by the time the trigger runs. The
logged gap is the model's own, not the effective one that produces the emitted
token.

### 8. Relative attention and crop (`rel_attention.py`)

**Baseline.** One forward pass per image under `"Write a general description of
the image."`, giving `u_base`. Dividing by it cancels the question-independent
bias — border patches, high-frequency texture — that makes raw attention a poor
localizer.

**Floored ratio.** Qwen2.5-VL merges each 2×2 block of 14px patches into one
image token, so the token grid is half `image_grid_thw` in each spatial
dimension:

```
ū_base   = mean_j u_base,j
ũ_base,j = max( u_base,j , φ · ū_base )              floor at φ = 25% of the mean
ρ_j      = u_{t,j} / ũ_base,j
ρ        = reshape( ρ , H_p × W_p ) ,  H_p = ⌊grid_h/2⌋ , W_p = ⌊grid_w/2⌋
```

mllms_know divides straight through, which is fine when numerator and denominator
are read at the same query position. Here the numerator is read mid-generation,
and the mismatch lets near-zero-baseline patches produce enormous ratios: 423 of
1247 patches in a typical photo, concentrated in exactly the regions the model
looks at least. The floor is what stops crops landing on empty sky.

**Adaptive window** (vendored from mllms_know, MIT). Patch size in pixels is
`b_x = W_img/W_p`, `b_y = H_img/H_p`. For each scale `r ∈ {1, 1.2, 1.4, 1.6, 1.8, 2}`:

```
n_x = min( ⌊S·r / b_x⌋ , W_p ) ,   n_y = min( ⌊S·r / b_y⌋ , H_p )

A_r(x, y) = Σ_{i<n_y} Σ_{j<n_x} ρ[ y+i , x+j ]                sliding window sum
(x*_r, y*_r) = argmax_{x,y} A_r(x, y)

D_r = [ A_r(x*, y*) − mean_{(x,y) ∈ 𝒩₄(x*,y*)} A_r(x, y) ] / (n_x · n_y)
```

`D_r` is how sharply the peak stands out from its 4-neighbours, per unit area.
Picking `r* = argmax_r D_r` is what makes the crop adaptive: a tight, confident
blob wins at a small scale and yields a tight crop; a diffuse one only stands out
once the window is wide.

```
S_crop = S · r*
x_c = x*·b_x + b_x·n_x/2 ,  clamped to [ S_crop/2 , W_img − S_crop/2 ]      (y likewise)
bbox = ( x_c − S_crop/2 , y_c − S_crop/2 , x_c + S_crop/2 , y_c + S_crop/2 ) ∩ image
```

### 9. Correct and force (`self_correct.py`, `_correct`)

The candidate list shown to the corrector is the top `max(k_t, N_corr)` ids
filtered to *content* tokens — word-initial and alphanumeric. Fewer than two
survive ⇒ no correction. The filter, not probability, is what keeps a correction
from turning `Look` into a newline.

`N_corr = 8` is a **rank** cut, not a probability cut, and that is the point: over
113 measured count/side/colour tokens the correct alternative sat at median rank
3–8 (85% within top-8 for left/right) while carrying ~1e−4 probability. The
nucleus of §4 cannot reach it — any probability-based cut misses it by
construction — so the corrector's list is widened by rank instead.

The base model — not a second model — is then shown `[full image, crop, prompt]`
and greedily decodes ≤ `correct_max_new_tokens`, returning `<evidence>…</evidence>`
and `<answer>i</answer>`. An unparseable reply is discarded.

```
v* = candidate[i]
if v* ≠ argmax:  scores[v*] ← max_v scores(v) + 50        hard override
if evidence ≠ "": E ← E ∪ { e_new }                       durable, affects all later steps
```

The `+50` is applied on top of the §6-blended scores, so the forced token wins
outright. Deliberately, `v*` may sit far outside the nucleus — `k_t = 1` for 90%
of left/right tokens at `p₍1₎ = 0.999`, so an earlier version that only applied
in-nucleus candidates was structurally unable to fix the errors the method exists
for.

### 10. Cost

| Stage | Frequency | Cost |
|---|---|---|
| baseline attention | 1 × per image | one forward pass |
| probe | every step | 2 extra matmuls, `O(H·d_h·K)` each |
| evidence cost | every groundable step | `O(M·T·k_t)` CPU gather, no forward |
| **crop + re-ask** | per firing | ≤ `correct_max_new_tokens` decode steps over a **two-image** prompt — the vision tower runs on the full image *and* the crop |
| evidence encode | per evidence added | one forward pass over a short sentence |

Only the last two rows matter. At ~6 firings per TreeBench question that is
~28s/question against ~8s for base; `cooldown` and `z_thresh` are the budget
knobs.

## Where each piece comes from

| Paper | Contributes | Equation used |
|---|---|---|
| **PAS** (arXiv:2511.11502, CVPR 2026) | *when* to intervene | `s_prel(k) = (1/H) Σ_h Σ_{j=m+1}^{k-1} A^(l,h)(k,j)` — attention from the current position back onto previously generated text, mean over heads. High = text-driven, not image-driven. |
| **MLLMs Know Where to Look** (arXiv:2502.17422, ICLR 2025) | *where* to look | Image attention divided by the same attention under `"Write a general description of the image."`, reshaped to the patch grid, then a multi-scale sliding window picks the crop. |
| **AFIP** (arXiv:2605.24602) | *which layer*, and the length critique | Deep layers only, after modality fusion. Its peak-difference gate `T_t − V_t` is available as `gate_signal="peak_diff"`. |
| **SAR** (ACL 2024) | *which tokens matter* — **not implemented faithfully**, see below | — |

## Three things we changed, and why

Each of these came out of a measurement, not a preference.

**PAS's layer 0 does not transfer to Qwen2.5-VL.** Layer 0 puts ~3% of its
attention on image tokens here, against 8–23% at layers 14–26. On 228 labelled
RH-Bench multiple-choice items, detection AUROC was 0.558 at layer 0 versus
**0.614 at layer 18**. Default is now layer 18. AFIP predicts exactly this.

**An absolute threshold fires on whatever comes last.** `pas_raw` is a *sum*
over prelim tokens, so it climbs as the answer lengthens. With an absolute cut,
every flagged token was `.\n\n`, `**`, or a meta-verb past step 90. Comparing
each token to a running mean and variance of the tokens before it fixed this —
flagged tokens became `side`, `Left`, `Right`, `traffic`, `lights`.

Note the two settings differ: cross-example detection at a fixed position
(PAS's own setting, and what the AUROC above measures) does **not** want this
correction and is hurt by it. Within one answer it is required.

**Dividing straight through by the baseline crops the sky.** A third of the
patches in a typical photo (423 of 1247, measured) draw near-zero baseline
attention, so the ratio explodes exactly where the model looks least. Flooring
the denominator at 25% of its mean moved crops on a four-traffic-light scene
from 2-of-9 on the lights to 5-of-9, and eliminated the empty-sky crops.

## What is deliberately not implemented

**SAR, faithfully.** Real SAR relevance needs a finished sequence, an external
cross-encoder, and several sampled generations. None of that is available at the
token you are about to emit. The plan was to substitute an attention-shape
proxy — function words like "of"/"and" were expected to dump their attention on
the immediately preceding token. **Measured across 16 layers, that separation is
0.01–0.04 and flips sign between layers.** Three alternatives (image mass,
prelim entropy, top-1 probability) reached at best Cohen's d ≈ 0.58 — far too
weak for a hard gate. `trigger.py` uses a stopword list instead, which is
reliable and unglamorous. `local_ratio` is still logged if you want to revisit it.

**AFIP's mitigation.** AFIP corrects by overwriting pre-softmax attention in
place, at zero extra forward passes — much cheaper than crop-and-reask. It is a
large separate change across many layers' attention math, and it is the natural
next step once detection is shown to fire on the right tokens.

## Honest status

- **Detection is weak, and not measurably better than a free baseline.** On 228
  labelled RH-Bench items, scoring the answer token:

  | signal | AUROC |
  |---|---|
  | `gap_1_2` (ECRD's existing trigger) | 0.587 |
  | `top1_prob` | 0.591 |
  | `entropy` | 0.594 |
  | `pas_raw` @ 18 | **0.614** |
  | PAS + inverted gap | 0.624 |

  Paired bootstrap, PAS vs gap: **+0.028, 95% CI [−0.070, +0.120]**. The
  interval straddles zero — on this test PAS buys nothing over the
  distributional signal you already get for free.

  Caveats that could hide a real effect: the PAS paper scores object mentions in
  generated captions, while this scores a multiple-choice answer token; n=228
  with 59 positives is underpowered; and the causal one-token offset (see
  `logits_processor.py`) may cost PAS accuracy.

  **What PAS gives that the gap cannot** — and the actual reason for this
  architecture — is a *location*. The gap is a scalar; it can say the model is
  unsure but never where to look. The attention map is what makes the crop
  possible.
- **Accuracy is unproven.** On 12 TreeBench items, base and PASC both scored
  8.3% (1/12). Every one of the 12 answers *changed* — so the evidence bank is
  doing something — but it fixed one question and broke another. Far too small
  and too hard a slice to conclude anything; run a real one.
- **Known failure mode: PASC can break the output format.** On TreeBench
  index=8 the model wrote `Answer: A</think>` instead of `<answer>A</answer>`,
  so the parser scored it wrong regardless of the content. No token was forced
  there -- the evidence reweighting alone did it, because "Answer" looks like a
  content word to `is_groundable`. Any benchmark that parses a tagged answer is
  exposed to this. Lowering `max_mix_alpha` reduces the pressure; properly
  fixing it needs the caller to declare its format vocabulary.
- **Cost is real.** ~6 crops per TreeBench question, each a full extra
  `generate()` — roughly 28s/question against ~8s for base. `cooldown` and
  `z_thresh` are the knobs for that budget.
- **Every threshold is uncalibrated** unless its docstring in `config.py` cites
  a measurement. Read that file before trusting a number.

## Files

| File | What it does |
|---|---|
| `config.py` | Every tunable, each with a note on whether it was measured or guessed. Read this first. |
| `attn_probe.py` | Per-step attention rows from two layers, without `output_attentions`. |
| `rel_attention.py` | Baseline attention, relative map, crop box. Vendors `bbox_from_att_image_adaptive` from mllms_know (MIT). |
| `trigger.py` | Running z-score gate plus the stopword filter. |
| `self_correct.py` | Crop, re-ask the base model, parse choice + evidence. |
| `scorer.py` | Evidence bank and per-candidate consistency cost. |
| `logits_processor.py` | Ties it together, one decoding step at a time. Owns `correction_log`. |
| `pipeline.py` | `pasc_generate(...)` — the one call the demo and the eval both use. |

## Running it

```bash
# one image, with the flagged-token table
bash scripts/run/run_pasc.sh image.jpg "your question"

# TreeBench, with per-question crops / corrections / evidence in the JSON
python scripts/test/eval_treebench_pasc.py --pasc --limit 50
python scripts/test/eval_treebench_pasc.py --base --limit 50

# measure and log, but never crop -- separates the cost of measuring
# from the effect of correcting
python scripts/test/eval_treebench_pasc.py --measure-only --limit 50
```

### Result-file fields

Per question, in `details[]`:

| Field | Meaning |
|---|---|
| `n_crops` | How many times a crop-and-reask fired. |
| `n_applied` | How many of those actually changed the token. Usually far fewer — a correction may only apply a token the model itself was seriously considering (inside the knee); beyond that, alternatives carry near-zero probability and forcing one produces wreckage like `Look` → newline. |
| `corrections[]` | Per firing: `step`, `z`, `token` (original), `chosen`, `applied`, `bbox`. |
| `evidence_added[]` | Every evidence sentence the crops produced, in order. |
| `n_steps` | Tokens generated, for the crops-per-token rate. |

In `summary`: `accuracy`, `mean_iou`, `crops_total`, `crops_applied`,
`crops_per_question`, `evidence_total`, and the full `config` used.

## Relation to `ecrd/`

`pasc/` imports nothing from `ecrd/` — that directory is the prior work, to be
replaced by an upstream clone for baseline comparison, and shared code would
break that. Two things are deliberately different: the trigger is visual rather
than purely distributional, and the corrector reuses the loaded base model
rather than loading a separate fine-tuned 3B that never actually crops.
