"""Calibrate a risk-controlled MixedGapTrigger threshold (delta) from logged runs.

Input: a JSONL file, one line per question, each shaped like:
    {"question_id": ..., "is_correct": bool, "calib_log": [
        {"step": int, "cand_ids": [...], "cand_mix_probs": [...], "chosen_id": int,
         "gap": float, "p1": float},
        ...
    ]}
`calib_log` is exactly what ECRDLogitsProcessor.calib_log accumulates per question when
constructed with collect_calibration_log=True (see ecrd/logits_processor.py).

Why this replaced the earlier APS-quantile approach: token-level conformal coverage
needs a true per-token label, and there is no such thing for free-form generation --
there's no annotated "correct" intermediate reasoning token. Quach et al., "Conformal
Language Modeling" (ICLR 2024), sidestep this for LMs by calibrating against an
*admission function* on the whole response (is it acceptable?) instead of a per-token
label. We adopt the same principle: is_correct (a real ground-truth comparison, already
computed by eval_rhbench.py) is the admission function. We calibrate a single scalar
decision threshold -- delta, the same gap_thresh MixedGapTrigger already takes -- against
it, using a Learn-Then-Test-style risk-control procedure (Angelopoulos et al., 2021)
instead of forcing token-level coverage with a proxy label.

Procedure:
  1. Per question, confidence(q) = min(gap) over every gate-eligible step -- a response
     is only as confident as its single most ambiguous moment. A question whose
     calib_log is empty (trigger conditions never arose) gets confidence(q) = 1.0
     (maximally confident by construction: nothing ever looked ambiguous).
  2. Candidate delta values = the sorted, unique confidence(q) scores observed -- these
     are the only values where the set of "would this question have been left alone"
     actually changes.
  3. For each candidate delta, look at the questions the trigger would have left alone
     (confidence(q) > delta, i.e. gap never dropped to/below delta) and compute a
     Hoeffding upper confidence bound on their wrong-rate.
  4. Every candidate is tested at a Bonferroni-corrected confidence level (conf_delta /
     number of candidates), so the guarantee holds regardless of which one is picked --
     this avoids assuming the bound is monotone in delta, which finite-sample noise can
     violate. The calibrated delta is the SMALLEST one whose bound is <= risk_alpha
     (cheapest trigger setting that still keeps residual risk in check).

This is a valid, dependency-free (no scipy) simplification of full Learn-Then-Test --
Hoeffding's inequality is looser than e.g. Hoeffding-Bentkus, so the chosen delta is
somewhat more conservative than an optimal LTT run would give, but the guarantee itself
is not an approximation.

Usage:
    python scripts/calib/build_calibration.py \
  --input calib_data.jsonl \
  --risk-alpha 0.2 0.3 0.4 \
  --conf-delta 0.1
"""
from __future__ import annotations
import argparse
import json
import math
from typing import List, Optional


def question_confidence(calib_log: List[dict]) -> float:
    """Smallest gap seen across a question's gate-eligible steps; 1.0 (max possible
    confidence) if the trigger conditions never arose at all."""
    gaps = [step["gap"] for step in calib_log if "gap" in step]
    return min(gaps) if gaps else 1.0


def hoeffding_upper_bound(n_wrong: int, n_total: int, conf_delta: float) -> float:
    """Upper (1 - conf_delta) confidence bound on a true Bernoulli rate, via Hoeffding's
    inequality. Looser than Clopper-Pearson but needs no special functions, so this
    project doesn't have to depend on scipy for a calibration script.
    """
    if n_total == 0:
        return 1.0  # no data at this threshold -- can't vouch for it, assume worst case
    p_hat = n_wrong / n_total
    margin = math.sqrt(math.log(1.0 / conf_delta) / (2.0 * n_total))
    return min(1.0, p_hat + margin)


def build_candidate_grid(all_confidences: List[float], max_candidates: Optional[int]) -> List[float]:
    """Coarsens the sorted, unique confidence scores down to at most max_candidates
    values (evenly spaced by rank), so the Bonferroni correction in calibrate_delta
    divides by a smaller number and each individual test gets a tighter bound. Testing
    every single observed value (max_candidates=None) is the most fine-grained choice
    of delta, but pays for it with the steepest Bonferroni penalty.
    """
    if max_candidates is None or len(all_confidences) <= max_candidates:
        return all_confidences
    n = len(all_confidences)
    ranks = sorted({round(i * (n - 1) / (max_candidates - 1)) for i in range(max_candidates)})
    return [all_confidences[r] for r in ranks]


def calibrate_delta(records: List[dict], risk_alpha: float, conf_delta: float,
                     max_candidates: Optional[int] = None) -> dict:
    """Finds the smallest MixedGapTrigger.gap_thresh (delta) such that, with confidence
    >= 1 - conf_delta, the true wrong-rate among questions the trigger would leave
    untouched (gap never dropped to/below delta) is <= risk_alpha.
    """
    scored = [
        (question_confidence(r.get("calib_log", [])), not r["is_correct"])
        for r in records
    ]
    all_confidences = sorted({c for c, _ in scored})
    candidates = build_candidate_grid(all_confidences, max_candidates)

    # Bonferroni correction across every threshold tested, so the overall guarantee
    # holds at conf_delta no matter which candidate ends up chosen -- this sidesteps
    # having to assume the bound is monotone in delta (finite-sample noise can violate
    # that: a larger delta leaves fewer questions untouched, so its bound can widen
    # even though the true residual risk is lower).
    per_test_conf_delta = conf_delta / max(len(candidates), 1)

    diagnostics = []
    valid_deltas = []
    for delta in candidates:
        untouched_wrong = [wrong for c, wrong in scored if c > delta]
        n_total = len(untouched_wrong)
        n_wrong = sum(untouched_wrong)
        ucb = hoeffding_upper_bound(n_wrong, n_total, per_test_conf_delta)
        passes = ucb <= risk_alpha
        diagnostics.append({
            "delta": delta,
            "n_untouched": n_total,
            "n_wrong": n_wrong,
            "wrong_rate_ucb": round(ucb, 4),
            "passes": passes,
        })
        if passes:
            valid_deltas.append(delta)

    chosen_delta = min(valid_deltas) if valid_deltas else None

    return {
        "risk_alpha": risk_alpha,
        "conf_delta": conf_delta,
        "n_questions_total": len(records),
        "n_candidates_tested": len(candidates),
        "delta": chosen_delta,
        "diagnostics": diagnostics,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="JSONL file, one line per question")
    ap.add_argument("--risk-alpha", type=float, nargs="+", default=[0.1, 0.2],
                     help="Target max wrong-rate among questions the trigger leaves untouched")
    ap.add_argument("--conf-delta", type=float, default=0.1,
                     help="Statistical confidence level for the risk bound (smaller = safer, needs more calib data)")
    ap.add_argument("--max-candidates", type=int, default=None,
                     help="Coarsen the delta grid to at most this many values, to ease the "
                          "Bonferroni penalty when calib data is scarce (default: test every "
                          "observed value)")
    ap.add_argument("--output", default="results/find_delta.json", help="Optional path to write the summary JSON")
    args = ap.parse_args()

    records = []
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    summary = {
        alpha: calibrate_delta(records, alpha, args.conf_delta, args.max_candidates)
        for alpha in args.risk_alpha
    }
    print(json.dumps(summary, indent=2))

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"Saved summary to {args.output}")


if __name__ == "__main__":
    main()
