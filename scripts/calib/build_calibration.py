"""Calibrate a risk-controlled MixedGapTrigger threshold (delta) from paired runs.

Input: two JSONL files, one line per question, produced by the SAME eval script on the
SAME dataset with two different trigger settings:
  --input-never:  MixedGapTrigger never fires (e.g. `--delta -1`, since gap is always
                  >= 0). This is the "never verify" baseline. Its calib_log carries the
                  gap signal we calibrate against, uncorrupted by any verification
                  having already happened.
  --input-always: MixedGapTrigger fires at every gate-eligible step (e.g. `--delta 1.0`,
                  since gap is always <= 1). This is the "always verify" ceiling.
Each line is shaped like:
    {"question_id": ..., "is_correct": bool, "calib_log": [
        {"step": int, "cand_ids": [...], "cand_mix_probs": [...], "chosen_id": int,
         "gap": float, "p1": float},
        ...
    ]}
`calib_log` is exactly what ECRDLogitsProcessor.calib_log accumulates per question when
constructed with collect_calibration_log=True (see ecrd/logits_processor.py); it is
recorded independently of whether the live trigger actually fired, so it is identical
signal in both runs -- only `is_correct` differs between the two files.

Why paired regret, not absolute risk: an earlier version of this script calibrated
delta by bounding the absolute wrong-rate among questions the trigger leaves untouched
(Angelopoulos et al., "Learn then Test", using a single mixed-policy run). That is
structurally unusable whenever the base model's wrong-rate is high (e.g. ~70% on
TreeBench): no delta can ever bring an already-~70%-wrong population below a sane alpha,
regardless of how much calibration data is collected, because the bounded quantity
(absolute error) never has room to shrink to a small alpha.

Instead we bound REGRET against the always-verify ceiling:
    regret(delta) = E[correct_always] - E[correct under policy(delta)]
For a question q with confidence(q) = min gap seen at its gate-eligible steps:
  - if confidence(q) > delta, the trigger never touches q -> policy uses is_correct_never
  - otherwise, the trigger touches q at least once -> policy uses is_correct_always
So regret(delta) = mean over untouched q of (is_correct_always(q) - is_correct_never(q)).
Each per-question term is bounded in [-1, 1], and we Hoeffding-bound the MEAN over the
full N (not a shrinking subset), so the bound can be driven arbitrarily small by
touching more questions (larger delta) -- unlike the absolute-risk version, this always
has room to pass for some delta, no matter how bad the base model is.

Procedure:
  1. Join the two files on question_id (both must cover the exact same questions).
  2. confidence(q) from the never-verify run's calib_log (see docstring above).
  3. Candidate delta values = the sorted, unique confidence(q) scores observed.
  4. For each candidate delta, per-question regret term = (is_correct_always(q) -
     is_correct_never(q)) if untouched (confidence(q) > delta) else 0, and Hoeffding-
     bound the mean of these N terms at a Bonferroni-corrected confidence level
     (conf_delta / number of candidates), so the guarantee holds regardless of which
     candidate is picked.
  5. The calibrated delta is the SMALLEST one whose bound is <= risk_alpha (cheapest
     trigger setting -- fewest verifications -- that still keeps residual regret small).

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


def regret_upper_bound(values: List[float], conf_delta: float) -> float:
    """Upper (1 - conf_delta) confidence bound on the mean of values bounded in
    [-1, 1], via Hoeffding's inequality.
    """
    n = len(values)
    if n == 0:
        return 1.0  # no data -- can't vouch for it, assume worst-case regret
    mean = sum(values) / n
    margin = math.sqrt(2.0 * math.log(1.0 / conf_delta) / n)  # range (b - a) = 2
    return min(1.0, mean + margin)


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


def join_never_always(never_records: List[dict], always_records: List[dict]) -> List[dict]:
    """Pairs each question's never-verify and always-verify outcome by question_id.
    The gap signal (calib_log) is taken from the never-verify run, since that is the
    only run whose generation trajectory wasn't already altered by verification.
    """
    always_by_id = {r["question_id"]: r for r in always_records}
    joined = []
    for never in never_records:
        qid = never["question_id"]
        if qid not in always_by_id:
            raise ValueError(f"question_id {qid!r} present in --input-never but missing from --input-always")
        joined.append({
            "question_id": qid,
            "calib_log": never.get("calib_log", []),
            "is_correct_never": bool(never["is_correct"]),
            "is_correct_always": bool(always_by_id[qid]["is_correct"]),
        })
    if len(joined) != len(always_records):
        raise ValueError("--input-never and --input-always cover a different number of questions")
    return joined


def calibrate_delta(records: List[dict], risk_alpha: float, conf_delta: float,
                     max_candidates: Optional[int] = None) -> dict:
    """Finds the smallest MixedGapTrigger.gap_thresh (delta) such that, with confidence
    >= 1 - conf_delta, the regret against the always-verify ceiling among questions the
    trigger leaves untouched (confidence(q) > delta) is <= risk_alpha.
    """
    scored = [
        (question_confidence(r["calib_log"]), r["is_correct_never"], r["is_correct_always"])
        for r in records
    ]
    all_confidences = sorted({c for c, _, _ in scored})
    candidates = build_candidate_grid(all_confidences, max_candidates)

    # Bonferroni correction across every threshold tested, so the overall guarantee
    # holds at conf_delta no matter which candidate ends up chosen.
    per_test_conf_delta = conf_delta / max(len(candidates), 1)

    diagnostics = []
    valid_deltas = []
    for delta in candidates:
        regret_terms = [
            (int(always) - int(never)) if conf > delta else 0.0
            for conf, never, always in scored
        ]
        n_untouched = sum(1 for conf, _, _ in scored if conf > delta)
        ucb = regret_upper_bound(regret_terms, per_test_conf_delta)
        passes = ucb <= risk_alpha
        diagnostics.append({
            "delta": delta,
            "n_untouched": n_untouched,
            "mean_regret": round(sum(regret_terms) / len(regret_terms), 4) if regret_terms else 0.0,
            "regret_ucb": round(ucb, 4),
            "passes": passes,
        })
        if passes:
            valid_deltas.append(delta)

    chosen_delta = min(valid_deltas) if valid_deltas else None
    n_total = len(records)

    return {
        "risk_alpha": risk_alpha,
        "conf_delta": conf_delta,
        "n_questions_total": n_total,
        "n_candidates_tested": len(candidates),
        "never_accuracy": round(sum(never for _, never, _ in scored) / n_total, 4) if n_total else None,
        "always_accuracy": round(sum(always for _, _, always in scored) / n_total, 4) if n_total else None,
        "delta": chosen_delta,
        "diagnostics": diagnostics,
    }


def _load_jsonl(path: str) -> List[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-never", required=True,
                     help="JSONL from a run where the trigger never fires (e.g. --delta -1)")
    ap.add_argument("--input-always", required=True,
                     help="JSONL from a run where the trigger always fires (e.g. --delta 1.0)")
    ap.add_argument("--risk-alpha", type=float, nargs="+", default=[0.05, 0.1],
                     help="Target max regret vs. always-verify among questions the trigger leaves untouched")
    ap.add_argument("--conf-delta", type=float, default=0.1,
                     help="Statistical confidence level for the risk bound (smaller = safer, needs more calib data)")
    ap.add_argument("--max-candidates", type=int, default=None,
                     help="Coarsen the delta grid to at most this many values, to ease the "
                          "Bonferroni penalty when calib data is scarce (default: test every "
                          "observed value)")
    ap.add_argument("--output", default="results/find_delta.json", help="Optional path to write the summary JSON")
    args = ap.parse_args()

    never_records = _load_jsonl(args.input_never)
    always_records = _load_jsonl(args.input_always)
    records = join_never_always(never_records, always_records)

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
