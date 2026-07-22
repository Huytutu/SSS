"""Build a Conformal-ECRD calibration set from logged baseline runs (Task #4 + #5).

Input: a JSONL file, one line per question, each shaped like:
    {"question_id": ..., "is_correct": bool, "calib_log": [
        {"step": int, "cand_ids": [int, ...], "cand_mix_probs": [float, ...], "chosen_id": int},
        ...
    ]}
`calib_log` is exactly what ECRDLogitsProcessor.calib_log accumulates per question when
constructed with collect_calibration_log=True (see ecrd/logits_processor.py).

Pipeline (Strategy 2 — correct-chain proxy, see project notes):
  1. Keep only questions with is_correct=True.
  2. For every gate-eligible step in those questions, the token the system actually chose
     (chosen_id) is treated as the pseudo ground-truth label.
  3. Compute the APS non-conformity score kappa = cumulative p_mix mass, in descending
     order, up to and including chosen_id's rank.
  4. Compute q_hat for one or more target alpha values from the pooled kappa values.

Usage:
    python scripts/calib/build_calibration.py --input calib_runs.jsonl --alpha 0.05 0.1 0.2
"""
from __future__ import annotations
import argparse
import json
import math
from typing import List


def compute_kappa(cand_ids: List[int], cand_mix_probs: List[float], chosen_id: int):
    """Cumulative p_mix mass (descending order) up to and including chosen_id's rank.
    Returns None if chosen_id is not among cand_ids (should not happen under Strategy 2,
    since chosen_id is always the argmax that produced these very candidates — see notes
    on why the mixing formula guarantees this; kept as a defensive check for when this
    script is later reused with an externally supplied label, e.g. Strategy 3).
    """
    if chosen_id not in cand_ids:
        return None
    order = sorted(range(len(cand_ids)), key=lambda j: cand_mix_probs[j], reverse=True)
    cum = 0.0
    for rank, j in enumerate(order, start=1):
        cum += cand_mix_probs[j]
        if cand_ids[j] == chosen_id:
            return cum
    return None  # unreachable


def q_hat_from_kappas(kappas: List[float], alpha: float) -> float:
    """Standard split-CP quantile: the ceil((N+1)(1-alpha))/N order statistic of kappas."""
    n = len(kappas)
    if n == 0:
        raise ValueError("no calibration points")
    rank = math.ceil((n + 1) * (1 - alpha))
    if rank > n:
        return 1.0  # not enough calibration data at this alpha: fall back to "always include everything"
    return sorted(kappas)[rank - 1]


def build_calibration(records: List[dict], alphas: List[float]) -> dict:
    kept_questions = [r for r in records if r.get("is_correct")]
    discarded_wrong_answer = len(records) - len(kept_questions)

    kappas = []
    discarded_not_in_cands = 0
    for q in kept_questions:
        for step in q.get("calib_log", []):
            kappa = compute_kappa(step["cand_ids"], step["cand_mix_probs"], step["chosen_id"])
            if kappa is None:
                discarded_not_in_cands += 1
                continue
            kappas.append(kappa)

    q_hats = {alpha: q_hat_from_kappas(kappas, alpha) for alpha in alphas}

    return {
        "n_questions_total": len(records),
        "n_questions_kept": len(kept_questions),
        "n_questions_discarded_wrong_answer": discarded_wrong_answer,
        "n_steps_total": len(kappas) + discarded_not_in_cands,
        "n_steps_kept": len(kappas),
        "n_steps_discarded_not_in_cands": discarded_not_in_cands,
        "q_hat": q_hats,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="JSONL file, one line per question")
    ap.add_argument("--alpha", type=float, nargs="+", default=[0.05, 0.1, 0.2])
    ap.add_argument("--output", default=None, help="Optional path to write the summary JSON")
    args = ap.parse_args()

    records = []
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    summary = build_calibration(records, args.alpha)
    print(json.dumps(summary, indent=2))

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"Saved summary to {args.output}")


if __name__ == "__main__":
    main()
