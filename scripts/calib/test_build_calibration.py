"""Sanity checks for build_calibration.py using synthetic data — no GPU/model needed.
Run with: python scripts/calib/test_build_calibration.py
"""
from build_calibration import question_confidence, hoeffding_upper_bound, calibrate_delta


def check(name, cond):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name}")
    assert cond, name


def main():
    # question_confidence: empty calib_log -> maximally confident (1.0).
    check("empty calib_log -> confidence 1.0", question_confidence([]) == 1.0)

    # question_confidence: smallest gap across steps wins.
    log = [{"gap": 0.5}, {"gap": 0.1}, {"gap": 0.3}]
    check("confidence == min(gap) over steps", abs(question_confidence(log) - 0.1) < 1e-9)

    # hoeffding_upper_bound: no data at this threshold -> can't vouch for it.
    check("n_total == 0 -> upper bound 1.0", hoeffding_upper_bound(0, 0, 0.1) == 1.0)

    # hoeffding_upper_bound: bound is always >= empirical rate (margin >= 0).
    ucb = hoeffding_upper_bound(n_wrong=3, n_total=100, conf_delta=0.1)
    check("bound >= empirical rate", ucb >= 3 / 100)

    # hoeffding_upper_bound: more data at the same empirical rate -> tighter bound.
    ucb_small_n = hoeffding_upper_bound(n_wrong=10, n_total=100, conf_delta=0.1)
    ucb_large_n = hoeffding_upper_bound(n_wrong=100, n_total=1000, conf_delta=0.1)
    check(f"more data tightens the bound (n=100: {ucb_small_n:.4f} >= n=1000: {ucb_large_n:.4f})",
          ucb_small_n >= ucb_large_n)

    # calibrate_delta: two clearly separated populations --
    # 1000 "confident" questions (gap 0.5) that are always correct, and
    # 20 "ambiguous" questions (gap 0.02) that are always wrong.
    # A properly calibrated delta should land at 0.02: leaving the confident group
    # alone (untouched) is safe, and the ambiguous group ends up triggered instead of
    # counted in the untouched pool at all.
    records = []
    for i in range(1000):
        records.append({
            "question_id": f"confident-{i}", "is_correct": True,
            "calib_log": [{"step": 0, "gap": 0.5}],
        })
    for i in range(20):
        records.append({
            "question_id": f"ambiguous-{i}", "is_correct": False,
            "calib_log": [{"step": 0, "gap": 0.02}],
        })

    summary = calibrate_delta(records, risk_alpha=0.15, conf_delta=0.1)
    check(f"chosen delta == 0.02 (got {summary['delta']})", summary["delta"] == 0.02)
    check("n_questions_total == 1020", summary["n_questions_total"] == 1020)

    # calibrate_delta: no data at all -> no valid delta, must not crash.
    empty_summary = calibrate_delta([], risk_alpha=0.1, conf_delta=0.1)
    check("no records -> delta is None", empty_summary["delta"] is None)

    print("\nAll build_calibration sanity checks passed.")


if __name__ == "__main__":
    main()
