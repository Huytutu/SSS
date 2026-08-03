"""Sanity checks for build_calibration.py using synthetic data — no GPU/model needed.
Run with: python scripts/calib/test_build_calibration.py
"""
from MORAI.SSS.scripts.calib.build_calibration import question_confidence, regret_upper_bound, calibrate_delta, join_never_always


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

    # regret_upper_bound: no data at this threshold -> can't vouch for it.
    check("no values -> upper bound 1.0", regret_upper_bound([], 0.1) == 1.0)

    # regret_upper_bound: bound is always >= empirical mean (margin >= 0).
    ucb = regret_upper_bound([1.0] * 3 + [0.0] * 97, 0.1)
    check("bound >= empirical mean", ucb >= 3 / 100)

    # regret_upper_bound: more data at the same empirical mean -> tighter bound.
    ucb_small_n = regret_upper_bound([1.0] * 10 + [0.0] * 90, 0.1)
    ucb_large_n = regret_upper_bound([1.0] * 100 + [0.0] * 900, 0.1)
    check(f"more data tightens the bound (n=100: {ucb_small_n:.4f} >= n=1000: {ucb_large_n:.4f})",
          ucb_small_n >= ucb_large_n)

    # join_never_always: pairs by question_id, pulls calib_log from the never-verify side.
    never = [{"question_id": "a", "is_correct": False, "calib_log": [{"gap": 0.3}]}]
    always = [{"question_id": "a", "is_correct": True, "calib_log": [{"gap": 0.9}]}]
    joined = join_never_always(never, always)
    check("joined record uses never-run calib_log", joined[0]["calib_log"] == [{"gap": 0.3}])
    check("joined record keeps both outcomes", joined[0] == {
        "question_id": "a", "calib_log": [{"gap": 0.3}],
        "is_correct_never": False, "is_correct_always": True,
    })

    # calibrate_delta: three populations chosen to mirror a realistic high-error
    # benchmark (~69% baseline accuracy overall) while still letting a tight alpha
    # (0.05) find a valid delta -- this is exactly the case where the old absolute-risk
    # calibration was structurally stuck at delta=None no matter how much data you fed
    # it, because a ~70%-wrong population can never satisfy a small absolute-risk bound.
    #
    #   "confident" (gap=0.5, N=8000): verification wouldn't have changed the outcome
    #       either way (is_correct_always == is_correct_never), 75% already correct.
    #   "semi-ambiguous" (gap=0.1, N=2000): half were wrong unverified but fixed by
    #       verification; the other half were already correct regardless.
    #   "ambiguous" (gap=0.02, N=200): always wrong unverified, always fixed by
    #       verification.
    #
    # Large N is deliberate: it shrinks the Hoeffding margin enough for the true
    # regret (which is exactly 0 once delta reaches 0.1, since every question with any
    # regret has been touched) to clear alpha=0.05.
    records = []
    for i in range(8000):
        correct = i < 6000  # 75% correct, verification doesn't change it
        records.append({
            "question_id": f"confident-{i}",
            "calib_log": [{"gap": 0.5}],
            "is_correct_never": correct, "is_correct_always": correct,
        })
    for i in range(2000):
        records.append({
            "question_id": f"semi-{i}",
            "calib_log": [{"gap": 0.1}],
            "is_correct_never": i >= 1000,  # half wrong unverified
            "is_correct_always": True,      # verification fixes all of them
        })
    for i in range(200):
        records.append({
            "question_id": f"ambiguous-{i}",
            "calib_log": [{"gap": 0.02}],
            "is_correct_never": False,
            "is_correct_always": True,
        })

    summary = calibrate_delta(records, risk_alpha=0.05, conf_delta=0.1)
    check(f"baseline accuracy is ~69% (got {summary['never_accuracy']})",
          abs(summary["never_accuracy"] - 7000 / 10200) < 1e-4)
    check(f"delta is not null at alpha=0.05 (got {summary['delta']})", summary["delta"] is not None)
    check(f"chosen delta == 0.1 (got {summary['delta']})", summary["delta"] == 0.1)

    # calibrate_delta: no data at all -> no valid delta, must not crash.
    empty_summary = calibrate_delta([], risk_alpha=0.1, conf_delta=0.1)
    check("no records -> delta is None", empty_summary["delta"] is None)

    print("\nAll build_calibration sanity checks passed.")


if __name__ == "__main__":
    main()
