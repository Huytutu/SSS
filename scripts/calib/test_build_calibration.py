"""Sanity checks for build_calibration.py using synthetic data — no GPU/model needed.
Run with: python scripts/calib/test_build_calibration.py
"""
from build_calibration import compute_kappa, q_hat_from_kappas, build_calibration


def check(name, cond):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name}")
    assert cond, name


def main():
    # kappa: chosen_id is top-1 -> kappa equals its own probability.
    k = compute_kappa(cand_ids=[5, 2, 9], cand_mix_probs=[0.6, 0.3, 0.1], chosen_id=5)
    check("kappa for top-1 choice == its own prob", abs(k - 0.6) < 1e-9)

    # kappa: chosen_id is rank 2 -> cumulative of top-2.
    k = compute_kappa(cand_ids=[5, 2, 9], cand_mix_probs=[0.6, 0.3, 0.1], chosen_id=2)
    check("kappa for rank-2 choice == cumulative top-2", abs(k - 0.9) < 1e-9)

    # kappa: chosen_id not present -> None (defensive path).
    k = compute_kappa(cand_ids=[5, 2, 9], cand_mix_probs=[0.6, 0.3, 0.1], chosen_id=42)
    check("kappa is None when chosen_id absent", k is None)

    # q_hat monotonicity: smaller alpha (stricter guarantee) -> larger or equal q_hat.
    kappas = [0.2, 0.3, 0.35, 0.4, 0.5, 0.55, 0.6, 0.7, 0.8, 0.9]
    q_10 = q_hat_from_kappas(kappas, alpha=0.1)
    q_20 = q_hat_from_kappas(kappas, alpha=0.2)
    q_30 = q_hat_from_kappas(kappas, alpha=0.3)
    check(f"q_hat monotone as alpha shrinks (q10={q_10} >= q20={q_20} >= q30={q_30})", q_10 >= q_20 >= q_30)
    check("q_hat within (0, 1]", all(0 < q <= 1 for q in (q_10, q_20, q_30)))

    # end-to-end: build_calibration filters wrong-answer questions and pools steps.
    records = [
        {"question_id": "q1", "is_correct": True, "calib_log": [
            {"step": 3, "cand_ids": [5, 2, 9], "cand_mix_probs": [0.6, 0.3, 0.1], "chosen_id": 5},
            {"step": 8, "cand_ids": [1, 4], "cand_mix_probs": [0.55, 0.45], "chosen_id": 1},
        ]},
        {"question_id": "q2", "is_correct": False, "calib_log": [
            {"step": 2, "cand_ids": [7, 3], "cand_mix_probs": [0.5, 0.5], "chosen_id": 7},
        ]},
        {"question_id": "q3", "is_correct": True, "calib_log": []},
    ]
    summary = build_calibration(records, alphas=[0.1])
    check("n_questions_total == 3", summary["n_questions_total"] == 3)
    check("n_questions_kept == 2 (q2 dropped, wrong answer)", summary["n_questions_kept"] == 2)
    check("n_steps_kept == 2 (only q1 has steps, q3 has none)", summary["n_steps_kept"] == 2)
    check("q_hat dict has alpha=0.1 key", 0.1 in summary["q_hat"])

    print("\nAll build_calibration sanity checks passed.")


if __name__ == "__main__":
    main()
