"""Standalone sanity check for ConformalTrigger — no model/GPU needed.

Verifies the APS cumulative-set logic against the two worked examples used to
motivate this trigger (see project notes): a diffuse distribution that the old
gap-rule misses, and a peaked distribution that correctly stays quiet.
Run with: python scripts/test/test_conformal_trigger.py
"""
import torch

from ecrd.triggers import ConformalTrigger


class FakeTokenizer:
    """Minimal stand-in: token 0 always looks like a word-start critical keyword."""
    def convert_ids_to_tokens(self, tid):
        return "Ġred"


def make_info(step, cand_mix_probs, k=None):
    k = k if k is not None else len(cand_mix_probs)
    return {
        "step": step,
        "k": k,
        "cand_ids": list(range(len(cand_mix_probs))),
        "cand_mix_probs": cand_mix_probs,
    }


def run_case(name, probs, q_hat, expect_trigger):
    trigger = ConformalTrigger(q_hat=q_hat, min_k=2, cooldown=5)
    info = make_info(step=10, cand_mix_probs=probs)
    fired = trigger(info, torch.zeros(1, 1, dtype=torch.long), FakeTokenizer())
    status = "PASS" if fired == expect_trigger else "FAIL"
    print(f"[{status}] {name}: probs={probs} q_hat={q_hat} -> fired={fired} (expected {expect_trigger})")
    assert fired == expect_trigger, f"{name} failed"


def main():
    # Diffuse distribution: top-1 is only 0.19, several close runners-up.
    # gap = 0.19-0.10 = 0.09 > delta(0.08) would NOT trigger under the old rule,
    # but cumulative mass needs 3 tokens to reach q_hat=0.35 -> should trigger.
    run_case("diffuse (gap-rule would miss this)", [0.19, 0.10, 0.09, 0.08, 0.07], q_hat=0.35, expect_trigger=True)

    # Peaked distribution: top-1 alone already clears q_hat -> singleton set, no trigger.
    run_case("peaked (confident)", [0.85, 0.08, 0.04, 0.02, 0.01], q_hat=0.35, expect_trigger=False)

    # Top-1 clearly above q_hat -> singleton, no trigger. (Not testing an exact
    # float boundary here: float32 round-tripping through torch makes exact
    # equality comparisons flaky, which is a property of floating point in
    # general, not a bug in the trigger.)
    run_case("top-1 clearly above q_hat", [0.40, 0.30, 0.20, 0.10], q_hat=0.35, expect_trigger=False)

    # min_k gate: only 1 candidate available (k*=1) -> never trigger regardless of q_hat.
    trigger = ConformalTrigger(q_hat=0.01, min_k=2, cooldown=5)
    info = make_info(step=10, cand_mix_probs=[0.99], k=1)
    fired = trigger(info, torch.zeros(1, 1, dtype=torch.long), FakeTokenizer())
    status = "PASS" if fired is False else "FAIL"
    print(f"[{status}] min_k gate blocks k=1: fired={fired} (expected False)")
    assert fired is False

    # Cooldown: firing twice within cooldown window should be blocked on the 2nd call.
    trigger = ConformalTrigger(q_hat=0.35, min_k=2, cooldown=5)
    info1 = make_info(step=10, cand_mix_probs=[0.19, 0.10, 0.09, 0.08, 0.07])
    first = trigger(info1, torch.zeros(1, 1, dtype=torch.long), FakeTokenizer())
    info2 = make_info(step=12, cand_mix_probs=[0.19, 0.10, 0.09, 0.08, 0.07])
    second = trigger(info2, torch.zeros(1, 1, dtype=torch.long), FakeTokenizer())
    status = "PASS" if (first is True and second is False) else "FAIL"
    print(f"[{status}] cooldown blocks re-fire at step+2: first={first} second={second}")
    assert first is True and second is False

    print("\nAll ConformalTrigger sanity checks passed.")


if __name__ == "__main__":
    main()
