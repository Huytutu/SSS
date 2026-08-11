#!/usr/bin/env python
"""Run PASC on one image and show what it flagged.

  python examples/pasc_demo.py --image path/to.jpg --question "..." --dump-flagged
"""
from __future__ import annotations

import argparse
import os
import sys

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pasc import PASConfig, pasc_generate

ANSWER_INSTRUCTION = "\n\nThink step by step, then put the final answer in <answer>...</answer>."


def build_messages(image, question, max_pixels):
    return [{"role": "user", "content": [
        {"type": "image", "image": image, "max_pixels": max_pixels},
        {"type": "text", "text": question + ANSWER_INSTRUCTION},
    ]}]


def print_flagged(proc, top_n=25):
    """Tokens the model was least visually grounded on, and what happened to them."""
    # Ranked by whatever the trigger is actually gating on, so the table shows
    # the tokens that came closest to firing.
    uncertainty_first = proc.cfg.gate_mode in ("uncertainty", "both")
    log = sorted(proc.step_log, key=(lambda r: r["gap"]) if uncertainty_first else (lambda r: -r["z"]))
    log = [r for r in log if r["groundable"]][:top_n]

    print(f"\ngate_mode={proc.cfg.gate_mode}   (most uncertain first)" if uncertainty_first
          else f"\ngate_mode={proc.cfg.gate_mode}   (least grounded first)")
    print(f"{'step':>5} {'token':>16} {'gap':>7} {'k':>3} {'p_top':>7} {'z':>6} {'pas_raw':>8} {'fired':>6}")
    print("-" * 68)
    for r in log:
        print(f"{r['step']:>5} {r['top_token'][:16]:>16} {r['gap']:7.3f} {r['k']:>3} "
              f"{r['top_prob']:7.3f} {r['z']:6.2f} {r['pas_raw']:8.3f} "
              f"{'YES' if r['fired'] else '':>6}")

    applied = sum(c["applied"] for c in proc.correction_log)
    print(f"\ncorrections: {proc.n_corrections} crops, {applied} changed the token")
    for c in proc.correction_log:
        outcome = (f"{c['original_token']} -> {c['chosen_token']}" if c["applied"]
                   else f"{c['original_token']} (kept)")
        print(f"  step {c['step']:>4}  z={c['z']:.2f}  {outcome}")
        print(f"            crop={c['bbox']}  knee={c['n_knee']}/{c['n_candidates']} shown")
        print(f"            evidence: {c['evidence'] or '(none)'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="weights/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--image", required=True)
    ap.add_argument("--question", required=True)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--max-pixels", type=int, default=1280 * 28 * 28)
    ap.add_argument("--tau", type=float, default=None, help="Override PASConfig.tau")
    ap.add_argument("--pas-layer", type=int, default=None, help="Override PASConfig.pas_layer")
    ap.add_argument("--no-correct", action="store_true", help="Measure and log, but never crop")
    ap.add_argument("--dump-flagged", action="store_true")
    args = ap.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = args.model if os.path.isdir(args.model) else os.path.join(root, args.model)

    processor = AutoProcessor.from_pretrained(model_path, local_files_only=os.path.isdir(model_path))
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="cuda:0",
        attn_implementation="sdpa", local_files_only=os.path.isdir(model_path)).eval()

    cfg = PASConfig()
    if args.tau is not None:
        cfg.tau = args.tau
    if args.pas_layer is not None:
        cfg.pas_layer = args.pas_layer

    image = Image.open(args.image).convert("RGB")
    text, proc = pasc_generate(
        model, processor, image, build_messages(image, args.question, args.max_pixels),
        cfg=cfg, max_new_tokens=args.max_new_tokens, max_pixels=args.max_pixels,
        correct=not args.no_correct,
    )

    print("\n=== ANSWER ===")
    print(text)
    if args.dump_flagged:
        print_flagged(proc)


if __name__ == "__main__":
    main()
