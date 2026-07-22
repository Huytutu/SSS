"""One-off fixup: retroactively regrade "reasoning" entries that were auto-marked wrong
by a grading bug (fixed upstream in eval_rhbench.py) before this script existed.

The bug: reasoning questions without lettered (A)(B)(C)(D) choices (free-response,
e.g. numeric answers) got correct_answer_option="" and were unconditionally scored
is_correct=False, regardless of whether the model's actual answer was right.

This script re-grades only the affected entries (category=="reasoning" and
ground_truth_option=="") using the same local LLM-judge approach eval_rhbench.py now
uses for free-response reasoning questions, WITHOUT re-running full generation —
prediction_raw is already saved in the results JSON, and grading is a cheap
text-only judge call (no image, max_new_tokens=10).

Usage:
    python scripts/calib/regrade_reasoning.py \
        --results results/rhbench_qwen2.5-vl-7b-instruct_grit.json \
        --calib-log /path/to/calib_data_200.jsonl   # optional, patches is_correct there too
"""
from __future__ import annotations
import argparse
import json
import os
import re

import torch
from transformers import AutoProcessor, AutoModelForVision2Seq

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("HF_HOME", os.path.join(_PROJECT_ROOT, ".cache"))


def parse_free_response_prediction(text: str) -> str:
    m = re.search(r"<answer>\s*(.*?)\s*</answer>", text, flags=re.I | re.S)
    return m.group(1).strip() if m else text.strip()


@torch.inference_mode()
def local_grade_answer(model, processor, question: str, model_answer: str, ground_truth: str) -> bool:
    if not model_answer.strip():
        return False
    prompt = f"""Compare the model's answer with the correct answer for the given question.
Question: {question}
Model's Answer: {model_answer}
Correct Answer: {ground_truth}

Is the model's answer correct and consistent with the correct answer? Respond with only "Correct" or "Incorrect"."""
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    chat = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[chat], padding=True, return_tensors="pt").to(model.device)
    out = model.generate(**inputs, do_sample=False, max_new_tokens=10, use_cache=True)
    output_text = processor.batch_decode(out[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0].strip()
    return "correct" in output_text.lower() and "incorrect" not in output_text.lower()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, help="results/rhbench_*.json to regrade in place")
    ap.add_argument("--calib-log", default=None, help="Optional calib_log JSONL to patch is_correct in")
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    args = ap.parse_args()

    with open(args.results, "r", encoding="utf-8") as f:
        data = json.load(f)

    affected = [d for d in data["details"] if d["category"] == "reasoning" and d.get("ground_truth_option", "") == ""]
    print(f"Found {len(affected)} affected reasoning entries out of {len(data['details'])} total.")
    if not affected:
        print("Nothing to regrade.")
        return

    print(f"Loading {args.model} for text-only judge calls...")
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        args.model, device_map="auto", load_in_4bit=True, trust_remote_code=True
    )

    flips = 0
    corrected_ids = {}
    for d in affected:
        old = d["is_correct"]
        pred = parse_free_response_prediction(d["prediction_raw"])
        new = local_grade_answer(model, processor, d["question"], pred, d["ground_truth"])
        d["prediction"] = pred
        d["is_correct"] = new
        corrected_ids[d["id"]] = new
        if new != old:
            flips += 1
    print(f"Regraded {len(affected)} entries, {flips} flipped from incorrect -> correct.")

    correct_reason = sum(1 for d in data["details"] if d["category"] == "reasoning" and d["is_correct"])
    total_reason = sum(1 for d in data["details"] if d["category"] == "reasoning")
    correct_perc = sum(1 for d in data["details"] if d["category"] == "perception" and d["is_correct"])
    total_perc = sum(1 for d in data["details"] if d["category"] == "perception")
    data["summary"]["reasoning"] = {"correct": correct_reason, "total": total_reason,
                                     "accuracy": correct_reason / total_reason if total_reason else 0}
    data["summary"]["perception"] = {"correct": correct_perc, "total": total_perc,
                                      "accuracy": correct_perc / total_perc if total_perc else 0}
    data["summary"]["rh_auc"] = data["summary"]["reasoning"]["accuracy"] * data["summary"]["perception"]["accuracy"]

    with open(args.results, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"Updated {args.results}")
    print(f"New reasoning accuracy: {correct_reason}/{total_reason} ({data['summary']['reasoning']['accuracy']*100:.2f}%)")
    print(f"Perception accuracy (unchanged): {correct_perc}/{total_perc} ({data['summary']['perception']['accuracy']*100:.2f}%)")

    if args.calib_log:
        # Keyed by (category, id): question_id numbering overlaps between categories
        # (both are 0-indexed independently), so keying by question_id alone would
        # silently patch the wrong category's record.
        corrected_by_key = {("reasoning", qid): val for qid, val in corrected_ids.items()}
        lines = []
        with open(args.calib_log, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                key = (rec["category"], rec["question_id"])
                if key in corrected_by_key:
                    rec["is_correct"] = corrected_by_key[key]
                lines.append(rec)
        with open(args.calib_log, "w", encoding="utf-8") as f:
            for rec in lines:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"Patched is_correct for {len(corrected_by_key)} (category, id) record(s) in {args.calib_log}")


if __name__ == "__main__":
    main()
