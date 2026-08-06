#!/usr/bin/env python
"""Evaluate TreeBench with the base Qwen2.5-VL decoder, VDGD decoding, or LeCo-on-VDGD.

All three decoders come from SSS.inference (qwen_base / qwen_vdgd / qwen_leco) --
this script only adds dataset loading, category bookkeeping, and JSON reporting
around them.

Usage:
  python eval/eval_treebench_qwen.py --method base
  python eval/eval_treebench_qwen.py --method vdgd --min-k 1 --max-k 64
  python eval/eval_treebench_qwen.py --method leco --min-k 1 --max-k 64 --max-iters 3
"""
import argparse
import base64
import io
import json
import os
import re
from datetime import datetime

import pandas as pd
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

from SSS.inference import load_qwen, qwen_base, qwen_vdgd, qwen_leco


def extract_answer(text: str) -> str:
    # TreeBench's ground-truth `answer` column is always a single option
    # letter (A-F). qwen_base/qwen_vdgd's prompt doesn't force the model to
    # answer with only that letter, so it typically writes e.g.
    # "<answer>C. Front right</answer>" -- comparing that whole string
    # against a bare "C" would mark an otherwise-correct answer wrong.
    # Pull out the leading option letter instead.
    match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    answer_text = match.group(1).strip() if match else text.strip()

    letter_match = re.match(r"([A-Za-z])\b", answer_text)
    if letter_match:
        return letter_match.group(1).upper()
    return answer_text.upper()


def resolve_local_model_path(model_path_or_id: str, project_root: str) -> str:
    if os.path.isdir(model_path_or_id):
        return os.path.abspath(model_path_or_id)
    rel_path = os.path.join(project_root, model_path_or_id)
    if os.path.isdir(rel_path):
        return rel_path
    basename = os.path.basename(model_path_or_id)
    weights_path = os.path.join(project_root, "weights", basename)
    if os.path.isdir(weights_path):
        return weights_path
    return model_path_or_id


def load_treebench(data_dir: str, limit: int = None):
    tsv_file = os.path.join(data_dir, "TreeBench.tsv") if os.path.isdir(data_dir) else data_dir
    if os.path.exists(tsv_file) and os.path.isfile(tsv_file):
        print(f"Loading local TreeBench dataset from: {tsv_file}")
        df = pd.read_csv(tsv_file, sep="\t")
        dataset = df.to_dict(orient="records")
    else:
        print("Loading dataset from Hugging Face Hub: HaochenWang/TreeBench")
        try:
            dataset = load_dataset("HaochenWang/TreeBench", split="train")
        except Exception:
            dataset = load_dataset("HaochenWang/TreeBench")["train"]

    if limit:
        dataset = dataset[:limit] if isinstance(dataset, list) else dataset.select(range(min(limit, len(dataset))))
    return dataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["base", "vdgd", "leco"], required=True,
                    help="Which SSS.inference decoder to evaluate: qwen_base, qwen_vdgd, or qwen_leco")
    ap.add_argument("--model", default="weights/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--min-k", type=int, default=None, help="VDGD knee-truncation floor (vdgd/leco only)")
    ap.add_argument("--max-k", type=int, default=None, help="VDGD knee-truncation ceiling (vdgd/leco only)")
    ap.add_argument("--max-iters", type=int, default=3, help="LeCo max rollback-and-regenerate rounds (leco only)")
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--min-pixels", type=int, default=256 * 28 * 28)
    ap.add_argument("--max-pixels", type=int, default=1280 * 28 * 28)
    ap.add_argument("--limit", type=int, default=None, help="Limit evaluation to first N samples")
    ap.add_argument("--data-dir", default=None, help="Directory for local dataset (default: data/TreeBench)")
    ap.add_argument("--output-dir", default="results", help="Directory to save evaluation results JSON")
    args = ap.parse_args()

    # .../SSS (two dirname calls up from eval/eval_treebench_qwen.py)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.environ.setdefault("HF_HOME", os.path.join(project_root, ".cache"))

    data_dir = args.data_dir or os.path.join(project_root, "data", "TreeBench")
    dataset = load_treebench(data_dir, args.limit)

    model_path = resolve_local_model_path(args.model, project_root)
    print(f"Loading model from: {model_path}")
    model, processor = load_qwen(model_path)

    subcategories = [
        "Perception/Attributes",
        "Perception/Material",
        "Perception/Physical State",
        "Perception/Object Retrieval",
        "Perception/OCR",
        "Reasoning/Perspective Transform",
        "Reasoning/Ordering",
        "Reasoning/Contact and Occlusion",
        "Reasoning/Spatial Containment",
        "Reasoning/Comparison",
    ]
    stats = {sub: {"correct": 0, "total": 0} for sub in subcategories}
    other_stats = {}
    details = []

    pbar = tqdm(dataset, desc=f"Evaluating TreeBench ({args.method})")
    for row in pbar:
        image = row.get("image")
        question_text = row.get("question")
        options_text = row.get("multi-choice options")
        ground_truth = str(row.get("answer", "")).strip().upper()
        category = row.get("category", "")
        idx = int(row.get("index", len(details)))

        if not question_text or not image:
            continue

        if isinstance(image, str):
            try:
                image = Image.open(io.BytesIO(base64.b64decode(image))).convert("RGB")
            except Exception as e:
                print(f"Error decoding base64 image for index {idx}: {e}")
                continue

        # Matches eval_treebench.py's " Options:\n" separator.
        full_question = question_text
        if options_text:
            full_question += " Options:\n" + options_text

        try:
            n_iters = None
            if args.method == "base":
                prediction_text = qwen_base(
                    image, full_question, model=model, processor=processor,
                    min_pixels=args.min_pixels, max_pixels=args.max_pixels,
                    max_new_tokens=args.max_new_tokens,
                )
            elif args.method == "vdgd":
                prediction_text, _description = qwen_vdgd(
                    image, full_question, model=model, processor=processor,
                    min_pixels=args.min_pixels, max_pixels=args.max_pixels,
                    min_k=args.min_k, max_k=args.max_k,
                    max_new_tokens=args.max_new_tokens,
                )
            else:
                prediction_text, _description, n_iters = qwen_leco(
                    image, full_question, model=model, processor=processor,
                    min_pixels=args.min_pixels, max_pixels=args.max_pixels,
                    min_k=args.min_k, max_k=args.max_k,
                    max_new_tokens=args.max_new_tokens, max_iters=args.max_iters,
                )

            pred_ans = extract_answer(prediction_text)
            is_correct = (pred_ans == ground_truth)

            if category in stats:
                stats[category]["total"] += 1
                if is_correct:
                    stats[category]["correct"] += 1
            else:
                other_stats.setdefault(category, {"correct": 0, "total": 0})
                other_stats[category]["total"] += 1
                if is_correct:
                    other_stats[category]["correct"] += 1

            details.append({
                "index": idx,
                "question": question_text,
                "options": options_text,
                "prediction": pred_ans,
                "ground_truth": ground_truth,
                "is_correct": is_correct,
                "category": category,
                "prediction_text": prediction_text,
                "n_iters": n_iters,
            })

            correct_overall = sum(c["correct"] for c in stats.values()) + sum(c["correct"] for c in other_stats.values())
            total_overall = sum(c["total"] for c in stats.values()) + sum(c["total"] for c in other_stats.values())
            accuracy = correct_overall / total_overall if total_overall > 0 else 0.0
            pbar.set_postfix(accuracy=f"{accuracy:.2%}", correct=correct_overall, total=total_overall)
        except Exception as ex:
            print(f"Error evaluating row {idx}: {ex}")
            continue

    print("\n" + "=" * 70)
    print(f"TreeBench Evaluation Results Summary ({args.method})")
    print("=" * 70)

    print("\n--- [Perception Categories] ---")
    perception_correct = perception_total = 0
    perception_summary = {}
    for sub in subcategories:
        if sub.startswith("Perception/"):
            cat_label = sub.split("/")[-1]
            correct, total = stats[sub]["correct"], stats[sub]["total"]
            acc = correct / total if total > 0 else 0.0
            print(f"  {cat_label:<25}: {correct:>3}/{total:<3} ({acc:.2%})")
            perception_correct += correct
            perception_total += total
            perception_summary[cat_label] = {"correct": correct, "total": total, "accuracy": acc}
    perc_overall_acc = perception_correct / perception_total if perception_total > 0 else 0.0
    print(f"  {'Perception Overall':<25}: {perception_correct:>3}/{perception_total:<3} ({perc_overall_acc:.2%})")

    print("\n--- [Reasoning Categories] ---")
    reasoning_correct = reasoning_total = 0
    reasoning_summary = {}
    for sub in subcategories:
        if sub.startswith("Reasoning/"):
            cat_label = sub.split("/")[-1]
            correct, total = stats[sub]["correct"], stats[sub]["total"]
            acc = correct / total if total > 0 else 0.0
            print(f"  {cat_label:<25}: {correct:>3}/{total:<3} ({acc:.2%})")
            reasoning_correct += correct
            reasoning_total += total
            reasoning_summary[cat_label] = {"correct": correct, "total": total, "accuracy": acc}
    reas_overall_acc = reasoning_correct / reasoning_total if reasoning_total > 0 else 0.0
    print(f"  {'Reasoning Overall':<25}: {reasoning_correct:>3}/{reasoning_total:<3} ({reas_overall_acc:.2%})")

    if other_stats:
        print("\n--- [Other Categories] ---")
        for sub, info in other_stats.items():
            acc = info["correct"] / info["total"] if info["total"] > 0 else 0.0
            print(f"  {sub:<25}: {info['correct']:>3}/{info['total']:<3} ({acc:.2%})")

    overall_correct = perception_correct + reasoning_correct + sum(c["correct"] for c in other_stats.values())
    overall_total = perception_total + reasoning_total + sum(c["total"] for c in other_stats.values())
    overall_acc = overall_correct / overall_total if overall_total > 0 else 0.0
    print("\n" + "-" * 70)
    print(f"{'Overall Summary Score':<27}: {overall_correct:>3}/{overall_total:<3} ({overall_acc:.2%})")
    print("=" * 70)

    iter_counts = [d["n_iters"] for d in details if d["n_iters"] is not None]
    avg_iters = sum(iter_counts) / len(iter_counts) if iter_counts else None

    out_dir = args.output_dir
    if not os.path.isabs(out_dir):
        out_dir = os.path.join(project_root, out_dir)
    os.makedirs(out_dir, exist_ok=True)

    output_data = {
        "metadata": {
            "model": args.model,
            "method": args.method,
            "min_k": args.min_k,
            "max_k": args.max_k,
            "max_new_tokens": args.max_new_tokens,
            "max_iters": args.max_iters if args.method == "leco" else None,
            "avg_iters": avg_iters,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "total_samples": overall_total,
        },
        "summary": {
            "overall": {"correct": overall_correct, "total": overall_total, "accuracy": overall_acc},
            "perception": {
                "overall": {"correct": perception_correct, "total": perception_total, "accuracy": perc_overall_acc},
                "subcategories": perception_summary,
            },
            "reasoning": {
                "overall": {"correct": reasoning_correct, "total": reasoning_total, "accuracy": reas_overall_acc},
                "subcategories": reasoning_summary,
            },
            "others": {
                sub: {
                    "correct": info["correct"],
                    "total": info["total"],
                    "accuracy": info["correct"] / info["total"] if info["total"] > 0 else 0.0,
                }
                for sub, info in other_stats.items()
            },
        },
        "details": details,
    }

    clean_model_name = args.model.split("/")[-1].lower()
    output_file = os.path.join(out_dir, f"treebench_{clean_model_name}_{args.method}.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    print(f"Detailed results saved to: {output_file}")


if __name__ == "__main__":
    main()
