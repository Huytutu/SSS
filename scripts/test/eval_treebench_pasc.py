#!/usr/bin/env python
"""TreeBench evaluation for PASC.

  python scripts/test/eval_treebench_pasc.py --pasc --limit 15
  python scripts/test/eval_treebench_pasc.py --base --limit 15

Writes a JSON with per-question accuracy and box IoU, plus -- for --pasc -- how
many crops each question triggered, what each correction did, and every evidence
sentence the crops produced.

The prompt, <think> prefill, answer extraction and IoU below follow TreeBench's
reference implementation so the accuracy is comparable to published numbers.
"""
from __future__ import annotations

import argparse
import ast
import base64
import io
import json
import os
import re
import sys
from datetime import datetime

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from pasc import PASConfig, pasc_generate

SYSTEM_PROMPT = (
    "A conversation between user and assistant. The user asks a question, and the "
    "Assistant solves it. The assistant MUST first think about the reasoning process "
    "in the mind and then provide the user with the answer. The reasoning process and "
    "answer are enclosed within <think> </think> and <answer> </answer> tags, "
    "respectively. When referring to particular objects in the reasoning process, the "
    "assistant MUST localize the object with bounding box coordinates between <box> "
    "and </box>. You MUST strictly follow the format."
)
ANSWER_INSTRUCTION = (
    "\nSelect the best answer to the above multiple-choice question based on the "
    "image. After the reasoning process, respond with only the letter of the correct "
    "option between <answer> and </answer>."
)

SUBCATEGORIES = [
    "Perception/Attributes", "Perception/Material", "Perception/Physical State",
    "Perception/Object Retrieval", "Perception/OCR", "Reasoning/Perspective Transform",
    "Reasoning/Ordering", "Reasoning/Contact and Occlusion",
    "Reasoning/Spatial Containment", "Reasoning/Comparison",
]


def build_messages(image, question, min_pixels, max_pixels):
    return [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": [
            {"type": "image", "image": image, "min_pixels": min_pixels, "max_pixels": max_pixels},
            {"type": "text", "text": question + ANSWER_INSTRUCTION},
        ]},
    ]


def extract_answer(text):
    match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    return match.group(1).strip().upper() if match else text.strip().upper()


def compute_box_iou(prediction, target_boxes):
    """Mean over ground-truth boxes of the best IoU against any predicted <box>."""
    if not target_boxes:
        return 0.0
    pred_boxes = []
    for raw in re.findall(r"<box>(.*?)</box>", prediction, re.DOTALL):
        coords = re.match(r"\[(\d+),(\d+),(\d+),(\d+)\]", raw.strip())
        if coords:
            x1, y1, x2, y2 = map(int, coords.groups())
            if x1 < x2 and y1 < y2:
                pred_boxes.append([x1, y1, x2, y2])

    def iou(a, b):
        w = max(0, min(a[2], b[2]) - max(a[0], b[0]))
        h = max(0, min(a[3], b[3]) - max(a[1], b[1]))
        inter = w * h
        union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
        return inter / union if union > 0 else 0.0

    return sum(max((iou(t, p) for p in pred_boxes), default=0.0) for t in target_boxes) / len(target_boxes)


def decode_image(value):
    if isinstance(value, str):
        return Image.open(io.BytesIO(base64.b64decode(value))).convert("RGB")
    return value.convert("RGB")


def load_treebench(data_dir):
    tsv = os.path.join(data_dir, "TreeBench.tsv") if os.path.isdir(data_dir) else data_dir
    if os.path.isfile(tsv):
        import pandas as pd
        return pd.read_csv(tsv, sep="\t").to_dict(orient="records")
    from datasets import load_dataset
    return load_dataset("HaochenWang/TreeBench", split="train")


@torch.inference_mode()
def run_base(model, processor, messages, max_new_tokens):
    from qwen_vl_utils import process_vision_info
    chat = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) + "<think>"
    imgs, vids = process_vision_info(messages)
    inputs = processor(text=[chat], images=imgs, videos=vids, padding=True,
                       return_tensors="pt").to(model.device)
    out = model.generate(**inputs, do_sample=False, use_cache=True, max_new_tokens=max_new_tokens)
    return processor.batch_decode(out[:, inputs["input_ids"].shape[1]:],
                                  skip_special_tokens=True)[0].strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="weights/Qwen2.5-VL-7B-Instruct")
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--base", action="store_true", help="Plain decoding, no probe, no corrections.")
    mode.add_argument("--pasc", action="store_true", help="PAS-gated cropping and evidence.")
    mode.add_argument("--measure-only", action="store_true",
                      help="Run the probe and log signals, but never crop -- isolates the "
                           "cost of measurement from the effect of correcting.")
    ap.add_argument("--tau", type=float, default=None, help="Override PASConfig.z_thresh")
    ap.add_argument("--pas-layer", type=int, default=None)
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--min-pixels", type=int, default=256 * 28 * 28)
    ap.add_argument("--max-pixels", type=int, default=1280 * 28 * 28)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--output-dir", default="results")
    args = ap.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    os.environ.setdefault("HF_HOME", os.path.join(root, ".cache"))

    dataset = load_treebench(args.data_dir or os.path.join(root, "data", "TreeBench"))
    if args.limit:
        dataset = dataset[:args.limit] if isinstance(dataset, list) else dataset.select(range(args.limit))

    model_path = args.model if os.path.isdir(args.model) else os.path.join(root, args.model)
    local = os.path.isdir(model_path)
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=local)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="cuda:0",
        attn_implementation="sdpa", local_files_only=local).eval()

    cfg = PASConfig()
    if args.tau is not None:
        cfg.z_thresh = args.tau
    if args.pas_layer is not None:
        cfg.pas_layer = args.pas_layer

    mode_name = "base" if args.base else ("pasc" if args.pasc else "measure-only")
    stats = {sub: {"correct": 0, "total": 0, "iou_sum": 0.0} for sub in SUBCATEGORIES}
    details = []

    for row in tqdm(dataset, desc=f"TreeBench [{mode_name}]"):
        question, image = row.get("question"), row.get("image")
        if not question or image is None:
            continue
        image = decode_image(image)
        category = row.get("category", "")
        ground_truth = str(row.get("answer", "")).strip().upper()
        index = int(row.get("index", len(details)))
        try:
            target_boxes = ast.literal_eval(row.get("target_instances") or "[]")
        except (ValueError, SyntaxError):
            target_boxes = []

        full_question = question + ("\n Options:\n" + row["multi-choice options"]
                                    if row.get("multi-choice options") else "")
        messages = build_messages(image, full_question, args.min_pixels, args.max_pixels)

        record = {"index": index, "question": question, "category": category,
                  "ground_truth": ground_truth}
        if args.base:
            prediction = run_base(model, processor, messages, args.max_new_tokens)
        else:
            prediction, proc = pasc_generate(
                model, processor, image, messages, cfg=cfg,
                max_new_tokens=args.max_new_tokens, prefill="<think>",
                max_pixels=args.max_pixels, correct=args.pasc,
            )
            record["n_crops"] = proc.n_corrections
            record["n_applied"] = sum(c["applied"] for c in proc.correction_log)
            record["corrections"] = [
                {"step": c["step"], "z": round(c["z"], 3),
                 "token": c["original_token"], "chosen": c["chosen_token"],
                 "applied": c["applied"], "bbox": c["bbox"]}
                for c in proc.correction_log
            ]
            record["evidence_added"] = [c["evidence"] for c in proc.correction_log if c["evidence"]]
            record["n_steps"] = len(proc.step_log)

        predicted = extract_answer(prediction)
        record["prediction"] = predicted
        record["prediction_text"] = prediction
        record["is_correct"] = predicted == ground_truth
        record["iou"] = compute_box_iou(prediction, target_boxes)
        details.append(record)

        bucket = stats.setdefault(category, {"correct": 0, "total": 0, "iou_sum": 0.0})
        bucket["total"] += 1
        bucket["iou_sum"] += record["iou"]
        bucket["correct"] += int(record["is_correct"])

    total = len(details)
    n_correct = sum(d["is_correct"] for d in details)
    summary = {
        "mode": mode_name,
        "model": model_path,
        "n": total,
        "accuracy": n_correct / total if total else 0.0,
        "mean_iou": sum(d["iou"] for d in details) / total if total else 0.0,
        "per_category": {k: v for k, v in stats.items() if v["total"]},
    }
    if not args.base:
        crops = sum(d.get("n_crops", 0) for d in details)
        summary["crops_total"] = crops
        summary["crops_applied"] = sum(d.get("n_applied", 0) for d in details)
        summary["crops_per_question"] = crops / total if total else 0.0
        summary["evidence_total"] = sum(len(d.get("evidence_added", [])) for d in details)
        summary["config"] = vars(cfg)

    os.makedirs(args.output_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(args.output_dir, f"treebench_pasc_{mode_name}_{stamp}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "details": details}, f, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2)[:1200])
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
