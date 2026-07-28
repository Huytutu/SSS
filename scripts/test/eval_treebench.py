#!/usr/bin/env python
"""Evaluation script for TreeBench using ECRD.

Usage:
  python scripts/test/eval_treebench.py --model Qwen/Qwen2.5-VL-7B-Instruct --use-grit
"""
import argparse
import ast
import os
import re
import json
from datetime import datetime
import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, LogitsProcessorList
from qwen_vl_utils import process_vision_info

from ecrd import Evidence, EvidenceScorer, ECRDLogitsProcessor, MixedGapTrigger, GRITClient
from ecrd.prompts import GLOBAL_DESCRIPTION_PROMPT

# System prompt and answer-instruction suffix, verbatim from the TreeBench reference
# implementation (github.com/Haochen-Wang409/TreeVGR/blob/main/inference_treebench.py),
# so a model's <think>/<answer>/<box> formatting is directly comparable to published results.
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

def build_messages(image, question: str, min_pixels: int, max_pixels: int):
    return [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image, "min_pixels": min_pixels, "max_pixels": max_pixels},
                {"type": "text", "text": question + ANSWER_INSTRUCTION},
            ],
        },
    ]

def compute_box_iou(predict_str: str, target_boxes: list) -> float:
    """Average IoU between <box>[x1,y1,x2,y2]</box> boxes in the prediction and the
    ground-truth target_instances, exactly as in the TreeBench reference implementation's
    compute_box_iou (inference_treebench.py): for each target box, take the best IoU
    among all predicted boxes, then average over targets.
    """
    pattern = r"<box>(.*?)</box>"
    matches = re.findall(pattern, predict_str, re.DOTALL)

    pred_boxes = []
    for match in matches:
        coord_match = re.match(r"\[(\d+),(\d+),(\d+),(\d+)\]", match.strip())
        if coord_match:
            x1, y1, x2, y2 = map(int, coord_match.groups())
            if x1 < x2 and y1 < y2:
                pred_boxes.append([x1, y1, x2, y2])

    if not target_boxes:
        return 0.0

    def iou(box1, box2):
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2
        inter_w = max(0, min(x1_max, x2_max) - max(x1_min, x2_min))
        inter_h = max(0, min(y1_max, y2_max) - max(y1_min, y2_min))
        inter_area = inter_w * inter_h
        area1 = (x1_max - x1_min) * (y1_max - y1_min)
        area2 = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = area1 + area2 - inter_area
        return inter_area / union_area if union_area > 0 else 0.0

    total_iou = 0.0
    for target in target_boxes:
        total_iou += max((iou(target, pred) for pred in pred_boxes), default=0.0)
    return total_iou / len(target_boxes)

@torch.inference_mode()
def generate_global_description(model, processor, image, question: str, min_pixels: int, max_pixels: int) -> str:
    prompt = GLOBAL_DESCRIPTION_PROMPT.format(instruction=question)
    messages = [{"role": "user", "content": [{"type": "image", "image": image, "min_pixels": min_pixels, "max_pixels": max_pixels}, {"type": "text", "text": prompt}]}]
    chat = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    img_inputs, vid_inputs = process_vision_info(messages)
    inputs = processor(text=[chat], images=img_inputs, videos=vid_inputs, padding=True, return_tensors="pt").to(model.device)
    out = model.generate(**inputs, do_sample=False, max_new_tokens=128, use_cache=True)
    return processor.batch_decode(out[:, inputs.input_ids.shape[1]:], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()

def extract_answer(text: str) -> str:
    # Matches the reference implementation's extraction exactly: content of the
    # <answer> tag, upper-cased, falling back to the raw output if the tag is absent.
    match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    return match.group(1).strip().upper() if match else text.strip().upper()

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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="weights/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--use-grit", action="store_true")
    ap.add_argument("--no-supervisor", action="store_true",
                     help="Skip the ECRD supervisor (negotiated reweighting) entirely and run plain "
                          "decoding -- the paper's raw Base row. Incompatible with --use-grit, "
                          "since the visual decider is only invoked through the supervisor's trigger.")
    ap.add_argument("--grit-model", default="weights/GRIT-20-Qwen2.5-VL-3B")
    ap.add_argument("--delta", type=float, default=0.08)
    ap.add_argument("--collect-calib-log", default=None,
                     help="If set, append one JSONL record per question (question_id, is_correct, calib_log) "
                          "to this path, for use with scripts/calib/build_calibration.py")
    ap.add_argument("--load-in-4bit", action="store_true", help="Load base model in 4-bit")
    ap.add_argument("--grit-in-4bit", action="store_true", help="Load GRIT model in 4-bit")
    ap.add_argument("--grit-device", default="cpu", help="Device to run GRIT model on (e.g., cpu, 0)")
    ap.add_argument("--max-new-tokens", type=int, default=1024, help="Reference implementation default is 1024")
    ap.add_argument("--min-pixels", type=int, default=256*28*28)
    ap.add_argument("--max-pixels", type=int, default=1280*28*28)
    ap.add_argument("--limit", type=int, default=None, help="Limit evaluation to first N samples")
    ap.add_argument("--data-dir", default=None, help="Directory for local dataset (default: data/TreeBench)")
    ap.add_argument("--output-dir", default="results", help="Directory to save evaluation results JSON")
    args = ap.parse_args()
    if args.no_supervisor and args.use_grit:
        ap.error("--no-supervisor and --use-grit are incompatible: the visual decider is only "
                  "invoked through the supervisor's trigger.")

    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    os.environ["HF_HOME"] = os.path.join(project_root, ".cache")

    data_dir = args.data_dir or os.path.join(project_root, "data", "TreeBench")
    tsv_file = os.path.join(data_dir, "TreeBench.tsv") if os.path.isdir(data_dir) else data_dir

    if os.path.exists(tsv_file) and os.path.isfile(tsv_file):
        print(f"Loading local TreeBench dataset from: {tsv_file}")
        import pandas as pd
        df = pd.read_csv(tsv_file, sep="\t")
        dataset = df.to_dict(orient="records")
    else:
        print(f"Loading dataset from Hugging Face Hub: HaochenWang/TreeBench")
        # TreeBench splits: all 405 examples are in the 'train' split
        try:
            dataset = load_dataset("HaochenWang/TreeBench", split="train")
        except Exception:
            try:
                dataset = load_dataset("HaochenWang/TreeBench")["train"]
            except Exception:
                dataset = load_dataset("HaochenWang/TreeBench")
                first_split = list(dataset.keys())[0]
                dataset = dataset[first_split]

    if args.limit:
        if isinstance(dataset, list):
            dataset = dataset[:min(args.limit, len(dataset))]
        else:
            dataset = dataset.select(range(min(args.limit, len(dataset))))

    model_path = resolve_local_model_path(args.model, project_root)
    is_local_model = os.path.isdir(model_path)

    model_kwargs = {
        "device_map": "auto",
        "trust_remote_code": True,
    }
    if is_local_model:
        print(f"Loading local base model weights from: {model_path}")
        model_kwargs["local_files_only"] = True
    else:
        print(f"Initializing model from Hub: {model_path}")

    processor = AutoProcessor.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=is_local_model
    )

    if args.load_in_4bit:
        model_kwargs["load_in_4bit"] = True
    else:
        model_kwargs["torch_dtype"] = torch.bfloat16

    try:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            attn_implementation="flash_attention_2",
            **model_kwargs
        ).eval()
    except ImportError:
        print("flash_attn is not installed. Falling back to sdpa attention implementation...")
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            attn_implementation="sdpa",
            **model_kwargs
        ).eval()

    grit = None
    if args.use_grit:
        grit_path = resolve_local_model_path(args.grit_model, project_root)
        print(f"Initializing GRIT model from: {grit_path}")
        grit = GRITClient(
            model_id=grit_path,
            device=args.grit_device,
            torch_dtype=torch.bfloat16,
            load_in_4bit=args.grit_in_4bit
        )


    # Categories tracking
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
        "Reasoning/Comparison"
    ]
    stats = {sub: {"correct": 0, "total": 0, "iou_sum": 0.0} for sub in subcategories}
    other_stats = {}
    
    # Detailed logs
    details = []

    pbar = tqdm(dataset, desc="Evaluating TreeBench")
    for row in pbar:
        image = row.get("image")
        question_text = row.get("question")
        options_text = row.get("multi-choice options")
        ground_truth = str(row.get("answer", "")).strip().upper()
        category = row.get("category", "")
        idx = int(row.get("index", len(details)))

        try:
            target_boxes = ast.literal_eval(row.get("target_instances") or "[]")
        except (ValueError, SyntaxError):
            target_boxes = []

        if not question_text or not image:
            continue

        # Decode base64 image string to PIL Image
        if isinstance(image, str):
            try:
                import base64
                import io
                from PIL import Image as PILImage
                img_bytes = base64.b64decode(image)
                image = PILImage.open(io.BytesIO(img_bytes)).convert("RGB")
            except Exception as e:
                print(f"Error decoding base64 image for index {idx}: {e}")
                continue

        # Combine question with options, matching the reference implementation's
        # " Options:\n" separator (inference_treebench.py's eval_model_row).
        full_question = question_text
        if options_text:
            full_question += " Options:\n" + options_text

        try:
            logits_processors = []
            proc = None
            if not args.no_supervisor:
                desc = generate_global_description(model, processor, image, full_question, args.min_pixels, args.max_pixels)
                scorer = EvidenceScorer(model=model, tokenizer=processor.tokenizer, max_prefix_len=128)
                scorer.add_evidence(Evidence(id="global-0", text=desc, source="global", time_step=0))

                proc = ECRDLogitsProcessor(
                    scorer=scorer, tokenizer=processor.tokenizer, min_k=1, max_k=64,
                    collect_calibration_log=bool(args.collect_calib_log),
                )

                if grit:
                    def grit_hook(*args, **kwargs):
                        return grit.decide_next_token(*args, **kwargs, max_new_tokens=64)
                    proc.set_grit_runtime(
                        hook=grit_hook,
                        trigger=MixedGapTrigger(gap_thresh=args.delta, min_k=2, cooldown=5),
                        evidence_pool=scorer,
                        question=full_question,
                        image=image,
                    )
                logits_processors.append(proc)

            messages = build_messages(image, full_question, args.min_pixels, args.max_pixels)
            chat = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            # Prefill the reasoning tag, matching the reference implementation's `text += "<think>"`.
            chat += "<think>"
            img_inputs, vid_inputs = process_vision_info(messages)
            inputs = processor(text=[chat], images=img_inputs, videos=vid_inputs, padding=True, return_tensors="pt").to(model.device)

            gen = model.generate(
                **inputs,
                do_sample=True,
                top_p=0.001,
                top_k=1,
                temperature=0.01,
                repetition_penalty=1.0,
                use_cache=True,
                max_new_tokens=args.max_new_tokens,
                logits_processor=LogitsProcessorList(logits_processors),
            )
            prediction_text = processor.batch_decode(gen[:, inputs.input_ids.shape[1]:], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

            pred_ans = extract_answer(prediction_text)

            is_correct = (pred_ans == ground_truth)
            box_iou = compute_box_iou(prediction_text, target_boxes)

            if args.collect_calib_log and grit:
                with open(args.collect_calib_log, "a", encoding="utf-8") as f:
                    f.write(json.dumps({
                        "question_id": idx,
                        "category": category,
                        "is_correct": is_correct,
                        "calib_log": proc.calib_log,
                    }, ensure_ascii=False) + "\n")

            # Record statistics
            if category in stats:
                stats[category]["total"] += 1
                stats[category]["iou_sum"] += box_iou
                if is_correct:
                    stats[category]["correct"] += 1
            else:
                if category not in other_stats:
                    other_stats[category] = {"correct": 0, "total": 0, "iou_sum": 0.0}
                other_stats[category]["total"] += 1
                other_stats[category]["iou_sum"] += box_iou
                if is_correct:
                    other_stats[category]["correct"] += 1

            # Record details log
            details.append({
                "index": idx,
                "question": question_text,
                "options": options_text,
                "prediction": pred_ans,
                "ground_truth": ground_truth,
                "is_correct": is_correct,
                "iou": box_iou,
                "category": category,
                "prediction_text": prediction_text,
                "grit_invocations": proc.grit_invocations if grit else None,
            })

            # Running overall stats
            correct_overall = sum(c["correct"] for c in stats.values()) + sum(c["correct"] for c in other_stats.values())
            total_overall = sum(c["total"] for c in stats.values()) + sum(c["total"] for c in other_stats.values())
            accuracy = correct_overall / total_overall if total_overall > 0 else 0.0
            
            pbar.set_postfix(accuracy=f"{accuracy:.2%}", correct=correct_overall, total=total_overall)
        except Exception as ex:
            print(f"Error evaluating row: {ex}")
            continue

    print("\n" + "="*70)
    print("TreeBench Evaluation Results Summary")
    print("="*70)

    # 1. Perception Category Group
    print("\n--- [Perception Categories] ---")
    perception_correct = 0
    perception_total = 0
    perception_summary = {}
    for sub in subcategories:
        if sub.startswith("Perception/"):
            cat_label = sub.split("/")[-1]
            correct = stats[sub]["correct"]
            total = stats[sub]["total"]
            acc = correct / total if total > 0 else 0.0
            mean_iou = stats[sub]["iou_sum"] / total if total > 0 else 0.0
            print(f"  {cat_label:<25}: {correct:>3}/{total:<3} ({acc:.2%})  IoU={mean_iou:.2%}")
            perception_correct += correct
            perception_total += total
            perception_summary[cat_label] = {"correct": correct, "total": total, "accuracy": acc, "mean_iou": mean_iou}
    perc_overall_acc = perception_correct / perception_total if perception_total > 0 else 0.0
    print(f"  {'Perception Overall':<25}: {perception_correct:>3}/{perception_total:<3} ({perc_overall_acc:.2%})")

    # 2. Reasoning Category Group
    print("\n--- [Reasoning Categories] ---")
    reasoning_correct = 0
    reasoning_total = 0
    reasoning_summary = {}
    for sub in subcategories:
        if sub.startswith("Reasoning/"):
            cat_label = sub.split("/")[-1]
            correct = stats[sub]["correct"]
            total = stats[sub]["total"]
            acc = correct / total if total > 0 else 0.0
            mean_iou = stats[sub]["iou_sum"] / total if total > 0 else 0.0
            print(f"  {cat_label:<25}: {correct:>3}/{total:<3} ({acc:.2%})  IoU={mean_iou:.2%}")
            reasoning_correct += correct
            reasoning_total += total
            reasoning_summary[cat_label] = {"correct": correct, "total": total, "accuracy": acc, "mean_iou": mean_iou}
    reas_overall_acc = reasoning_correct / reasoning_total if reasoning_total > 0 else 0.0
    print(f"  {'Reasoning Overall':<25}: {reasoning_correct:>3}/{reasoning_total:<3} ({reas_overall_acc:.2%})")

    # 3. Other unexpected categories (if any)
    if other_stats:
        print("\n--- [Other Categories] ---")
        for sub, info in other_stats.items():
            correct = info["correct"]
            total = info["total"]
            acc = correct / total if total > 0 else 0.0
            print(f"  {sub:<25}: {correct:>3}/{total:<3} ({acc:.2%})")

    # 4. Final Overall Score
    overall_correct = perception_correct + reasoning_correct + sum(c["correct"] for c in other_stats.values())
    overall_total = perception_total + reasoning_total + sum(c["total"] for c in other_stats.values())
    overall_acc = overall_correct / overall_total if overall_total > 0 else 0.0
    # Mean IoU exactly as the reference implementation reports it: np.mean over every
    # item's box IoU, regardless of category.
    overall_mean_iou = float(np.mean([d["iou"] for d in details])) if details else 0.0
    print("\n" + "-"*70)
    print(f"{'Overall Summary Score':<27}: {overall_correct:>3}/{overall_total:<3} ({overall_acc:.2%})")
    print(f"{'Mean IoU':<27}: {overall_mean_iou:.2%}")
    print("="*70)

    # Save to JSON
    out_dir = args.output_dir
    if not os.path.isabs(out_dir):
        out_dir = os.path.join(project_root, out_dir)
    os.makedirs(out_dir, exist_ok=True)
    
    output_data = {
        "metadata": {
            "model": args.model,
            "use_grit": args.use_grit,
            "no_supervisor": args.no_supervisor,
            "grit_model": args.grit_model if args.use_grit else None,
            "delta": args.delta,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "total_samples": overall_total,
            "total_grit_invocations": sum(d["grit_invocations"] or 0 for d in details) if args.use_grit else None,
        },
        "summary": {
            "overall": {
                "correct": overall_correct,
                "total": overall_total,
                "accuracy": overall_acc,
                "mean_iou": overall_mean_iou
            },
            "perception": {
                "overall": {
                    "correct": perception_correct,
                    "total": perception_total,
                    "accuracy": perc_overall_acc
                },
                "subcategories": perception_summary
            },
            "reasoning": {
                "overall": {
                    "correct": reasoning_correct,
                    "total": reasoning_total,
                    "accuracy": reas_overall_acc
                },
                "subcategories": reasoning_summary
            },
            "others": {
                sub: {
                    "correct": info["correct"],
                    "total": info["total"],
                    "accuracy": info["correct"] / info["total"] if info["total"] > 0 else 0.0
                }
                for sub, info in other_stats.items()
            }
        },
        "details": details
    }
    
    clean_model_name = args.model.split("/")[-1].lower()
    mode = "grit" if args.use_grit else ("base" if args.no_supervisor else "supervisor")
    output_file = os.path.join(out_dir, f"treebench_{clean_model_name}_{mode}.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    print(f"Detailed results saved to: {output_file}")

if __name__ == "__main__":
    main()
