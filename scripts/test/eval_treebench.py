#!/usr/bin/env python
"""Evaluation script for TreeBench using ECRD.

Usage:
  python scripts/test/eval_treebench.py --model Qwen/Qwen2.5-VL-7B-Instruct --use-grit
"""
import argparse
import os
import re
import json
from datetime import datetime
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, LogitsProcessorList
from qwen_vl_utils import process_vision_info

from ecrd import Evidence, EvidenceScorer, ECRDLogitsProcessor, MixedGapTrigger, GRITClient
from ecrd.prompts import GLOBAL_DESCRIPTION_PROMPT

def build_messages(image, question: str):
    return [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": question + "\n\nThink step by step and put the final answer in <answer>...</answer>."},
        ],
    }]

@torch.inference_mode()
def generate_global_description(model, processor, image, question: str) -> str:
    prompt = GLOBAL_DESCRIPTION_PROMPT.format(instruction=question)
    messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}]
    chat = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    img_inputs, vid_inputs = process_vision_info(messages)
    inputs = processor(text=[chat], images=img_inputs, videos=vid_inputs, padding=True, return_tensors="pt").to(model.device)
    out = model.generate(**inputs, do_sample=False, max_new_tokens=128, use_cache=True)
    return processor.batch_decode(out[:, inputs.input_ids.shape[1]:], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()

def extract_answer(text: str) -> str:
    m = re.search(r"<answer>\s*(.*?)\s*</answer>", text, flags=re.S | re.I)
    if m:
        ans = m.group(1).strip()
        ans_char = re.search(r"\b([A-K])\b", ans, flags=re.I)
        if ans_char:
            return ans_char.group(1).upper()
        return ans.upper()
    
    ans_char = re.search(r"\b([A-K])\b", text[-20:], flags=re.I)
    if ans_char:
        return ans_char.group(1).upper()
    return ""

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--use-grit", action="store_true")
    ap.add_argument("--grit-model", default="yfan1997/GRIT-20-Qwen2.5-VL-3B")
    ap.add_argument("--delta", type=float, default=0.08)
    ap.add_argument("--load-in-4bit", action="store_true", help="Load base model in 4-bit")
    ap.add_argument("--grit-in-4bit", action="store_true", help="Load GRIT model in 4-bit")
    ap.add_argument("--grit-device", default="cpu", help="Device to run GRIT model on (e.g., cpu, 0)")
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--limit", type=int, default=None, help="Limit evaluation to first N samples")
    ap.add_argument("--output-dir", default="results", help="Directory to save evaluation results JSON")
    args = ap.parse_args()

    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    os.environ["HF_HOME"] = os.path.join(project_root, ".cache")

    print(f"Loading dataset: HaochenWang/TreeBench")
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
        dataset = dataset.select(range(min(args.limit, len(dataset))))

    print(f"Initializing model: {args.model}")
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    
    model_kwargs = {
        "device_map": "auto",
        "trust_remote_code": True,
    }
    if args.load_in_4bit:
        model_kwargs["load_in_4bit"] = True
    else:
        model_kwargs["torch_dtype"] = torch.bfloat16

    try:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.model,
            attn_implementation="flash_attention_2",
            **model_kwargs
        ).eval()
    except ImportError:
        print("flash_attn is not installed. Falling back to sdpa attention implementation...")
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.model,
            attn_implementation="sdpa",
            **model_kwargs
        ).eval()

    grit = None
    if args.use_grit:
        print(f"Initializing GRIT model: {args.grit_model}")
        grit = GRITClient(
            model_id=args.grit_model,
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
    stats = {sub: {"correct": 0, "total": 0} for sub in subcategories}
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

        # Combine question with options
        full_question = question_text
        if options_text:
            full_question += "\n" + options_text

        try:
            desc = generate_global_description(model, processor, image, full_question)
            scorer = EvidenceScorer(model=model, tokenizer=processor.tokenizer, max_prefix_len=128)
            scorer.add_evidence(Evidence(id="global-0", text=desc, source="global", time_step=0))
            
            proc = ECRDLogitsProcessor(scorer=scorer, tokenizer=processor.tokenizer, min_k=1, max_k=64)

            if grit:
                def grit_hook(img, q, prefix_text, candidates):
                    return grit.decide_next_token(
                        image=img,
                        question=q,
                        prefix_text=prefix_text,
                        candidates=candidates,
                        max_new_tokens=64,
                    )
                proc.set_grit_runtime(
                    hook=grit_hook,
                    trigger=MixedGapTrigger(gap_thresh=args.delta, min_k=2, cooldown=5),
                    evidence_pool=scorer,
                    question=full_question,
                    image=image,
                )

            messages = build_messages(image, full_question)
            chat = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            img_inputs, vid_inputs = process_vision_info(messages)
            inputs = processor(text=[chat], images=img_inputs, videos=vid_inputs, padding=True, return_tensors="pt").to(model.device)

            gen = model.generate(
                **inputs,
                do_sample=False,
                use_cache=True,
                max_new_tokens=args.max_new_tokens,
                logits_processor=LogitsProcessorList([proc]),
            )
            prediction_text = processor.batch_decode(gen[:, inputs.input_ids.shape[1]:], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            
            pred_ans = extract_answer(prediction_text)
            
            is_correct = (pred_ans == ground_truth)
            
            # Record statistics
            if category in stats:
                stats[category]["total"] += 1
                if is_correct:
                    stats[category]["correct"] += 1
            else:
                if category not in other_stats:
                    other_stats[category] = {"correct": 0, "total": 0}
                other_stats[category]["total"] += 1
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
                "category": category,
                "prediction_text": prediction_text
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
            print(f"  {cat_label:<25}: {correct:>3}/{total:<3} ({acc:.2%})")
            perception_correct += correct
            perception_total += total
            perception_summary[cat_label] = {"correct": correct, "total": total, "accuracy": acc}
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
            print(f"  {cat_label:<25}: {correct:>3}/{total:<3} ({acc:.2%})")
            reasoning_correct += correct
            reasoning_total += total
            reasoning_summary[cat_label] = {"correct": correct, "total": total, "accuracy": acc}
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
    print("\n" + "-"*70)
    print(f"{'Overall Summary Score':<27}: {overall_correct:>3}/{overall_total:<3} ({overall_acc:.2%})")
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
            "grit_model": args.grit_model if args.use_grit else None,
            "delta": args.delta,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "total_samples": overall_total
        },
        "summary": {
            "overall": {
                "correct": overall_correct,
                "total": overall_total,
                "accuracy": overall_acc
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
    mode = "grit" if args.use_grit else "basic"
    output_file = os.path.join(out_dir, f"treebench_{clean_model_name}_{mode}.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    print(f"Detailed results saved to: {output_file}")

if __name__ == "__main__":
    main()
