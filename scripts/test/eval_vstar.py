#!/usr/bin/env python
"""Evaluation script for V* (V*Bench) using ECRD.

Usage:
  python scripts/test/eval_vstar.py --model Qwen/Qwen2.5-VL-7B-Instruct --use-grit
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
        ans_char = re.search(r"\b([A-D])\b", ans, flags=re.I)
        if ans_char:
            return ans_char.group(1).upper()
        return ans.upper()
    
    ans_char = re.search(r"\b([A-D])\b", text[-20:], flags=re.I)
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

    # Set cache dir to .cache folder in project
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    os.environ["HF_HOME"] = os.path.join(project_root, ".cache")

    print(f"Loading dataset: lmms-lab/vstar-bench")
    try:
        dataset = load_dataset("lmms-lab/vstar-bench", split="test")
    except Exception:
        dataset = load_dataset("lmms-lab/vstar-bench")["test"]

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

    # Dictionary to keep track of category stats
    cat_mapping = {
        "direct_attributes": "Attr",
        "direct_spatial_relations": "Spatial"
    }
    stats = {
        "Attr": {"correct": 0, "total": 0},
        "Spatial": {"correct": 0, "total": 0}
    }
    
    # Store detailed predictions
    details = []

    pbar = tqdm(dataset, desc="Evaluating V*Bench")
    for row in pbar:
        image = row.get("image")
        question = row.get("text")
        ground_truth = str(row.get("label", "")).strip().upper()
        raw_category = row.get("category", "direct_attributes")
        category = cat_mapping.get(raw_category, "Attr")
        qid = str(row.get("question_id", len(details)))

        if not question or not image:
            continue

        try:
            desc = generate_global_description(model, processor, image, question)
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
                    question=question,
                    image=image,
                )

            messages = build_messages(image, question)
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
            
            stats[category]["total"] += 1
            if is_correct:
                stats[category]["correct"] += 1
            
            # Save detail log
            details.append({
                "question_id": qid,
                "question": question,
                "prediction": pred_ans,
                "ground_truth": ground_truth,
                "is_correct": is_correct,
                "category": category,
                "prediction_text": prediction_text
            })
            
            # Compute running overall stats
            correct_overall = sum(c["correct"] for c in stats.values())
            total_overall = sum(c["total"] for c in stats.values())
            accuracy = correct_overall / total_overall if total_overall > 0 else 0.0
            
            pbar.set_postfix(accuracy=f"{accuracy:.2%}", correct=correct_overall, total=total_overall)
        except Exception as ex:
            print(f"Error evaluating row: {ex}")
            continue

    print("\n" + "="*50)
    print("V*Bench Evaluation Results Summary")
    print("="*50)
    
    total_correct = 0
    total_samples = 0
    for cat_name, info in stats.items():
        cat_correct = info["correct"]
        cat_total = info["total"]
        cat_acc = cat_correct / cat_total if cat_total > 0 else 0.0
        print(f"{cat_name:<10}: {cat_correct:>3}/{cat_total:<3} ({cat_acc:.2%})")
        total_correct += cat_correct
        total_samples += cat_total
        
    overall_acc = total_correct / total_samples if total_samples > 0 else 0.0
    print("-"*50)
    print(f"{'Overall':<10}: {total_correct:>3}/{total_samples:<3} ({overall_acc:.2%})")
    print("="*50)

    # Save to JSON
    # Absolute output directory path relative to project root if relative path given
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
            "total_samples": total_samples
        },
        "summary": {
            "overall": {
                "correct": total_correct,
                "total": total_samples,
                "accuracy": overall_acc
            },
            "categories": {
                cat_name: {
                    "correct": info["correct"],
                    "total": info["total"],
                    "accuracy": info["correct"] / info["total"] if info["total"] > 0 else 0.0
                }
                for cat_name, info in stats.items()
            }
        },
        "details": details
    }
    
    clean_model_name = args.model.split("/")[-1].lower()
    mode = "grit" if args.use_grit else "basic"
    output_file = os.path.join(out_dir, f"vstar_{clean_model_name}_{mode}.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    print(f"Detailed results saved to: {output_file}")

if __name__ == "__main__":
    main()
