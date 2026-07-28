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

def build_messages(image, question: str, min_pixels: int, max_pixels: int):
    return [{
        "role": "user",
        "content": [
            {"type": "image", "image": image, "min_pixels": min_pixels, "max_pixels": max_pixels},
            {"type": "text", "text": question + "\n\nThink step by step and put the final answer in <answer>...</answer>."},
        ],
    }]

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
    # Search for the letter anywhere inside the <answer> tag, since the model
    # often writes "(B) Mickey Mouse" rather than a bare letter right after the tag.
    tag_match = re.search(r"<answer>\s*(.*?)\s*</answer>", text, flags=re.I | re.S)
    if tag_match:
        letter_match = re.search(r"\b([A-D])\b", tag_match.group(1), flags=re.I)
        if letter_match:
            return letter_match.group(1).upper()
        return ""

    ans_last = re.search(r"\b([A-D])\b", text[-20:], flags=re.I)
    if ans_last:
        return ans_last.group(1).upper()
    return ""

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
                          "greedy decoding -- the paper's raw Base row. Incompatible with --use-grit, "
                          "since the visual decider is only invoked through the supervisor's trigger.")
    ap.add_argument("--grit-model", default="weights/GRIT-20-Qwen2.5-VL-3B")
    ap.add_argument("--delta", type=float, default=0.08)
    ap.add_argument("--collect-calib-log", default=None,
                     help="If set, append one JSONL record per question (question_id, is_correct, calib_log) "
                          "to this path, for use with scripts/calib/build_calibration.py")
    ap.add_argument("--load-in-4bit", action="store_true", help="Load base model in 4-bit")
    ap.add_argument("--grit-in-4bit", action="store_true", help="Load GRIT model in 4-bit")
    ap.add_argument("--grit-device", default="cpu", help="Device to run GRIT model on (e.g., cpu, 0)")
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--min-pixels", type=int, default=256*28*28)
    ap.add_argument("--max-pixels", type=int, default=1280*28*28)
    ap.add_argument("--limit", type=int, default=None, help="Limit evaluation to first N samples")
    ap.add_argument("--data-dir", default=None, help="Directory for local dataset (default: data/VStar)")
    ap.add_argument("--output-dir", default="results", help="Directory to save evaluation results JSON")
    args = ap.parse_args()
    if args.no_supervisor and args.use_grit:
        ap.error("--no-supervisor and --use-grit are incompatible: the visual decider is only "
                  "invoked through the supervisor's trigger.")

    # Set cache dir to .cache folder in project
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    os.environ["HF_HOME"] = os.path.join(project_root, ".cache")

    data_dir = args.data_dir or os.path.join(project_root, "data", "VStar")
    parquet_file = os.path.join(data_dir, "data", "test-00000-of-00001.parquet")
    if not os.path.exists(parquet_file):
        import glob
        parquets = glob.glob(os.path.join(data_dir, "*.parquet")) + glob.glob(os.path.join(data_dir, "**", "*.parquet"), recursive=True)
        if parquets:
            parquet_file = parquets[0]

    if os.path.exists(parquet_file):
        print(f"Loading local V*Bench dataset from: {parquet_file}")
        dataset = load_dataset("parquet", data_files={"test": parquet_file})["test"]
    else:
        print(f"Loading dataset from Hugging Face Hub: lmms-lab/vstar-bench")
        try:
            dataset = load_dataset("lmms-lab/vstar-bench", split="test")
        except Exception:
            dataset = load_dataset("lmms-lab/vstar-bench")["test"]

    if args.limit:
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

    # Dictionary to keep track of category stats
    cat_mapping = {
        "direct_attributes": "Attr",
        "relative_position": "Spatial"
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
        question = row.get("text", "")
        if question:
            question = re.sub(
                r"\n?Answer with the option's letter from the given choices directly\.?",
                "",
                question
            ).strip()
        ground_truth = str(row.get("label", "")).strip().upper()
        raw_category = row.get("category", "direct_attributes")
        category = cat_mapping.get(raw_category, "Attr")
        qid = str(row.get("question_id", len(details)))

        if not question or not image:
            continue

        try:
            logits_processors = []
            proc = None
            if not args.no_supervisor:
                desc = generate_global_description(model, processor, image, question, args.min_pixels, args.max_pixels)
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
                        question=question,
                        image=image,
                    )
                logits_processors.append(proc)

            messages = build_messages(image, question, args.min_pixels, args.max_pixels)
            chat = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            img_inputs, vid_inputs = process_vision_info(messages)
            inputs = processor(text=[chat], images=img_inputs, videos=vid_inputs, padding=True, return_tensors="pt").to(model.device)

            gen = model.generate(
                **inputs,
                do_sample=False,
                use_cache=True,
                max_new_tokens=args.max_new_tokens,
                logits_processor=LogitsProcessorList(logits_processors),
            )
            prediction_text = processor.batch_decode(gen[:, inputs.input_ids.shape[1]:], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            
            pred_ans = extract_answer(prediction_text)
            
            is_correct = (pred_ans == ground_truth)

            if args.collect_calib_log and grit:
                with open(args.collect_calib_log, "a", encoding="utf-8") as f:
                    f.write(json.dumps({
                        "question_id": qid,
                        "category": category,
                        "is_correct": is_correct,
                        "calib_log": proc.calib_log,
                    }, ensure_ascii=False) + "\n")

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
                "prediction_text": prediction_text,
                "grit_invocations": proc.grit_invocations if grit else None,
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
            "no_supervisor": args.no_supervisor,
            "grit_model": args.grit_model if args.use_grit else None,
            "delta": args.delta,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "total_samples": total_samples,
            "total_grit_invocations": sum(d["grit_invocations"] or 0 for d in details) if args.use_grit else None,
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
    mode = "grit" if args.use_grit else ("base" if args.no_supervisor else "supervisor")
    output_file = os.path.join(out_dir, f"vstar_{clean_model_name}_{mode}.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    print(f"Detailed results saved to: {output_file}")

if __name__ == "__main__":
    main()