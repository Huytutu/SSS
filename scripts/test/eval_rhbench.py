import os
import json
import re
import math
import argparse
import datetime
from tqdm import tqdm
from PIL import Image
import torch
import torch.nn.functional as F
from datasets import load_dataset
from huggingface_hub import hf_hub_download

from transformers import AutoProcessor, AutoModelForVision2Seq
from qwen_vl_utils import process_vision_info

# ECRD imports
from ecrd.scorer import Evidence, EvidenceScorer
from ecrd.logits_processor import ECRDLogitsProcessor
from transformers.generation import LogitsProcessorList

GLOBAL_DESCRIPTION_PROMPT = (
    "Provide a detailed and comprehensive description of this image. "
    "Do not answer the instruction: '{instruction}' yet. "
    "Focus only on describing the visible details, objects, layout, and visual features of the image."
)

@torch.inference_mode()
def generate_global_description(model, processor, image, question: str, min_pixels: int, max_pixels: int) -> str:
    prompt = GLOBAL_DESCRIPTION_PROMPT.format(instruction=question)
    messages = [{"role": "user", "content": [{"type": "image", "image": image, "min_pixels": min_pixels, "max_pixels": max_pixels}, {"type": "text", "text": prompt}]}]
    chat = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    img_inputs, vid_inputs = process_vision_info(messages)
    inputs = processor(text=[chat], images=img_inputs, videos=vid_inputs, padding=True, return_tensors="pt").to(model.device)
    out = model.generate(**inputs, do_sample=False, max_new_tokens=128, use_cache=True)
    return processor.batch_decode(out[:, inputs.input_ids.shape[1]:], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()

def build_messages(image, question: str, min_pixels: int, max_pixels: int):
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image, "min_pixels": min_pixels, "max_pixels": max_pixels},
                {"type": "text", "text": question + "\n\nThink step by step and put the final answer in <answer>...</answer>."},
            ],
        }
    ]

def get_correct_option(question: str, answer: str) -> str:
    # choices are formatted like (A) ... (B) ...
    matches = re.findall(r"\(([A-H])\)\s*(.*?)(?=\s*\([A-H]\)|$)", question, re.DOTALL)
    ans_clean = answer.replace(" ", "").lower()
    for letter, text in matches:
        if text.replace(" ", "").lower() == ans_clean:
            return letter.upper()
    
    # Fallback to direct letter check
    if len(answer.strip()) == 1 and answer.strip().upper() in "ABCDEFGH":
        return answer.strip().upper()
    return ""

def parse_multi_choice_prediction(text: str) -> str:
    ans_match = re.search(r"<answer>\s*([A-H])\b", text, flags=re.I)
    if ans_match:
        return ans_match.group(1).upper()
    
    ans_char = re.search(r"Answer:\s*([A-H])\b", text, flags=re.I)
    if ans_char:
        return ans_char.group(1).upper()
    
    ans_last = re.search(r"\b([A-H])\b", text[-20:], flags=re.I)
    if ans_last:
        return ans_last.group(1).upper()
    return ""

@torch.inference_mode()
def local_grade_perception(model, processor, question: str, model_answer: str, ground_truth: str) -> bool:
    if not model_answer.strip():
        return False
    
    prompt = f"""Compare the model's answer with the correct answer for the given question.
Question: {question}
Model's Answer: {model_answer}
Correct Answer: {ground_truth}

Is the model's answer correct and consistent with the correct answer? Respond with only "Correct" or "Incorrect"."""

    messages = [
        {
            "role": "user",
            "content": [{"type": "text", "text": prompt}]
        }
    ]
    chat = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[chat], padding=True, return_tensors="pt").to(model.device)
    
    out = model.generate(**inputs, do_sample=False, max_new_tokens=10, use_cache=True)
    output_text = processor.batch_decode(out[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0].strip()
    
    if "correct" in output_text.lower() and "incorrect" not in output_text.lower():
        return True
    return False

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--use-grit", action="store_true")
    ap.add_argument("--grit-model", default="yfan1997/GRIT-20-Qwen2.5-VL-3B")
    ap.add_argument("--delta", type=float, default=0.08)
    ap.add_argument("--load-in-4bit", action="store_true", help="Load base model in 4-bit")
    ap.add_argument("--grit-in-4bit", action="store_true", help="Load GRIT model in 4-bit")
    ap.add_argument("--grit-device", default="cpu", help="Device to run GRIT model on (e.g., cpu, 0)")
    ap.add_argument("--device", default=None, help="Device to run base model on (e.g., cpu, cuda:0)")
    ap.add_argument("--min-pixels", type=int, default=256*28*28)
    ap.add_argument("--max-pixels", type=int, default=512*28*28)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--limit", type=int, default=None, help="Limit evaluation to first N samples")
    ap.add_argument("--output-dir", default="results", help="Directory to save evaluation results JSON")
    args = ap.parse_args()

    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    os.environ["HF_HOME"] = os.path.join(project_root, ".cache")

    print("Loading LCZZZZ/RH-Bench dataset...")
    dataset = load_dataset("LCZZZZ/RH-Bench", split="train")

    print("Downloading metadata json files...")
    halu_path = hf_hub_download(repo_id="LCZZZZ/RH-Bench", filename="halu_data.json", repo_type="dataset")
    reason_path = hf_hub_download(repo_id="LCZZZZ/RH-Bench", filename="reason_data.json", repo_type="dataset")

    with open(halu_path, "r", encoding="utf-8") as f:
        halu_metadata = json.load(f)
    with open(reason_path, "r", encoding="utf-8") as f:
        reason_metadata = json.load(f)

    # Build lookup metadata dicts
    halu_dict = {item["image"]: item for item in halu_metadata}
    reason_dict = {item["image"]: item for item in reason_metadata}

    # Extract original paths from Arrow Table to align imagefolder splits
    print("Extracting Arrow table file paths for alignment...")
    all_paths = [x["path"] if x is not None else None for x in dataset.data["image"].to_pylist()]

    # Filter items that are aligned
    aligned_items = []
    for idx, path in enumerate(all_paths):
        if not path:
            continue
        parts = path.replace("\\", "/").split("/")
        rel_path = ""
        category = ""
        if "per_images" in parts:
            idx_part = parts.index("per_images")
            rel_path = "/".join(parts[idx_part:])
            category = "perception"
        elif "reason_images" in parts:
            idx_part = parts.index("reason_images")
            rel_path = "/".join(parts[idx_part:])
            category = "reasoning"
        
        if not rel_path:
            continue

        meta_item = None
        if category == "perception":
            meta_item = halu_dict.get(rel_path)
        elif category == "reasoning":
            meta_item = reason_dict.get(rel_path)
        
        if meta_item:
            aligned_items.append({
                "dataset_index": idx,
                "category": category,
                "rel_path": rel_path,
                "question": meta_item["question"],
                "answer": meta_item["answer"],
                "question_type": meta_item["question_type"],
                "id": meta_item["id"]
            })

    if args.limit:
        aligned_items = aligned_items[:args.limit]

    print(f"Total aligned samples to evaluate: {len(aligned_items)}")

    print(f"Initializing model: {args.model}")
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    
    device = args.device
    if not device:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    if device == "cpu":
        print("Loading base model on CPU (bfloat16)...")
        model = AutoModelForVision2Seq.from_pretrained(
            args.model,
            device_map="cpu",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )
    else:
        if args.load_in_4bit:
            print("Loading base model in 4-bit precision with device_map='auto'...")
            model = AutoModelForVision2Seq.from_pretrained(
                args.model,
                device_map="auto",
                load_in_4bit=True,
                trust_remote_code=True
            )
        else:
            print("Loading base model in bfloat16 with device_map='auto'...")
            model = AutoModelForVision2Seq.from_pretrained(
                args.model,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                trust_remote_code=True
            )

    grit_client = None
    if args.use_grit:
        print(f"Initializing GRIT Client with decider: {args.grit_model}")
        from ecrd.grit_client import GRITClient
        grit_device = args.grit_device
        if grit_device != "cpu" and grit_device.isdigit():
            grit_device = f"cuda:{grit_device}"
        
        grit_client = GRITClient(
            model_name=args.grit_model,
            device=grit_device,
            load_in_4bit=args.grit_in_4bit
        )

    details = []
    correct_reason = 0
    total_reason = 0
    correct_perc = 0
    total_perc = 0

    pbar = tqdm(aligned_items, desc="Evaluating RH-Bench")
    for item in pbar:
        # Load image from HF dataset row
        row = dataset[item["dataset_index"]]
        image = row["image"]
        
        category = item["category"]
        question = item["question"]
        ground_truth = item["answer"]
        question_type = item["question_type"]

        # Run model inference
        if args.use_grit:
            desc = generate_global_description(model, processor, image, question, args.min_pixels, args.max_pixels)
            scorer = EvidenceScorer(model=model, tokenizer=processor.tokenizer, max_prefix_len=128)
            scorer.add_evidence(Evidence(id="global-0", text=desc, source="global", time_step=0))
            
            proc = ECRDLogitsProcessor(scorer=scorer, tokenizer=processor.tokenizer, min_k=1, max_k=64)
            
            # Setup GRIT decider hook inside ECRD
            from ecrd.decider import build_grit_trigger
            trigger = build_grit_trigger(delta=args.delta)
            
            # Decider execution config copies the generation config
            import copy
            gen_config = copy.copy(model.generation_config)
            gen_config.do_sample = False
            gen_config.temperature = None
            gen_config.top_p = None
            gen_config.top_k = None
            
            def grit_hook_fn(*args, **kwargs):
                return grit_client.verify_token_candidates(*args, **kwargs)
            
            proc.set_grit_runtime(
                hook=grit_hook_fn,
                trigger=trigger,
                evidence_pool=scorer,
                question=question,
                image=image
            )
            
            messages = build_messages(image, question, args.min_pixels, args.max_pixels)
            chat = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            img_inputs, vid_inputs = process_vision_info(messages)
            inputs = processor(text=[chat], images=img_inputs, videos=vid_inputs, padding=True, return_tensors="pt").to(model.device)
            
            gen = model.generate(
                **inputs,
                generation_config=gen_config,
                max_new_tokens=args.max_new_tokens,
                logits_processor=LogitsProcessorList([proc]),
            )
        else:
            messages = build_messages(image, question, args.min_pixels, args.max_pixels)
            chat = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            img_inputs, vid_inputs = process_vision_info(messages)
            inputs = processor(text=[chat], images=img_inputs, videos=vid_inputs, padding=True, return_tensors="pt").to(model.device)
            
            gen = model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=args.max_new_tokens,
                use_cache=True,
            )

        pred_text = processor.batch_decode(gen[:, inputs.input_ids.shape[1]:], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()

        is_correct = False
        prediction = ""
        correct_answer_option = ""

        if category == "reasoning":
            correct_answer_option = get_correct_option(question, ground_truth)
            prediction = parse_multi_choice_prediction(pred_text)
            is_correct = (prediction == correct_answer_option) if correct_answer_option else False
            
            total_reason += 1
            if is_correct:
                correct_reason += 1
        else:  # perception
            # Local judge grading
            is_correct = local_grade_perception(model, processor, question, pred_text, ground_truth)
            prediction = pred_text
            
            total_perc += 1
            if is_correct:
                correct_perc += 1

        details.append({
            "id": item["id"],
            "category": category,
            "question": question,
            "prediction": prediction,
            "prediction_raw": pred_text,
            "ground_truth": ground_truth,
            "ground_truth_option": correct_answer_option if category == "reasoning" else "",
            "is_correct": is_correct
        })

        # Update progress bar description
        reason_acc = (correct_reason / total_reason * 100) if total_reason > 0 else 0
        perc_acc = (correct_perc / total_perc * 100) if total_perc > 0 else 0
        pbar.set_postfix({
            "reas": f"{reason_acc:.1f}%",
            "perc": f"{perc_acc:.1f}%"
        })

    # Calculations for metrics
    reas_accuracy = correct_reason / total_reason if total_reason > 0 else 0
    perc_accuracy = correct_perc / total_perc if total_perc > 0 else 0
    # RH-AUC approximation for single evaluation point: Reas * Perc
    rh_auc = reas_accuracy * perc_accuracy

    print("\n" + "=" * 54)
    print("RH-Bench Evaluation Results Summary")
    print("=" * 54)
    print(f"Reasoning (Reas)  :   {correct_reason}/{total_reason}   ({reas_accuracy * 100:.2f}%)")
    print(f"Perception (Perc) :   {correct_perc}/{total_perc}   ({perc_accuracy * 100:.2f}%)")
    print("-" * 54)
    print(f"RH-AUC (Rectangle):   {rh_auc:.4f}")
    print("=" * 54)

    # Save detailed JSON file
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    mode = "grit" if args.use_grit else "basic"
    model_name = os.path.basename(args.model).lower()
    output_file = os.path.join(args.output_dir, f"rhbench_{model_name}_{mode}.json")

    output_data = {
        "metadata": {
            "model": args.model,
            "use_grit": args.use_grit,
            "grit_model": args.grit_model if args.use_grit else None,
            "delta": args.delta,
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat() + "Z",
            "total_samples": len(aligned_items)
        },
        "summary": {
            "reasoning": {
                "correct": correct_reason,
                "total": total_reason,
                "accuracy": reas_accuracy
            },
            "perception": {
                "correct": correct_perc,
                "total": total_perc,
                "accuracy": perc_accuracy
            },
            "rh_auc": rh_auc
        },
        "details": details
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"Detailed results saved to: {output_file}")

if __name__ == "__main__":
    main()
