import os
import json
import re
import math
import argparse
import datetime
import numpy as np
from tqdm import tqdm
from PIL import Image
import torch
import torch.nn.functional as F
from datasets import load_dataset
from huggingface_hub import hf_hub_download

from transformers import AutoProcessor, AutoModelForImageTextToText
from qwen_vl_utils import process_vision_info

# ECRD imports
from MORAI.SSS.ecrd.scorer import Evidence, EvidenceScorer
from MORAI.SSS.ecrd.logits_processor import ECRDLogitsProcessor
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

def parse_free_response_prediction(text: str) -> str:
    # Reasoning questions without lettered choices (e.g. numeric answers) still use
    # the same "<answer>...</answer>" prompt format, just without the A-H restriction.
    ans_match = re.search(r"<answer>\s*(.*?)\s*</answer>", text, flags=re.I | re.S)
    if ans_match:
        return ans_match.group(1).strip()
    return text.strip()

@torch.inference_mode()
def local_grade_answer(model, processor, question: str, model_answer: str, ground_truth: str) -> bool:
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

# --- Budget-forcing (reasoning-length control), following the RH-Bench reference
# implementation: github.com/MLRM-Halu/MLRM-Halu/blob/main/length_control/budget_forcing.py

THINK_PROMPT_SUFFIX = (
    " You FIRST think about the reasoning process as an internal monologue and "
    "then provide the final answer. The reasoning process MUST BE enclosed "
    "within <think> </think> tags. The final answer MUST BE in <answer> </answer> tags."
)

def build_think_messages(image, question: str, min_pixels: int, max_pixels: int):
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image, "min_pixels": min_pixels, "max_pixels": max_pixels},
                {"type": "text", "text": question + THINK_PROMPT_SUFFIX},
            ],
        }
    ]

def extract_think(text: str) -> str:
    m = re.search(r"<think>(.*?)</think>", text, flags=re.S | re.I)
    return (m.group(1) if m else text).strip()

def build_forced_answer_messages(image, question: str, truncated_think: str, min_pixels: int, max_pixels: int):
    prompt = (
        f"{question}\n<think>\n{truncated_think}\n</think>\n\n"
        "So the final answer is (put it in <answer>...</answer>):"
    )
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image, "min_pixels": min_pixels, "max_pixels": max_pixels},
                {"type": "text", "text": prompt},
            ],
        }
    ]

def generate_once(model, processor, messages, max_new_tokens: int, proc=None, gen_config=None) -> str:
    chat = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    img_inputs, vid_inputs = process_vision_info(messages)
    inputs = processor(text=[chat], images=img_inputs, videos=vid_inputs, padding=True, return_tensors="pt").to(model.device)
    if proc is not None:
        gen = model.generate(
            **inputs,
            generation_config=gen_config,
            max_new_tokens=max_new_tokens,
            logits_processor=LogitsProcessorList([proc]),
        )
    else:
        gen = model.generate(**inputs, do_sample=False, max_new_tokens=max_new_tokens, use_cache=True)
    return processor.batch_decode(gen[:, inputs.input_ids.shape[1]:], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()

def generate_response(model, processor, image, question: str, args, proc=None, gen_config=None, think_char_budget=None) -> str:
    """Runs one full model response for a question.

    think_char_budget=None keeps the original single-shot behavior (one
    generation call with the answer embedded via <answer> tags).

    A numeric think_char_budget switches to the budget-forcing protocol from
    the RH-Bench reference implementation: generate a free chain-of-thought,
    truncate the <think> block to think_char_budget characters, then force a
    final answer conditioned on only that much visible reasoning. ECRD/GRIT
    supervision (proc/gen_config) applies to the thinking stage, mirroring how
    the single-shot mode supervises the whole chain-of-thought; the short
    forced final-answer stage is left unconstrained, matching the reference
    script.
    """
    if think_char_budget is None:
        messages = build_messages(image, question, args.min_pixels, args.max_pixels)
        return generate_once(model, processor, messages, args.max_new_tokens, proc, gen_config)

    think_messages = build_think_messages(image, question, args.min_pixels, args.max_pixels)
    think_text = generate_once(model, processor, think_messages, args.max_new_tokens, proc, gen_config)
    truncated_think = extract_think(think_text)[:think_char_budget]

    answer_messages = build_forced_answer_messages(image, question, truncated_think, args.min_pixels, args.max_pixels)
    return generate_once(model, processor, answer_messages, 128)

def build_ecrd_processor(model, processor, image, question: str, args, grit_client):
    desc = generate_global_description(model, processor, image, question, args.min_pixels, args.max_pixels)
    scorer = EvidenceScorer(model=model, tokenizer=processor.tokenizer, max_prefix_len=128)
    scorer.add_evidence(Evidence(id="global-0", text=desc, source="global", time_step=0))

    proc = ECRDLogitsProcessor(
        scorer=scorer, tokenizer=processor.tokenizer, min_k=1, max_k=64,
        collect_calibration_log=bool(args.collect_calib_log),
    )

    if grit_client is not None:
        from MORAI.SSS.ecrd.triggers import MixedGapTrigger
        trigger = MixedGapTrigger(gap_thresh=args.delta, min_k=2, cooldown=5)

        def grit_hook_fn(*a, **kw):
            return grit_client.decide_next_token(*a, **kw)

        proc.set_grit_runtime(hook=grit_hook_fn, trigger=trigger, evidence_pool=scorer, question=question, image=image)

    import copy
    gen_config = copy.copy(model.generation_config)
    gen_config.do_sample = False
    gen_config.temperature = None
    gen_config.top_p = None
    gen_config.top_k = None
    return proc, gen_config

def compute_rh_auc(points) -> float:
    """RH-AUC exactly as defined in the RH-Bench reference implementation
    (MLRM-Halu/RH-AUC.py, `auc_smooth`): min-max normalize both perception and
    reasoning accuracy across a reasoning-length sweep, then trapezoidally
    integrate perception (y) over reasoning (x). A single evaluation point has
    no length axis to integrate over, so this needs >=2 sweep points -- unlike
    the reas_acc * perc_acc rectangle used as a quick single-point stand-in.

    points: list of (perception_accuracy, reasoning_accuracy) tuples, one per
    reasoning-length budget in the sweep.
    """
    if len(points) < 2:
        raise ValueError("RH-AUC requires at least 2 length-sweep points")
    points_sorted = sorted(points, key=lambda p: p[1])
    y = [p[0] for p in points_sorted]
    x = [p[1] for p in points_sorted]
    x_min, x_max = min(x), max(x)
    y_min, y_max = min(y), max(y)
    x_norm = [(xi - x_min) / (x_max - x_min) for xi in x]
    y_norm = [(yi - y_min) / (y_max - y_min) for yi in y]
    return float(np.trapezoid(y_norm, x_norm))

def run_eval_pass(model, processor, dataset, aligned_items, grit_client, args, length_fraction=None):
    """Evaluates every item once and returns (correct_reason, total_reason,
    correct_perc, total_perc, details). length_fraction=None reproduces the
    original single-shot evaluation; a float in [0, 1] runs the budget-forcing
    protocol at length_fraction * category_max_chars (600 chars for reasoning
    items, 300 for perception items, per the RH-Bench project page's stated
    reasoning-length ranges).
    """
    details = []
    correct_reason = total_reason = correct_perc = total_perc = 0

    desc = "Evaluating RH-Bench" if length_fraction is None else f"Evaluating RH-Bench (length_fraction={length_fraction:.2f})"
    pbar = tqdm(aligned_items, desc=desc)
    for item in pbar:
        if item.get("local_img_path"):
            image = Image.open(item["local_img_path"]).convert("RGB")
        else:
            row = dataset[item["dataset_index"]]
            image = row["image"]

        category = item["category"]
        question = item["question"]
        ground_truth = item["answer"]

        proc, gen_config = (None, None)
        if args.use_supervisor:
            proc, gen_config = build_ecrd_processor(model, processor, image, question, args, grit_client if args.use_grit else None)

        think_char_budget = None
        if length_fraction is not None:
            category_max_chars = 600 if category == "reasoning" else 300
            think_char_budget = round(length_fraction * category_max_chars)

        pred_text = generate_response(model, processor, image, question, args, proc, gen_config, think_char_budget)

        is_correct = False
        prediction = ""
        correct_answer_option = ""

        if category == "reasoning":
            correct_answer_option = get_correct_option(question, ground_truth)
            if correct_answer_option:
                prediction = parse_multi_choice_prediction(pred_text)
                is_correct = (prediction == correct_answer_option)
            else:
                # No lettered choices in the question -> free-response (e.g. numeric)
                # answer, so grade it with the same local judge used for perception.
                prediction = parse_free_response_prediction(pred_text)
                is_correct = local_grade_answer(model, processor, question, prediction, ground_truth)

            total_reason += 1
            if is_correct:
                correct_reason += 1
        else:  # perception
            is_correct = local_grade_answer(model, processor, question, pred_text, ground_truth)
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
            "is_correct": is_correct,
            "grit_invocations": proc.grit_invocations if args.use_grit else None,
        })

        if args.collect_calib_log and args.use_grit:
            with open(args.collect_calib_log, "a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "question_id": item["id"],
                    "category": category,
                    "is_correct": is_correct,
                    "calib_log": proc.calib_log,
                }, ensure_ascii=False) + "\n")

        reason_acc = (correct_reason / total_reason * 100) if total_reason > 0 else 0
        perc_acc = (correct_perc / total_perc * 100) if total_perc > 0 else 0
        pbar.set_postfix({"reas": f"{reason_acc:.1f}%", "perc": f"{perc_acc:.1f}%"})

    return correct_reason, total_reason, correct_perc, total_perc, details

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
    mode_group = ap.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--base", action="store_true",
                             help="Raw decoding: no ECRD supervisor, no GRIT decider (paper's Base row).")
    mode_group.add_argument("--supervisor", action="store_true",
                             help="ECRD supervisor (negotiated reweighting) without the GRIT visual decider "
                                  "(paper's '+supervisor' row).")
    mode_group.add_argument("--ecrd", action="store_true",
                             help="Full ECRD: supervisor + GRIT visual decider (paper's '+ECRD' row).")
    ap.add_argument("--grit-model", default="weights/GRIT-20-Qwen2.5-VL-3B")
    ap.add_argument("--delta", type=float, default=0.08)
    ap.add_argument("--trigger", choices=["gap", "conformal"], default="gap",
                     help="gap = original MixedGapTrigger (uses --delta); conformal = ConformalTrigger (uses --q-hat)")
    ap.add_argument("--q-hat", type=float, default=None, help="Calibrated threshold for --trigger conformal")
    ap.add_argument("--collect-calib-log", default=None,
                     help="If set, append one JSONL record per question (question_id, is_correct, calib_log) to this path")
    ap.add_argument("--load-in-4bit", action="store_true", help="Load base model in 4-bit")
    ap.add_argument("--grit-in-4bit", action="store_true", help="Load GRIT model in 4-bit")
    ap.add_argument("--grit-device", default="0", help="Device to run GRIT model on (e.g., cpu, 0)")
    ap.add_argument("--device", default=None, help="Device to run base model on (e.g., cpu, cuda:0)")
    ap.add_argument("--min-pixels", type=int, default=256*28*28)
    ap.add_argument("--max-pixels", type=int, default=1280*28*28)
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--limit", type=int, default=None, help="Limit evaluation to N samples (stratified across categories)")
    ap.add_argument("--offset", type=int, default=0, help="Skip the first N stratified samples per category before applying --limit (for disjoint calib/test splits)")
    ap.add_argument("--data-dir", default=None, help="Directory for local dataset (default: data/RH-Bench)")
    ap.add_argument("--output-dir", default="results", help="Directory to save evaluation results JSON")
    ap.add_argument("--length-sweep", action="store_true",
                     help="Compute RH-AUC the paper-accurate way: sweep reasoning-length budgets via "
                          "budget forcing and trapezoidally integrate perception over reasoning, instead "
                          "of the single-point reas_acc*perc_acc approximation.")
    ap.add_argument("--length-fractions", type=float, nargs="+", default=[0.0, 0.25, 0.5, 0.75, 1.0],
                     help="Fractions of the category-specific max reasoning length (600 chars for "
                          "reasoning items, 300 for perception items) to sweep when --length-sweep is set.")
    args = ap.parse_args()
    # Derived flags used throughout: --ecrd implies the supervisor is on too.
    args.use_grit = args.ecrd
    args.use_supervisor = args.supervisor or args.ecrd

    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    os.environ["HF_HOME"] = os.path.join(project_root, ".cache")

    data_dir = args.data_dir or os.path.join(project_root, "data", "RH-Bench")
    local_halu = os.path.join(data_dir, "halu_data.json")
    local_reason = os.path.join(data_dir, "reason_data.json")

    aligned_items = []
    dataset = None

    if os.path.exists(local_halu) and os.path.exists(local_reason):
        print(f"Loading local RH-Bench dataset from: {data_dir}")
        with open(local_halu, "r", encoding="utf-8") as f:
            halu_metadata = json.load(f)
        with open(local_reason, "r", encoding="utf-8") as f:
            reason_metadata = json.load(f)

        for item in halu_metadata:
            aligned_items.append({
                "dataset_index": None,
                "category": "perception",
                "rel_path": item["image"],
                "local_img_path": os.path.join(data_dir, item["image"]),
                "question": item["question"],
                "answer": item["answer"],
                "question_type": item.get("question_type", ""),
                "id": item["id"]
            })
        for item in reason_metadata:
            aligned_items.append({
                "dataset_index": None,
                "category": "reasoning",
                "rel_path": item["image"],
                "local_img_path": os.path.join(data_dir, item["image"]),
                "question": item["question"],
                "answer": item["answer"],
                "question_type": item.get("question_type", ""),
                "id": item["id"]
            })
    else:
        print("Loading dataset from Hugging Face Hub: LCZZZZ/RH-Bench...")
        dataset = load_dataset("LCZZZZ/RH-Bench", split="train")

        print("Downloading metadata json files...")
        halu_path = hf_hub_download(repo_id="LCZZZZ/RH-Bench", filename="halu_data.json", repo_type="dataset")
        reason_path = hf_hub_download(repo_id="LCZZZZ/RH-Bench", filename="reason_data.json", repo_type="dataset")

        with open(halu_path, "r", encoding="utf-8") as f:
            halu_metadata = json.load(f)
        with open(reason_path, "r", encoding="utf-8") as f:
            reason_metadata = json.load(f)

        halu_dict = {item["image"]: item for item in halu_metadata}
        reason_dict = {item["image"]: item for item in reason_metadata}

        print("Extracting Arrow table file paths for alignment...")
        all_paths = [x["path"] if x is not None else None for x in dataset.data["image"].to_pylist()]

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
        # Dataset is stored as one contiguous block of "perception" items followed by
        # one contiguous block of "reasoning" items, so a plain prefix slice would only
        # ever sample one category. Split the limit evenly across both categories instead.
        # --offset skips the first N stratified items per category first, so a later run
        # can select a disjoint slice (e.g. calibration vs. held-out test) from the same
        # underlying dataset order.
        perc_items = [x for x in aligned_items if x["category"] == "perception"][args.offset:]
        reason_items = [x for x in aligned_items if x["category"] == "reasoning"][args.offset:]
        half = args.limit // 2
        aligned_items = perc_items[:half] + reason_items[:args.limit - half]

    print(f"Total aligned samples to evaluate: {len(aligned_items)}")

    model_path = resolve_local_model_path(args.model, project_root)
    is_local_model = os.path.isdir(model_path)

    if is_local_model:
        print(f"Loading local base model weights from: {model_path}")
    else:
        print(f"Initializing model from Hub: {model_path}")

    processor = AutoProcessor.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=is_local_model
    )
    
    device = args.device
    if not device:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    model_kwargs = {"trust_remote_code": True}
    if is_local_model:
        model_kwargs["local_files_only"] = True

    if device == "cpu":
        print("Loading base model on CPU (bfloat16)...")
        model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            device_map="cpu",
            torch_dtype=torch.bfloat16,
            **model_kwargs
        )
    else:
        if args.load_in_4bit:
            print("Loading base model in 4-bit precision with device_map='auto'...")
            model = AutoModelForImageTextToText.from_pretrained(
                model_path,
                device_map="auto",
                load_in_4bit=True,
                **model_kwargs
            )
        else:
            print("Loading base model in bfloat16 with device_map='auto'...")
            model = AutoModelForImageTextToText.from_pretrained(
                model_path,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                **model_kwargs
            )

    grit_client = None
    if args.use_grit:
        grit_path = resolve_local_model_path(args.grit_model, project_root)
        print(f"Initializing GRIT Client with decider: {grit_path}")
        from MORAI.SSS.ecrd.grit_client import GRITClient
        grit_device = args.grit_device
        if grit_device != "cpu" and grit_device.isdigit():
            grit_device = f"cuda:{grit_device}"
        
        grit_client = GRITClient(
            model_id=grit_path,
            device=grit_device,
            load_in_4bit=args.grit_in_4bit
        )

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    mode = "ecrd" if args.ecrd else ("supervisor" if args.supervisor else "base")
    model_name = os.path.basename(args.model).lower()

    if args.length_sweep:
        length_points = []
        for frac in args.length_fractions:
            correct_reason, total_reason, correct_perc, total_perc, details = run_eval_pass(
                model, processor, dataset, aligned_items, grit_client, args, length_fraction=frac
            )
            reas_accuracy = correct_reason / total_reason if total_reason > 0 else 0
            perc_accuracy = correct_perc / total_perc if total_perc > 0 else 0
            print(f"[length_fraction={frac:.2f}] Reas={reas_accuracy * 100:.2f}%  Perc={perc_accuracy * 100:.2f}%")
            length_points.append({
                "length_fraction": frac,
                "reasoning": {"correct": correct_reason, "total": total_reason, "accuracy": reas_accuracy},
                "perception": {"correct": correct_perc, "total": total_perc, "accuracy": perc_accuracy},
                "details": details,
            })

        rh_auc = compute_rh_auc([(p["perception"]["accuracy"], p["reasoning"]["accuracy"]) for p in length_points])
        # Headline Reas/Perc numbers are the full-budget (largest fraction) point, matching
        # how the ECRD paper's Table 4 reports a single Reas/Perc pair alongside RH-AUC.
        headline = length_points[-1]

        print("\n" + "=" * 54)
        print("RH-Bench Evaluation Results Summary (length sweep)")
        print("=" * 54)
        print(f"Reasoning (Reas)  :   {headline['reasoning']['accuracy'] * 100:.2f}%  (full-budget point)")
        print(f"Perception (Perc) :   {headline['perception']['accuracy'] * 100:.2f}%  (full-budget point)")
        print("-" * 54)
        print(f"RH-AUC ({len(length_points)}-point trapezoid, paper-accurate):   {rh_auc:.4f}")
        print("=" * 54)

        output_data = {
            "metadata": {
                "model": args.model,
                "mode": mode,
                "use_supervisor": args.use_supervisor,
                "use_grit": args.use_grit,
                "grit_model": args.grit_model if args.use_grit else None,
                "delta": args.delta,
                "length_fractions": args.length_fractions,
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat() + "Z",
                "total_samples": len(aligned_items)
            },
            "summary": {
                "reasoning": headline["reasoning"],
                "perception": headline["perception"],
                "rh_auc": rh_auc,
                "length_points": length_points,
            },
        }
    else:
        correct_reason, total_reason, correct_perc, total_perc, details = run_eval_pass(
            model, processor, dataset, aligned_items, grit_client, args, length_fraction=None
        )
        reas_accuracy = correct_reason / total_reason if total_reason > 0 else 0
        perc_accuracy = correct_perc / total_perc if total_perc > 0 else 0
        # Single-point stand-in, not the paper's real RH-AUC -- pass --length-sweep for that.
        rh_auc = reas_accuracy * perc_accuracy

        print("\n" + "=" * 54)
        print("RH-Bench Evaluation Results Summary")
        print("=" * 54)
        print(f"Reasoning (Reas)  :   {correct_reason}/{total_reason}   ({reas_accuracy * 100:.2f}%)")
        print(f"Perception (Perc) :   {correct_perc}/{total_perc}   ({perc_accuracy * 100:.2f}%)")
        print("-" * 54)
        print(f"RH-AUC (single-point rectangle, use --length-sweep for the paper-accurate metric):   {rh_auc:.4f}")
        print("=" * 54)

        output_data = {
            "metadata": {
                "model": args.model,
                "mode": mode,
                "use_supervisor": args.use_supervisor,
                "use_grit": args.use_grit,
                "grit_model": args.grit_model if args.use_grit else None,
                "delta": args.delta,
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat() + "Z",
                "total_samples": len(aligned_items)
            },
            "summary": {
                "reasoning": {"correct": correct_reason, "total": total_reason, "accuracy": reas_accuracy},
                "perception": {"correct": correct_perc, "total": total_perc, "accuracy": perc_accuracy},
                "rh_auc": rh_auc
            },
            "details": details
        }

    output_file = os.path.join(args.output_dir, f"rhbench_{model_name}_{mode}.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"Detailed results saved to: {output_file}")

if __name__ == "__main__":
    main()
