"""TreeBench evaluation for ECRD, structured as a minimal diff on top of
eval/inference_treebench.py (the reference-faithful "qwen goc" harness).

Everything that isn't ECRD-specific -- system prompt, options formatting,
generation kwargs, box IoU, answer extraction, per-category tallying -- is
copied verbatim from inference_treebench.py so the two scripts differ only
in the ECRD supervisor/GRIT machinery, keeping the comparison controlled.

Usage:
  python eval/inference_treebench_ecrd.py --device 0 --grit-device 1
"""
import argparse
import ast
import os
import re

from tqdm import tqdm
import torch
import numpy as np
from datasets import load_dataset
from qwen_vl_utils import process_vision_info
from transformers import LogitsProcessorList

# .../SSS -> .../MORAI (project root, holding weights/ and data/)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(PROJECT_ROOT, "weights", "Qwen2.5-VL-7B-Instruct")
GRIT_MODEL_PATH = os.path.join(PROJECT_ROOT, "weights", "GRIT-20-Qwen2.5-VL-3B")
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "TreeBench", "TreeBench.tsv")

from SSS.ecrd import Evidence, EvidenceScorer, ECRDLogitsProcessor, MixedGapTrigger, GRITClient
from SSS.ecrd.prompts import GLOBAL_DESCRIPTION_PROMPT

SYSTEM_PROMPT = """A conversation between user and assistant. The user asks a question, and the Assistant solves it. The assistant MUST first think about the reasoning process in the mind and then provide the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively. When referring to particular objects in the reasoning process, the assistant MUST localize the object with bounding box coordinates between <box> and </box>. You MUST strictly follow the format."""
ANSWER_INSTRUCTION = "\nSelect the best answer to the above multiple-choice question based on the image. After the reasoning process, respond with only the letter of the correct option between <answer> and </answer>."


def compute_box_iou(predict_str: str, target_boxes: list) -> float:
    pattern = r"<box>(.*?)</box>"
    matches = re.findall(pattern, predict_str, re.DOTALL)

    all_boxes = []

    for match in matches:
        box = match.strip()

        coord_pattern = r'\[(\d+),(\d+),(\d+),(\d+)\]'
        coord_match = re.match(coord_pattern, box)

        if coord_match:
            x1, y1, x2, y2 = map(int, coord_match.groups())

            if x1 < x2 and y1 < y2:
                all_boxes.append([x1, y1, x2, y2])

    def calculate_average_iou(pred_boxes, target_boxes):
        def compute_iou(box1, box2):
            x1_min, y1_min, x1_max, y1_max = box1
            x2_min, y2_min, x2_max, y2_max = box2

            inter_x_min = max(x1_min, x2_min)
            inter_y_min = max(y1_min, y2_min)
            inter_x_max = min(x1_max, x2_max)
            inter_y_max = min(y1_max, y2_max)

            inter_width = max(0, inter_x_max - inter_x_min)
            inter_height = max(0, inter_y_max - inter_y_min)
            inter_area = inter_width * inter_height

            area1 = (x1_max - x1_min) * (y1_max - y1_min)
            area2 = (x2_max - x2_min) * (y2_max - y2_min)

            union_area = area1 + area2 - inter_area

            return inter_area / union_area if union_area > 0 else 0.0

        pred_coords = pred_boxes
        target_coords = target_boxes  # x1,y1,x2,y2

        total_iou = 0.0
        num_targets = len(target_boxes)

        if num_targets == 0:
            return 0.0

        for t_coord in target_coords:
            best_iou = 0.0
            for p_coord in pred_coords:
                iou = compute_iou(t_coord, p_coord)
                if iou > best_iou:
                    best_iou = iou
            total_iou += best_iou

        return total_iou / num_targets

    return calculate_average_iou(all_boxes, target_boxes)


@torch.inference_mode()
def generate_global_description(image_content: str, question: str) -> str:
    prompt = GLOBAL_DESCRIPTION_PROMPT.format(instruction=question)
    messages = [{"role": "user", "content": [
        {"type": "image", "image": f"data:image/jpeg;base64,{image_content}"},
        {"type": "text", "text": prompt},
    ]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to(model.device)
    out = model.generate(**inputs, do_sample=False, use_cache=True, max_new_tokens=256)
    return processor.batch_decode(out[:, inputs.input_ids.shape[1]:], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()


def eval_model_row(item):
    if item["category"] == "OCR":
        qs = item["question"]
    else:
        qs = item["question"] + " Options:\n" + item["multi-choice options"]

    content = [
        {
            "type": "image",
            "image": f"data:image/jpeg;base64,{item['image']}",
        },
        {
            "type": "text",
            "text": qs + ANSWER_INSTRUCTION,
        },
    ]

    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": SYSTEM_PROMPT}],
        },
        {
            "role": "user",
            "content": content,
        },
    ]

    # --- ECRD: build the supervisor for this question ---
    desc = generate_global_description(item["image"], qs)
    scorer = EvidenceScorer(model=model, tokenizer=processor.tokenizer, max_prefix_len=128)
    scorer.add_evidence(Evidence(id="global-0", text=desc, source="global", time_step=0))

    proc = ECRDLogitsProcessor(scorer=scorer, tokenizer=processor.tokenizer, min_k=1, max_k=64)
    if grit is not None:
        def grit_hook(*args, **kwargs):
            return grit.decide_next_token(*args, **kwargs, max_new_tokens=64)
        proc.set_grit_runtime(
            hook=grit_hook,
            trigger=MixedGapTrigger(gap_thresh=DELTA, min_k=2, cooldown=5),
            evidence_pool=scorer,
            question=qs,
            image=f"data:image/jpeg;base64,{item['image']}",
        )
    # --- /ECRD ---

    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    text += "<think>"

    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)

    # Inference: Generation of the output
    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            top_p=0.001,
            top_k=1,
            temperature=0.01,
            repetition_penalty=1.0,
            max_new_tokens=1024,
            use_cache=True,
            do_sample=True,
            logits_processor=LogitsProcessorList([proc]),
        )

    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=False, clean_up_tokenization_spaces=False
    )

    box_iou = compute_box_iou(output_text[0], ast.literal_eval(item["target_instances"]))

    pattern = r"<answer>(.*?)</answer>"
    match = re.search(pattern, output_text[0], re.DOTALL)
    ans = match.group(1).strip().upper() if match else output_text[0]

    item["prediction"] = ans
    item["iou"] = box_iou
    item["grit_invocations"] = proc.grit_invocations

    return item


model = None
processor = None
grit = None
DELTA = 0.08


def load_model(base_gpu, grit_gpu, delta):
    global model, processor, grit, DELTA
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

    DELTA = delta

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map=f"cuda:{base_gpu}",
        low_cpu_mem_usage=True,
    )
    processor = AutoProcessor.from_pretrained(
        MODEL_PATH,
        min_pixels=1280 * 28 * 28, max_pixels=16384 * 28 * 28,
    )

    if grit_gpu is not None:
        grit = GRITClient(model_id=GRIT_MODEL_PATH, device=grit_gpu, torch_dtype=torch.bfloat16)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="0", help="GPU id for the base Qwen2.5-VL-7B model")
    ap.add_argument("--grit-device", default=None, help="GPU id for the GRIT-3B visual decider (omit to disable GRIT, supervisor-only)")
    ap.add_argument("--delta", type=float, default=0.08, help="Uncertainty threshold that triggers the GRIT visual decider")
    ap.add_argument("--limit", type=int, default=None, help="Evaluate only the first N rows (dev convenience, no effect on eval logic)")
    args = ap.parse_args()

    load_model(args.device, args.grit_device, args.delta)

    # load data
    df = load_dataset("csv", data_files=DATA_PATH, delimiter="\t")["train"]
    if args.limit:
        df = df.select(range(min(args.limit, len(df))))

    # obtain results (sequential: base + GRIT are co-resident per question)

    data = []
    for item in tqdm(df, desc="Processing"):
        result = eval_model_row(item)
        if result is not None:
            data.append(result)

    results = {}
    tags = ["Perception/Attributes", "Perception/Material", "Perception/Physical State",
            "Perception/Object Retrieval", "Perception/OCR",
            "Reasoning/Perspective Transform", "Reasoning/Ordering", "Reasoning/Contact and Occlusion",
            "Reasoning/Spatial Containment", "Reasoning/Comparison"]
    total = 0
    correct = 0

    for tag in tags:
        results[tag] = {"correct": 0, "total": 0}
        for item in data:
            if tag == item["category"]:
                total += 1
                results[tag]["total"] += 1
                # exact matching
                if item["prediction"].upper() == item["answer"].upper():
                    results[tag]["correct"] += 1
                    correct += 1

        acc = results[tag]["correct"] / results[tag]["total"] if results[tag]["total"] else 0.0
        print(tag, f"{results[tag]['correct']}/{results[tag]['total']}={round(acc * 100, 2)}")
    print("==> Overall", f"{correct}/{total}={round(correct / total * 100, 2)}")

    iou = np.array([x["iou"] for x in data])
    print("==> Mean IoU:", round(np.mean(iou) * 100, 2))

    grit_calls = np.array([x["grit_invocations"] for x in data])
    print("==> Total GRIT invocations:", int(grit_calls.sum()), f"({grit_calls.mean():.2f}/question)")
