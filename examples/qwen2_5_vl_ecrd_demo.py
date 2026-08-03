#!/usr/bin/env python
"""Minimal Qwen2.5-VL + ECRD demo.

This script is intentionally small: it shows how to attach ECRD as a HuggingFace
LogitsProcessor. For LLaVA/InternVL, keep the ECRD objects the same and replace
only the model-specific image/chat preprocessing.

python3 qwen2_5_vl_ecrd_demo.py --image --question
"""
from __future__ import annotations
import argparse
import os
import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, LogitsProcessorList
from qwen_vl_utils import process_vision_info

from SSS.ecrd import Evidence, EvidenceScorer, ECRDLogitsProcessor, MixedGapTrigger, GRITClient
from SSS.ecrd.prompts import GLOBAL_DESCRIPTION_PROMPT


def build_messages(image: str, question: str):
    return [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": question + "\n\nThink step by step and put the final answer in <answer>...</answer>."},
        ],
    }]


@torch.inference_mode()
def generate_global_description(model, processor, image: str, question: str, max_new_tokens: int = 256) -> str:
    prompt = GLOBAL_DESCRIPTION_PROMPT.format(instruction=question)
    messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}]
    chat = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    img_inputs, vid_inputs = process_vision_info(messages)
    inputs = processor(text=[chat], images=img_inputs, videos=vid_inputs, padding=True, return_tensors="pt").to(model.device)
    out = model.generate(**inputs, do_sample=False, max_new_tokens=max_new_tokens, use_cache=True)
    return processor.batch_decode(out[:, inputs.input_ids.shape[1]:], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--image", required=True, help="local path or file:// URI")
    ap.add_argument("--question", required=True)
    ap.add_argument("--load-in-4bit", action="store_true", help="Load base model in 4-bit")
    ap.add_argument("--use-grit", action="store_true")
    ap.add_argument("--grit-model", default="yfan1997/GRIT-20-Qwen2.5-VL-3B")
    ap.add_argument("--grit-in-4bit", action="store_true", help="Load GRIT model in 4-bit")
    ap.add_argument("--grit-device", default="0", help="Device to run GRIT model on (e.g., cpu, 0)")
    ap.add_argument("--delta", type=float, default=0.08)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    args = ap.parse_args()

    image_uri = args.image if args.image.startswith("file://") else "file://" + os.path.abspath(args.image)
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    
    model_kwargs = {
        "device_map": "cuda:0" if torch.cuda.is_available() else "auto",
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

    desc = generate_global_description(model, processor, image_uri, args.question)
    scorer = EvidenceScorer(model=model, tokenizer=processor.tokenizer, max_prefix_len=128)
    scorer.add_evidence(Evidence(id="global-0", text=desc, source="global", time_step=0))

    proc = ECRDLogitsProcessor(scorer=scorer, tokenizer=processor.tokenizer, min_k=1, max_k=64)

    if args.use_grit:
        grit = GRITClient(
            model_id=args.grit_model,
            device=args.grit_device,
            torch_dtype=torch.bfloat16,
            load_in_4bit=args.grit_in_4bit
        )
        def hook(image, question, prefix_text, candidates):
            return grit.decide_next_token(
                image=image,
                question=question,
                prefix_text=prefix_text,
                candidates=candidates,
                max_new_tokens=64,
            )
        proc.set_grit_runtime(
            hook=hook,
            trigger=MixedGapTrigger(gap_thresh=args.delta, min_k=2, cooldown=5),
            evidence_pool=scorer,
            question=args.question,
            image=image_uri,
        )

    messages = build_messages(image_uri, args.question)
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
    text = processor.batch_decode(gen[:, inputs.input_ids.shape[1]:], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    print(text)


if __name__ == "__main__":
    main()
