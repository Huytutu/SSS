from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch

DEFAULT_MODEL_PATH = "weights/Qwen2.5-VL-7B-Instruct"


def build_messages(image, question, min_pixels=None, max_pixels=None):
    image_content = {"type": "image", "image": image}
    if min_pixels is not None:
        image_content["min_pixels"] = min_pixels
    if max_pixels is not None:
        image_content["max_pixels"] = max_pixels
    messages = [
        {
            "role": "user",
            "content": [
                image_content,
                {"type": "text", "text": question + "\n\nThink step by step and put the final answer in <answer>...</answer>."},
            ],
        }
    ]
    return messages


def load_qwen(model_path: str = DEFAULT_MODEL_PATH):
    # We recommend enabling flash_attention_2 for better acceleration and memory saving
    processor = AutoProcessor.from_pretrained(model_path)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        # attn_implementation="flash_attention_2",
        device_map="cuda:0",
    )
    return model, processor


def qwen_base(
    image,
    question,
    model=None,
    processor=None,
    model_path: str = DEFAULT_MODEL_PATH,
    min_pixels=None,
    max_pixels=None,
    max_new_tokens: int = 512,
):
    """Greedy decoding, no logits processor"""
    if model is None or processor is None:
        model, processor = load_qwen(model_path)

    messages = build_messages(image, question, min_pixels, max_pixels)
    chat = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    img_inputs, vid_inputs = process_vision_info(messages)
    inputs = processor(text=[chat], images=img_inputs, videos=vid_inputs, padding=True, return_tensors="pt").to(model.device)
    gen = model.generate(
        **inputs,
        do_sample=False,
        max_new_tokens=max_new_tokens,
        use_cache=True,
    )
    text = processor.batch_decode(gen[:, inputs.input_ids.shape[1]:], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    return text
