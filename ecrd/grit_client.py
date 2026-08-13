from __future__ import annotations
import json
import os
import re
import traceback
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, StoppingCriteria, StoppingCriteriaList

try:
    from qwen_vl_utils import process_vision_info
except Exception:  # pragma: no cover
    process_vision_info = None

PROMPT_TEMPLATE = (
    "You are given an image and the tail of a partially generated answer (the text "
    "immediately before the next token). Your job is to pick exactly ONE next token "
    "from the candidate set C.\n\n"
    "Inputs:\n"
    "- Prefix tail: \"{prefix}\"\n"
    "- Candidates C (index -> token):\n"
    "{candidates_lines}\n\n"
    "Return EXACTLY two XML blocks and NOTHING ELSE; however, if needed, you may first "
    "output a single JSON object with key 'bbox_2d' containing any coordinates you used. "
    "If present, this JSON must appear BEFORE the XML blocks and contain only that key.\n\n"
    "<evidence>\n"
    "One short sentence (<=60 words) stating a VISUAL fact that makes your chosen token "
    "more plausible than at least one alternative given the prefix tail. "
    "Do NOT copy or paraphrase any span (>=8 consecutive words) from the prefix tail. "
    "Do NOT mention candidate strings or any index.\n"
    "If no visual grounding is needed (e.g., all candidates are function words like 'Ġof', "
    "'Ġand', 'Ġis', punctuation, etc.), write: None\n"
    "</evidence>\n\n"
    "<answer>\n"
    "<INDEX of your chosen candidate in C, one integer only>\n"
    "</answer>\n\n"
    "Hard rules:\n"
    "- Consider ONLY the candidates in C.\n"
    "- The <answer> must contain a single integer 0..|C|-1 on its own line.\n"
    "- The <evidence> must be about the image; if you cannot ground it visually, write 'None'.\n"
    "- Never include candidate text, indexes, or any part of the prefix in <evidence>.\n"
)

_ANS_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", flags=re.S | re.I)
_EVID_RE = re.compile(r"<evidence>\s*(.*?)\s*</evidence>", flags=re.S | re.I)
_THINK_LEAD = re.compile(r"^\s*<think>[\s\S]*?</think>\s*", flags=re.I)
_ASSISTANT_MARKERS = ["\nassistant\n", "\nAssistant:\n", "ASSISTANT:\n", "<|im_start|>assistant", "<|im_start|> assistant", "\n### Assistant:\n"]
_JSON_LEAD_RE = re.compile(r'^\s*\{\s*"bbox_2d"\s*:\s*\[.*?\]\s*\}\s*', flags=re.S)


def _tail_after_last_assistant(prefix_text: str) -> str:
    if not prefix_text:
        return ""
    last_pos, last_len = -1, 0
    for marker in _ASSISTANT_MARKERS:
        pos = prefix_text.rfind(marker)
        if pos > last_pos:
            last_pos, last_len = pos, len(marker)
    return prefix_text[last_pos + last_len:] if last_pos >= 0 else prefix_text


def _strip_leading_think(s: str) -> str:
    return _THINK_LEAD.sub("", s or "")


def _extract_between(text: str, cre: re.Pattern) -> Optional[str]:
    m = cre.search(text or "")
    return m.group(1).strip() if m else None


def _parse_answer_index(out: str) -> Optional[int]:
    raw = _extract_between(out, _ANS_RE)
    if raw is None:
        return None
    m = re.search(r"([0-9]+)", raw)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def _parse_evidence_text(out: str) -> Optional[str]:
    ev = _extract_between(out, _EVID_RE)
    return ev.strip() if ev is not None else None


def _strip_leading_bbox_json(s: str) -> str:
    return _JSON_LEAD_RE.sub("", s or "")


def _tokenize_words(s: str) -> List[str]:
    return re.findall(r"[A-Za-z0-9]+", (s or "").lower())


def _remove_copied_spans(prefix_tail: str, evidence: str, n: int = 8, max_patterns: int = 2000) -> str:
    if not evidence:
        return ""
    words = _tokenize_words(prefix_tail)
    if len(words) < n:
        return evidence
    cleaned = evidence
    for i in range(min(len(words) - n + 1, max_patterns)):
        pattern = re.compile(r"\b" + r"\s+".join(map(re.escape, words[i:i+n])) + r"\b", flags=re.I)
        cleaned = pattern.sub(" ", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()


def _limit_words(s: str, max_words: int = 128) -> str:
    words = s.strip().split()
    return " ".join(words[:max_words]) if len(words) > max_words else " ".join(words)


def _clean_evidence(prefix_tail: str, evidence: Optional[str], max_words: int = 60) -> Optional[str]:
    if not evidence:
        return None
    ev = _remove_copied_spans(prefix_tail, evidence.strip(), n=8)
    ev = _limit_words(ev, max_words=max_words)
    return "None" if (not ev or ev.lower() == "none") else ev


class _EndsWithAny(StoppingCriteria):
    def __init__(self, tokenizer, patterns: List[str], start_len: int = 0):
        super().__init__()
        self.pats = [tokenizer.encode(p, add_special_tokens=False) for p in patterns]
        self.start_len = int(start_len)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        tail = input_ids[0, self.start_len:].tolist()
        for pat in self.pats:
            L = len(pat)
            if L > 0 and len(tail) >= L and tail[-L:] == pat:
                return True
        return False


class GRITClient:
    """Minimal GRIT-based visual decider for ECRD.

    It returns a candidate token id and a short textual evidence sentence.
    This class is Qwen2.5-VL/GRIT specific; the ECRD supervisor itself is model-agnostic.
    """

    def __init__(
        self,
        model_id: str = "yfan1997/GRIT-20-Qwen2.5-VL-3B",
        device: Union[str, int] = 0,
        torch_dtype: Union[str, torch.dtype] = torch.bfloat16,
        max_prefix_chars: int = 512,
        trace_dir: Optional[str] = None,
        trace_file: str = "grit_trace.jsonl",
    ):
        # This checkpoint's config.json has use_cache=null, which newer huggingface_hub
        # strict validation rejects. Both the model and processor load re-read this file,
        # so patch it on disk once rather than in memory.
        config_path = os.path.join(model_id, "config.json")
        if os.path.isfile(config_path):
            with open(config_path) as f:
                config_dict = json.load(f)
            if config_dict.get("use_cache") is None:
                config_dict["use_cache"] = True
                with open(config_path, "w") as f:
                    json.dump(config_dict, f, indent=2)

        # device_map="auto" lets accelerate shard across every visible GPU, which can land
        # on GPUs other jobs are using. Pin to the requested device instead.
        device_map = f"cuda:{device}" if str(device).isdigit() else device

        try:
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_id,
                torch_dtype=torch_dtype,
                device_map=device_map,
                attn_implementation="flash_attention_2",
            ).eval()
        except ImportError:
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_id,
                torch_dtype=torch_dtype,
                device_map=device_map,
                attn_implementation="sdpa",
            ).eval()
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.max_prefix_chars = int(max_prefix_chars)
        self._trace_fp = None
        if trace_dir:
            os.makedirs(trace_dir, exist_ok=True)
            self._trace_fp = os.path.join(trace_dir, trace_file)

    def _write_trace(self, rec: Dict[str, Any]) -> None:
        if not self._trace_fp:
            return
        try:
            rec = dict(**rec)
            rec.setdefault("ts", datetime.utcnow().isoformat() + "Z")
            with open(self._trace_fp, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except Exception:
            pass

    @staticmethod
    def _format_candidates_lines(candidates: List[Dict[str, Any]]) -> str:
        lines = []
        for i, cand in enumerate(candidates):
            tok = str(cand.get("text")).replace("\n", "\\n")
            lines.append(f'{i}: "{tok}"')
        return "\n".join(lines)

    def _prepare_prefix(self, prefix_text: str) -> str:
        prefix = _tail_after_last_assistant(prefix_text or "")
        prefix = _strip_leading_think(prefix)
        if self.max_prefix_chars > 0 and len(prefix) > self.max_prefix_chars:
            prefix = prefix[-self.max_prefix_chars:]
        return prefix.strip()

    def build_messages(self, image: Union[str, Any], question: str, prefix_text: str, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        prompt = PROMPT_TEMPLATE.format(
            prefix=self._prepare_prefix(prefix_text or ""),
            candidates_lines=self._format_candidates_lines(candidates),
        )
        return [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}]

    @torch.inference_mode()
    def decide_next_token(
        self,
        image: Union[str, Any],
        question: str,
        prefix_text: str,
        candidates: List[Dict[str, Any]],
        qid: str = "",
        max_new_tokens: int = 64,
        debug: bool = False,
        return_prompt: bool = False,
    ) -> Dict[str, Any]:
        if process_vision_info is None:
            raise RuntimeError("qwen_vl_utils is required for GRITClient. Install with `pip install qwen-vl-utils`.")

        messages = self.build_messages(image, question, prefix_text, candidates)
        prompt_text = messages[0]["content"][1]["text"]
        if debug:
            print("\n[GRIT] --- Prompt (text) ---")
            print(prompt_text)
            print("[GRIT] --- /Prompt ---\n")

        chat_text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        img_inputs, vid_inputs = process_vision_info(messages)
        inputs = self.processor(text=[chat_text], images=img_inputs, videos=vid_inputs, padding=True, return_tensors="pt").to(self.model.device)

        gen_cfg = self.model.generation_config
        gen_cfg.max_new_tokens = max_new_tokens
        gen_cfg.do_sample = False
        gen_cfg.temperature = 0.0
        gen_cfg.top_k = 1
        gen_cfg.top_p = 0.0
        stopper = _EndsWithAny(self.processor.tokenizer, ["</answer>"], start_len=int(inputs.input_ids.shape[1]))
        gen_ids = self.model.generate(**inputs, generation_config=gen_cfg, stopping_criteria=StoppingCriteriaList([stopper]), use_cache=True)
        out = self.processor.batch_decode(gen_ids[:, inputs.input_ids.shape[1]:], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        out = _strip_leading_bbox_json(out)
        if debug:
            print("[GRIT] raw output:\n", out)

        idx = _parse_answer_index(out)
        prefix_tail = self._prepare_prefix(prefix_text or "")
        evidence = _clean_evidence(prefix_tail, _parse_evidence_text(out), max_words=60)
        choice_id = None
        choice_text = None
        if idx is not None and 0 <= idx < len(candidates):
            choice_id = int(candidates[idx]["id"])
            choice_text = str(candidates[idx]["text"])

        self._write_trace({
            "qid": qid,
            "image": image if isinstance(image, str) else "<obj>",
            "prompt_text": prompt_text,
            "raw_output": out,
            "prefix_tail": prefix_tail,
            "candidates": candidates,
            "parsed": {"idx": idx, "choice_id": choice_id, "choice_text": choice_text, "evidence": evidence},
        })
        ret = {"choice_id": choice_id, "choice_text": choice_text, "unified_evidence": evidence, "bboxes": [], "raw_text": out}
        if return_prompt:
            ret["prompt_text"] = prompt_text
        return ret