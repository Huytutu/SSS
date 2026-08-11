from __future__ import annotations

import types
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    apply_multimodal_rotary_pos_emb,
    repeat_kv,
)

__all__ = ["StepSignals", "AttentionProbe", "find_image_span"]

EPS = 1e-8


def find_image_span(input_ids: torch.Tensor, image_token_id: int) -> Tuple[int, int]:
    """Locate the contiguous run of image tokens in input_ids[0], as (start, end).

    Qwen2.5-VL puts one image's pad tokens in a single run between
    <|vision_start|> and <|vision_end|>, so a non-contiguous run means a
    multi-image prompt, which this probe does not handle.
    """
    matches = (input_ids[0] == image_token_id).nonzero(as_tuple=True)[0]
    if matches.numel() == 0:
        raise ValueError(f"No image_token_id={image_token_id} in input_ids.")
    start, end = int(matches[0]), int(matches[-1]) + 1
    if end - start != matches.numel():
        raise ValueError("Image tokens are not contiguous -- multi-image prompts aren't supported.")
    return start, end


@dataclass
class StepSignals:
    """Attention-derived signals for one decoding step."""

    pas_raw: float          # PAS eq.10: attention mass on previously generated text.
    pas_share: float        # pas_raw / (pas_raw + image mass) -- see PASConfig.gate_signal.
    peak_diff: float        # AFIP's peak textual minus peak visual attention.
    local_ratio: float      # share of prelim mass sitting on the immediately preceding token.
    img_row: Optional[torch.Tensor] = None   # [n_image_tokens], from localize_layer.

    def value(self, gate_signal: str) -> float:
        return {"pas_raw": self.pas_raw, "pas_share": self.pas_share, "peak_diff": self.peak_diff}[gate_signal]


class AttentionProbe:
    """Reads one attention row per decoding step, from two chosen decoder layers.

    Why not `model.generate(output_attentions=True)`: that keeps every layer's
    full [heads, q_len, kv_len] tensor alive for the whole generation. On an
    image-heavy Qwen2.5-VL prompt (~1500 tokens x 28 layers x hundreds of
    steps) that alone exhausts a 24GB GPU -- mllms_know hit the same wall and
    worked around it by loading the weights in 4-bit (their run.py:211).

    Instead this patches `forward` on just the two layers we need. Each patched
    forward runs the model's real attention kernel unchanged (so the model's
    output is bit-identical to an unpatched run), then does one extra tiny
    matmul for the last query row only -- [heads, kv_len], a few hundred KB --
    and keeps only scalars off it.

    The last row needs no causal mask: it is the newest position, so every
    cached key is already in its past. That is why this can skip mask handling
    entirely, and also why only the last row is ever read.

    Usage:
        probe = AttentionProbe(model, cfg)
        probe.set_context(prompt_len=..., img_span=...)
        try:
            ...model.generate(...)  # pop_step() once per decoding step
        finally:
            probe.restore()
    """

    def __init__(self, model, cfg):
        self.cfg = cfg
        self.prompt_len = 0
        self.img_span = (0, 0)
        self._rows = {}          # layer_idx -> [heads, kv_len] attention row for the current step
        self._originals = {}     # layer_idx -> unpatched bound forward
        self._modules = {}

        layers = model.model.language_model.layers
        for layer_idx in {cfg.pas_layer, cfg.localize_layer}:
            attn = layers[layer_idx].self_attn
            self._modules[layer_idx] = attn
            self._originals[layer_idx] = attn.forward
            attn.forward = types.MethodType(self._build_forward(layer_idx), attn)

    def set_context(self, prompt_len: int, img_span: Tuple[int, int]) -> None:
        """Tell the probe where the prompt ends and where the image tokens are.
        Call once per prompt -- both differ between the baseline pass and the
        real generation."""
        self.prompt_len = int(prompt_len)
        self.img_span = img_span
        self._rows.clear()

    def _build_forward(probe, layer_idx):
        def patched_forward(self, hidden_states, attention_mask=None, position_ids=None,
                            past_key_values=None, position_embeddings=None, **kwargs):
            # The model's own attention path, untouched.
            attn_output, attn_weights = probe._originals[layer_idx](
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                position_embeddings=position_embeddings,
                **kwargs,
            )

            # Recompute just the newest query row and score it against the keys
            # the call above already wrote into the cache. Passes made with
            # use_cache=False have no cache to read, and are never the passes
            # we want to measure anyway (encoding an evidence sentence, say),
            # so leave the buffer untouched -- pop_step() will complain if a
            # step that *should* have been measured wasn't.
            if past_key_values is None:
                return attn_output, attn_weights
            last_hidden = hidden_states[:, -1:, :]
            query = self.q_proj(last_hidden).view(1, 1, -1, self.head_dim).transpose(1, 2)
            cos, sin = position_embeddings
            query, _ = apply_multimodal_rotary_pos_emb(
                query, query, cos[..., -1:, :], sin[..., -1:, :],
                self.config.rope_parameters["mrope_section"],
            )
            keys = repeat_kv(past_key_values.layers[layer_idx].keys, self.num_key_value_groups)
            logits = torch.matmul(query, keys.transpose(2, 3)) * self.scaling
            probe._rows[layer_idx] = torch.softmax(logits.float(), dim=-1)[0, :, 0, :].detach()

            return attn_output, attn_weights

        return patched_forward

    def pop_step(self) -> StepSignals:
        """Signals for the step whose forward pass just finished. Clears the
        buffer, so call exactly once per step."""
        cfg = self.cfg
        if cfg.pas_layer not in self._rows:
            raise RuntimeError("AttentionProbe.pop_step() called without a forward pass since the last one.")
        row = self._rows[cfg.pas_layer].mean(dim=0)   # mean over heads, as PAS eq.10 does
        kv_len = row.shape[0]
        img_start, img_end = self.img_span

        # Prelim = previously generated text, excluding this position itself.
        prelim = row[self.prompt_len:kv_len - 1]
        image = row[img_start:img_end]

        pas_raw = float(prelim.sum()) if prelim.numel() else 0.0
        img_mass = float(image.sum())
        peak_text = float(prelim.max()) if prelim.numel() else 0.0

        local_ratio = 0.0
        if prelim.numel():
            local_ratio = float(prelim[-1]) / (pas_raw + EPS)

        img_row = None
        if cfg.localize_layer in self._rows:
            loc = self._rows[cfg.localize_layer].mean(dim=0)
            img_row = loc[img_start:img_end].clone()

        self._rows.clear()
        return StepSignals(
            pas_raw=pas_raw,
            pas_share=pas_raw / (pas_raw + img_mass + EPS),
            peak_diff=peak_text - float(image.max()),
            local_ratio=local_ratio,
            img_row=img_row,
        )

    def restore(self) -> None:
        """Undo the patch. Always call this, including on exceptions -- a left-over
        patch keeps paying the extra matmul and keeps this probe alive by closure."""
        for layer_idx, forward in self._originals.items():
            self._modules[layer_idx].forward = forward
        self._originals.clear()
        self._modules.clear()
