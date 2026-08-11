from __future__ import annotations
from typing import Dict, List, Optional, Tuple

import types

import torch
from transformers import LogitsProcessor

from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    apply_multimodal_rotary_pos_emb,
    eager_attention_forward,
)

__all__ = ["AttentionCostCollector", "AttnCostRecorder"]


class AttentionCostCollector:
    """Computes per-token AFIP eq.3 Dkl (cross-head attention inconsistency)
    for a chosen range of Qwen2.5-VL decoder layers, WITHOUT
    `model.generate(output_attentions=True)`.

    Why: output_attentions=True forces every layer's full [heads, q_len,
    kv_len] attention tensor to be computed AND kept alive in `gen_out.
    attentions` for the whole generation -- for an image-heavy prompt
    (~1500+ tokens) x 28 layers x up to `max_new_tokens` steps, this alone
    exceeds a 24GB GPU (confirmed empirically: OOM on every one of 15
    TreeBench examples even after cutting max_new_tokens and clearing the
    cache between examples -- the peak happens *inside* a single generate()
    call, before any of that cleanup runs).

    Instead, this monkey-patches `forward` on only the requested layers
    (matching AFIP's own reference implementation's approach, see
    https://github.com/MIKUZ12/AFIP `src/ih_open/modify_attention.py` --
    adapted here for Qwen2.5-VL's modern modeling code, which that repo's
    compat shim targets the older Qwen-VL architecture and doesn't match).
    Each patched layer's forward computes attention with eager math (needed
    to get weights at all), immediately slices out just the last query
    row's image-token columns (a [heads, n_image_tokens] tensor -- tiny) to
    compute that layer's Dkl contribution, and lets the full [heads, q_len,
    kv_len] tensor be garbage-collected right away instead of returning it.
    Peak memory added is bounded by (patched layers) x (one row's worth of
    attention), not (all layers) x (every step's full matrix).

    Usage:
        collector = AttentionCostCollector(model, layer_range=(5, 18), img_span=img_span)
        ...call model.generate() normally, WITHOUT output_attentions=True...
        # after each generated token, collector.per_layer_costs has one
        # entry per patched layer for that token; average+clear with:
        cost = collector.pop_step_cost()
        ...
        collector.restore()  # always call when done, even on error
    """

    def __init__(self, model, layer_range: Tuple[int, int], img_span: Tuple[int, int], eps: float = 1e-8):
        self.img_span = img_span
        self.eps = eps
        self.per_layer_costs: List[float] = []
        self._patched_modules: Dict[int, torch.nn.Module] = {}
        self._originals: Dict[int, callable] = {}

        layers = model.model.language_model.layers
        lo, hi = layer_range
        self.patched_layers = [i for i in range(lo, hi + 1) if i < len(layers)]
        for i in self.patched_layers:
            attn_module = layers[i].self_attn
            self._patched_modules[i] = attn_module
            self._originals[i] = attn_module.forward
            attn_module.forward = types.MethodType(self._build_forward(), attn_module)

    def _dkl(self, row: torch.Tensor) -> float:
        """row: [heads, n_image_tokens] raw post-softmax attention for the
        last query position. AFIP eq.3: mean cross-head KL divergence
        between each head's image-token attention and the collective
        (head-averaged) distribution, for this one layer."""
        n = row.shape[-1]
        p_head = (row + self.eps) / (row.sum(dim=-1, keepdim=True) + n * self.eps)
        row_avg = row.mean(dim=0)
        p_collective = (row_avg + self.eps) / (row_avg.sum() + n * self.eps)
        kl_per_head = (p_head * (p_head.log() - p_collective.log())).sum(dim=-1)
        return float(kl_per_head.mean().item())

    def _build_forward(collector):
        s, e = collector.img_span

        def patched_forward(
            self,
            hidden_states,
            attention_mask=None,
            position_ids=None,
            past_key_values=None,
            output_attentions: bool = False,
            use_cache: bool = False,
            position_embeddings=None,
            **kwargs,
        ):
            bsz, q_len, _ = hidden_states.size()
            query_states = self.q_proj(hidden_states).view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
            key_states = self.k_proj(hidden_states).view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
            value_states = self.v_proj(hidden_states).view(bsz, q_len, -1, self.head_dim).transpose(1, 2)

            cos, sin = position_embeddings
            query_states, key_states = apply_multimodal_rotary_pos_emb(
                query_states, key_states, cos, sin, self.config.rope_parameters["mrope_section"]
            )
            if past_key_values is not None:
                key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

            attn_output, attn_weights = eager_attention_forward(
                self, query_states, key_states, value_states, attention_mask,
                dropout=0.0, scaling=self.scaling,
            )

            # Last query row (the position predicting the next/current
            # token), image columns only. Everything else about
            # attn_weights is discarded below.
            row = attn_weights[0, :, -1, s:e].detach().to(torch.float32)
            collector.per_layer_costs.append(collector._dkl(row))
            del attn_weights

            attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
            attn_output = self.o_proj(attn_output)
            return attn_output, None

        return patched_forward

    def pop_step_cost(self) -> float:
        """Average Dkl across the patched layers for the token just
        generated, then clear the buffer for the next token. Call this once
        per decoding step (e.g. from a LogitsProcessor, right after the
        forward pass that produced this step's logits)."""
        if not self.per_layer_costs:
            return 0.0
        val = sum(self.per_layer_costs) / len(self.per_layer_costs)
        self.per_layer_costs = []
        return val

    def restore(self) -> None:
        """Undo the monkey-patch. Always call when done (including on
        exceptions) -- an un-restored layer keeps paying eager-attention
        cost and keeps this collector alive via closure."""
        for i, fwd in self._originals.items():
            self._patched_modules[i].forward = fwd
        self._originals.clear()
        self._patched_modules.clear()


class AttnCostRecorder(LogitsProcessor):
    """Bridges an AttentionCostCollector into `proc.step_log` (VDGDLogitsProcessor's
    per-token log, see fovea/logits_processor.py): after each decoding step,
    pops that step's Dkl cost off the collector and writes it into the
    step_log entry `proc` just appended for the same token.

    Must be placed in `LogitsProcessorList` AFTER `proc` -- relies on
    `proc.step_log` already having this step's entry by the time this runs
    (LogitsProcessorList calls processors in list order, each seeing the
    previous one's output; this processor doesn't modify scores, only reads
    proc.step_log, so its position relative to any CFG-style processor that
    *does* modify scores doesn't matter -- only being after `proc` does).
    """

    def __init__(self, collector: AttentionCostCollector, proc):
        self.collector = collector
        self.proc = proc

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        cost = self.collector.pop_step_cost()
        if self.proc.step_log:
            self.proc.step_log[-1]["attn_cost"] = cost
        return scores
