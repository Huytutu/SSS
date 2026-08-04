from __future__ import annotations
from typing import Any, List, Optional

import torch
import torch.nn.functional as F

__all__ = ["VisionScorer"]


class VisionScorer:
    """ReVisiT-style vision-token grounding (arXiv:2506.09522).

    Projects vision-token hidden states through the LM head into vocabulary
    space, so VDGDLogitsProcessor can JSD-select the vision token whose
    projected distribution best matches the current decoding context and
    fold it into the log-space refinement alongside TextScorer.

    Unlike the ReVisiT reference implementation (which caches the full
    projected+normalized softmax over the whole vocabulary for every
    (layer, vision_token) pair up front), set_prompt_from_hidden_states()
    caches raw hidden states only -- a few tens of MB rather than several GB
    for ~1000 vision tokens x ~14 layers x ~150k vocab. score_token_ids()
    does the LM-head projection lazily, restricted to the small per-step
    candidate set, so no repeated transformer forward pass is needed during
    decoding either way.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        image_token_id: Optional[int] = None,
        layers: Optional[List[int]] = None,
        backend_model: Optional[torch.nn.Module] = None,
    ) -> None:
        self.model = model
        self.backend_model = backend_model or model
        self.image_token_id = int(image_token_id) if image_token_id is not None else int(model.config.image_token_id)
        self.layers = layers
        try:
            self.device = str(next(self.backend_model.parameters()).device)
        except Exception:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._hidden_cache: Optional[torch.Tensor] = None  # [J, |v|, H]
        self._lm_head: Optional[torch.nn.Module] = None
        self.image_grid_thw: Optional[torch.Tensor] = None  # for idx_to_rc(); single-image only

    def _default_layers(self) -> List[int]:
        """Mirrors the ReVisiT reference's `early_exit_layers == "all"`:
        every even decoder layer, starting from 2 if word embeddings are
        tied (layer 0 would be an identity early-exit otherwise) or 0
        if not, up to and including the final layer."""
        text_config = self.model.config.get_text_config()
        final_layer = int(text_config.num_hidden_layers)
        start_layer = 2 if bool(getattr(text_config, "tie_word_embeddings", False)) else 0
        return list(range(start_layer, final_layer + 1, 2))

    @torch.inference_mode()
    def set_prompt_from_hidden_states(
        self, hidden_states: Any, input_ids: torch.Tensor, image_grid_thw: Optional[torch.Tensor] = None
    ) -> None:
        """hidden_states: the `output_hidden_states=True` tuple from the same
        forward pass TextScorer._cache_from_logits consumes. input_ids: the
        corresponding (1, seq_len) prompt token ids. image_grid_thw (optional):
        the processor's (num_images, 3) grid tensor, kept only so idx_to_rc()
        can map a vision-token index back to a spatial patch for visualization
        -- not used by the scoring path itself."""
        vis_pos = (input_ids[0] == self.image_token_id).nonzero(as_tuple=True)[0]
        if vis_pos.numel() == 0:
            raise ValueError(f"No vision tokens found for image_token_id={self.image_token_id} in input_ids.")

        layers = self.layers or self._default_layers()
        self.layers = layers
        stacked = torch.stack([hidden_states[j][0, vis_pos, :] for j in layers], dim=0)  # [J, |v|, H]
        self._hidden_cache = stacked.detach().to(self.device)
        self._lm_head = self.model.get_output_embeddings()
        self.image_grid_thw = image_grid_thw

    def grid_hw(self) -> tuple[int, int]:
        """Merged-grid (rows, cols) for the single cached image -- the model's
        vision tower reverses its internal window-attention token shuffle
        before merging, so vision-token index i is a plain row-major index
        into this grid (see Qwen2_5_VisionTransformerPretrainedModel.forward,
        `reverse_indices = torch.argsort(window_index)` applied pre-merge)."""
        if self.image_grid_thw is None:
            raise RuntimeError("image_grid_thw was not provided to set_prompt_from_hidden_states().")
        merge = int(self.model.config.vision_config.spatial_merge_size)
        _, h, w = [int(x) for x in self.image_grid_thw[0]]
        return h // merge, w // merge

    def idx_to_rc(self, vision_idx: int) -> tuple[int, int]:
        """Row-major (row, col) of a vision-token index within grid_hw()."""
        _, cols = self.grid_hw()
        return divmod(int(vision_idx), cols)

    @torch.no_grad()
    def score_token_ids(self, cand_idx: torch.Tensor) -> torch.Tensor:
        """Returns log-probabilities, renormalized over `cand_idx`, for every
        cached (layer, vision_token) pair: shape [J, |v|, k]."""
        if self._hidden_cache is None:
            raise RuntimeError("VisionScorer.set_prompt_from_hidden_states() must be called before scoring.")
        cand_idx = cand_idx.to(device=self.device, dtype=torch.long)
        weight = self._lm_head.weight[cand_idx]  # [k, H]
        logits = torch.matmul(self._hidden_cache.to(torch.float32), weight.to(torch.float32).T)  # [J, |v|, k]
        bias = getattr(self._lm_head, "bias", None)
        if bias is not None:
            logits = logits + bias[cand_idx].to(torch.float32)
        return F.log_softmax(logits, dim=-1)
