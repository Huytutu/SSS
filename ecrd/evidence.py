from dataclasses import dataclass
from typing import Optional, List
import torch


@dataclass
class Evidence:
    """A short, verifiable textual evidence item used by ECRD."""
    id: str
    text: str
    bbox: Optional[List[int]] = None
    source: str = "desc"
    time_step: Optional[int] = None

    # Optional cached fields; populated by downstream scorers if needed.
    token_ids: Optional[torch.Tensor] = None
    prefix_logprobs: Optional[torch.Tensor] = None
    vocab_device: Optional[str] = None