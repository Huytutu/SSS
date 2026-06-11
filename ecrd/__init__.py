from .evidence import Evidence
from .scorer import EvidenceScorer
from .logits_processor import ECRDLogitsProcessor, knee_topk
from .triggers import MixedGapTrigger

try:
    from .grit_client import GRITClient
except Exception:  # optional dependency for GRIT/Qwen-VL users
    GRITClient = None

__all__ = [
    "Evidence",
    "EvidenceScorer",
    "ECRDLogitsProcessor",
    "MixedGapTrigger",
    "GRITClient",
    "knee_topk",
]
