from .evidence import Evidence
from .scorer import EvidenceScorer
from .logits_processor import ECRDLogitsProcessor, knee_topk
from .triggers import MixedGapTrigger
from .branch_probe import find_earliest_divergence, default_extract_answer
from .rethink_loop import ecrd_generate_with_rethink

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
    "find_earliest_divergence",
    "default_extract_answer",
    "ecrd_generate_with_rethink",
]
