"""Contrastive white-box robustness prediction package."""

from contrastive.causal_graph import extract_causal_nodes
from contrastive.contrastive_detector import (
    CausalConsistencyMetrics,
    ContrastiveMetrics,
    compute_causal_consistency_batch,
    extract_contrastive_batch,
)
from contrastive.decision import predict_contrastive_robustness
from contrastive.model_loader import get_or_load_model
from contrastive.perturbation_engine import (
    build_perturbation_triplet,
    generate_evolutionary_variants,
    generate_variants,
)

__all__ = [
    "predict_contrastive_robustness",
    "extract_causal_nodes",
    "generate_evolutionary_variants",
    "compute_causal_consistency_batch",
    "extract_contrastive_batch",
    "CausalConsistencyMetrics",
    "ContrastiveMetrics",
    "generate_variants",
    "build_perturbation_triplet",
    "get_or_load_model",
]
