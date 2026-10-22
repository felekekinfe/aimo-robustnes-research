"""Configuration parameters and constants for the Neuro-Symbolic solution."""

from __future__ import annotations

# Model aliases mapping Codabench aliases to exact Hugging Face checkpoint IDs
MODEL_ALIASES: dict[str, str] = {
    "qwen3-8b:low": "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
    "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B": "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
}

# Fusion weight: when neural probe is available, combined_prob = ALPHA * neural_prob + (1 - ALPHA) * symbolic_prob
FUSION_ALPHA: float = 0.65

# Combined decision threshold
DECISION_THRESHOLD: float = 0.50

# Fallback decision threshold when neural model is unavailable (calibrated on training data)
FALLBACK_DECISION_THRESHOLD: float = 0.40

# Filenames
NEURAL_PROBE_NAME = "probe_artifact.pkl"
SYMBOLIC_ARTIFACT_NAME = "symbolic_pipeline.joblib"
