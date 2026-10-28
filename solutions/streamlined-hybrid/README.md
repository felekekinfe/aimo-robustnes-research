# Streamlined Hybrid Robustness Solution

## Approach Overview
This solution implements the **Streamlined Hybrid Approach** for the NeurIPS 2026 AIMO Interpretability Challenge:
1. **Prefill Micro-Perturbation Stability**: Fast 2-pass prefill measuring cosine similarity across layer quartiles (25%, 50%, 75%, 100%) and linear CKA under meaning-preserving variable/syntax substitution.
2. **First-64-Token CoT Dynamics**: Inspects epistemic friction via token Shannon entropy (mean + std) and Top-1 vs Top-2 logit margin.
3. **Activation Stable Rank**: Measures effective geometric dimensionality $\text{SR} = \|H\|_F^2 / \|H\|_2^2$ to detect low-dimensional representation collapse.
4. **L2-Regularized Linear Ensemble**: Evaluates normalized features using analytical weights compiled directly in pure Python (zero `.pkl` or `.joblib` dependencies).

## Operational Guarantees
- Single-GPU RTX 6000 96GB execution in < 3600 seconds.
- Persistent model caching across batches (saves ~2,800s).
- Closed and oversized model fallback guards.
- Offline container compliant (`local_files_only=True`).
- Strict per-problem timeout (never crashes, never hangs).
