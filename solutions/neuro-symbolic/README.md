# Neuro-Symbolic Hybrid Baseline

This solution fuses two complementary views of mathematical reasoning robustness:

1. **Neural Representation Branch**:
   - Evaluates linear probe ensembles on Layer 31 hidden states of the target reasoning model (`deepseek-ai/DeepSeek-R1-0528-Qwen3-8B`), assessing internal conceptual stability.
2. **Symbolic & Structural Branch**:
   - Extracts 27 deterministic numerical features capturing mathematical markup density, LaTeX equations, variables, numeric complexity, and problem domain signals.
   - Evaluates a calibrated decision tree ensemble.
3. **Adaptive Fusion & Safe Fallback**:
   - When offline model weights and GPU acceleration are present (evaluation container), decisions are fused: `0.65 * p_neural + 0.35 * p_symbolic`.
   - When model weights are absent or loading fails (e.g., local CPU smoke testing or unmapped large models), the method cleanly falls back to the symbolic model, guaranteeing 100% coverage and zero invalid predictions.

## Directory Layout

```text
neuro-symbolic/
├── solution.py                 # Codabench entrypoint
├── neuro_symbolic/
│   ├── config.py               # Constants, model aliases, thresholds
│   ├── symbolic_extractor.py   # 27-feature symbolic extractor
│   ├── neural_probe.py         # Forward pass & hidden-state probe margin scorer
│   └── fusion.py               # Neuro-symbolic decision fusion & fallback
├── artifacts/
│   └── symbolic_pipeline.joblib # Trained symbolic classifier artifact
└── probe_artifacts/
    └── probe_artifact.pkl      # Pretrained neural probe ensemble artifact
```
