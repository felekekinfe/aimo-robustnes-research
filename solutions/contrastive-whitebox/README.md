# Contrastive White-Box Robustness Predictor

This solution tests mathematical reasoning robustness by directly measuring **internal representation stability (latent invariance)** under meaning-preserving problem perturbations.

## Scientific Motivation

A mathematically robust language model should map semantically identical problem formulations to nearby latent states. When a harmless perturbation (such as variable substitution $x \to t$ or LaTeX formatting equivalences $\le \leftrightarrow \leq$) causes large representational drift $\|\mathbf{h}(x) - \mathbf{h}(x')\|$, the model's internal geometry is fragile.

## Architecture

1. **Perturbation Engine**: Generates meaning-preserving variants $x \to x'$ (variable substitution, LaTeX syntactic equivalences, framing rephrasing).
2. **Contrastive Forward Passes**: Extracts Layer 31 prompt hidden states $\mathbf{h}(x)$ and $\mathbf{h}(x')$.
3. **White-Box Invariance Metrics**:
   - Cosine Similarity: $\cos(\mathbf{h}(x), \mathbf{h}(x'))$
   - Latent Representation Drift: $1 - \cos(\mathbf{h}(x), \mathbf{h}(x'))$
   - Probe Margin Agreement: $\text{margin}(x)$ vs $\text{margin}(x')$
4. **Adaptive Decision & Fallback**:
   - Fuses latent invariance with probe margins and symbolic priors.
   - On CPU smoke tests or missing weights, falls back gracefully to the calibrated symbolic classifier, guaranteeing 100% coverage and zero invalid predictions.
