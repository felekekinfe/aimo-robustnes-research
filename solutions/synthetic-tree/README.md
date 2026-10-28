# Synthetic Tree Robustness Solution

## Unified Final Architecture Blueprint
This solution implements the complete **Final Architecture Blueprint** for the NeurIPS 2026 AIMO Interpretability Challenge:

1. **On-the-Fly Synthetic 7-Type Tree Generator**:
   Generates 14 distinct variants across all 7 official perturbation categories:
   - Group 1: Typos & Formatting (4 branches)
   - Group 2: Rename & Rephrase (4 branches)
   - Group 3: Domain & Distract (4 branches)
   - Group 4: Expert & Sensitivity Stress (2 branches)

2. **Batched Prefill Execution (Batch Size = 15)**:
   Processes Root + 14 branches in a single parallel GPU forward pass in ~0.25s.

3. **Invariant Token Pooling & Early CoT Dynamics**:
   Pools hidden states across middle ($0.50L$) and late ($0.75L$) layers over invariant problem tokens, alongside first-64-token CoT entropy and logit margins.

4. **Type-Stratified Threshold Engine**:
   Evaluates each branch against its dedicated baseline threshold ($\tau_t$) to compute type pass rates and causal sensitivity.

5. **L2-Regularized Linear Meta-Classifier**:
   Fuses model prior, structural invariants, synthetic tree invariance, and CoT dynamics in pure Python without `.pkl` dependencies.
