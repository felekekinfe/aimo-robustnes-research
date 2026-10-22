"""Interactive demonstration of Active GPU Latent Representation Drift.

Runs real forward passes on GPU (CUDA) to measure:
1. Meaning-preserving perturbation generation (x -> x').
2. Layer-by-layer internal hidden state extraction.
3. Cosine similarity and latent representation drift: 1 - cos(h(x), h(x')).
4. Geometric stability comparison across shallow vs deep layers.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
SOLUTION_DIR = ROOT / "solutions" / "contrastive-whitebox"
if str(SOLUTION_DIR) not in sys.path:
    sys.path.insert(0, str(SOLUTION_DIR))

from contrastive.perturbation_engine import generate_contrastive_variant


def run_gpu_drift_demo(model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"):
    print("=" * 70)
    print(" ACTIVE GPU LATENT REPRESENTATION DRIFT DEMONSTRATION")
    print("=" * 70)

    # 1. Check Hardware
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Hardware Device: {device.upper()}")
    if device == "cuda":
        print(f"GPU Model:       {torch.cuda.get_device_name(0)}")
        print(f"Allocated VRAM:  {torch.cuda.memory_allocated(0) / 1e6:.1f} MB")
        print(f"Total VRAM:      {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("Note: Running on CPU. For maximum speed, ensure Colab Runtime is T4 GPU.")
    print("-" * 70)

    # 2. Load Model & Tokenizer
    print(f"\n[1/4] Loading model checkpoint '{model_name}'...")
    start_load = time.time()
    dtype = torch.float16 if device == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype)
    model.to(device)
    model.eval()
    print(f"      Model loaded in {time.time() - start_load:.2f}s.")

    # 3. Test Problems
    test_cases = [
        {
            "name": "Algebraic Quadratic",
            "problem": "Find all real values of $x$ such that $x^2 + 5x + 6 = 0$.",
        },
        {
            "name": "Modular Arithmetic & Inequalities",
            "problem": "What integer $n$ satisfies $0 \\le n < 18$ and $n \\equiv -11213141 \\pmod{18}$?",
        },
        {
            "name": "Geometry & Triangles",
            "problem": "In triangle $ABC$ with sides $a = 5, b = 6, c = 7$, compute the area.",
        },
    ]

    print("\n[2/4] Generating meaning-preserving perturbations (x -> x')...")
    for case in test_cases:
        p_orig = case["problem"]
        p_var = generate_contrastive_variant(p_orig)
        case["variant"] = p_var
        print(f"\n  Problem:   '{p_orig}'")
        print(f"  Variant:   '{p_var}'")

    print("\n[3/4] Running forward passes and extracting layer-by-layer hidden states...")

    for case in test_cases:
        p_orig = case["problem"]
        p_var = case["variant"]

        # Tokenize
        inputs_orig = tokenizer(p_orig, return_tensors="pt").to(device)
        inputs_var = tokenizer(p_var, return_tensors="pt").to(device)

        with torch.no_grad():
            out_orig = model(**inputs_orig, output_hidden_states=True, return_dict=True, use_cache=False)
            out_var = model(**inputs_var, output_hidden_states=True, return_dict=True, use_cache=False)

        hs_orig = out_orig.hidden_states  # tuple of (num_layers + 1)
        hs_var = out_var.hidden_states
        num_layers = len(hs_orig) - 1

        print(f"\n" + "=" * 70)
        print(f" Case: {case['name']} (Model Depth: {num_layers} layers)")
        print(f"{'Layer':<10} | {'Cosine Sim':<15} | {'Latent Drift':<15} | {'Relative L2 Dist':<15}")
        print("-" * 65)

        # Sample across layers: Embedding (0), 25%, 50%, 75%, and Final layer
        sampled_layers = sorted(list({0, num_layers // 4, num_layers // 2, (3 * num_layers) // 4, num_layers}))

        final_drift = 0.0
        final_sim = 1.0

        for layer_idx in sampled_layers:
            # Extract final prompt token vector
            v_orig = hs_orig[layer_idx][0, -1, :].detach().float().cpu().numpy()
            v_var = hs_var[layer_idx][0, -1, :].detach().float().cpu().numpy()

            norm_orig = np.linalg.norm(v_orig)
            norm_var = np.linalg.norm(v_var)
            cos_sim = float(np.dot(v_orig, v_var) / max(1e-8, (norm_orig * norm_var)))
            cos_sim = max(-1.0, min(1.0, cos_sim))
            drift = 1.0 - cos_sim
            l2_rel = float(np.linalg.norm(v_orig - v_var) / max(1e-8, 0.5 * (norm_orig + norm_var)))

            layer_label = f"Layer {layer_idx}" if layer_idx > 0 else "Embeddings"
            if layer_idx == num_layers:
                layer_label = f"Final ({layer_idx})"
                final_drift = drift
                final_sim = cos_sim

            print(f"{layer_label:<10} | {cos_sim:<15.4f} | {drift:<15.4f} | {l2_rel:<15.4f}")

        # Mechanistic interpretation
        print("-" * 65)
        if final_drift < 0.05:
            assessment = "HIGHLY STABLE (Robust to perturbation)"
        elif final_drift < 0.12:
            assessment = "MODERATE STABILITY (Acceptable invariance)"
        else:
            assessment = "HIGH DRIFT (Fragile representation under rewrite)"
        print(f" White-Box Assessment: {assessment}")

    print("\n" + "=" * 70)
    print(" DEMO COMPLETE: Real GPU forward passes and latent metrics confirmed.")
    print("=" * 70)


if __name__ == "__main__":
    model = "Qwen/Qwen2.5-1.5B-Instruct" if len(sys.argv) <= 1 else sys.argv[1]
    run_gpu_drift_demo(model)
