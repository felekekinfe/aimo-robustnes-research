"""Test Contrastive White-Box Robustness on Unseen Olympiad Math Problems (AIME / IMO).

Supports both:
1. Pure offline mode with structural symbolic priors.
2. Active GPU neural mode (--neural) measuring live Transformer hidden-state
   drift, cosine similarity, and probe margin stability on CUDA.
"""

from __future__ import annotations

import argparse
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

from contrastive.decision import predict_contrastive_robustness
from contrastive.perturbation_engine import generate_contrastive_variant
from contrastive.symbolic_extractor import extract_feature_dataframe


UNSEEN_OLYMPIAD_PROBLEMS = [
    {
        "id": "AIME-Algebra",
        "domain": "Algebra / Polynomials",
        "problem": (
            "Let $P(x)$ be a polynomial of degree 2024 such that $P(k) = \\frac{1}{k}$ "
            "for each $k = 1, 2, \\dots, 2025$. Find the value of $2026 \\cdot P(2026)$."
        ),
    },
    {
        "id": "AIME-NumberTheory",
        "domain": "Number Theory / Modular Powers",
        "problem": (
            "Find the least positive integer $n$ such that $2^n + 3^n$ is a multiple of $5^7$."
        ),
    },
    {
        "id": "AIME-Combinatorics",
        "domain": "Combinatorics / Extremal Sets",
        "problem": (
            "A subset $S$ of $\\{1, 2, \\dots, 100\\}$ is called sum-free if no two elements "
            "of $S$ sum to an element of $S$. What is the maximum possible number of elements "
            "in a sum-free subset $S$?"
        ),
    },
    {
        "id": "AIME-Geometry",
        "domain": "Geometry / Incircle & Similar Triangles",
        "problem": (
            "In triangle $ABC$, $AB = 13$, $BC = 14$, and $CA = 15$. Let $I$ be the incenter "
            "of $\\triangle ABC$. The line passing through $I$ parallel to $BC$ intersects "
            "$AB$ at $P$ and $AC$ at $Q$. Determine the length of the segment $PQ$."
        ),
    },
]


def evaluate_unseen_olympiad(model_name: str | None = None):
    print("=" * 80)
    print(" EVALUATING CONTRASTIVE WHITE-BOX DETECTOR ON UNSEEN OLYMPIAD PROBLEMS")
    print("=" * 80)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Hardware Device: {device.upper()}")
    if device == "cuda":
        print(f"GPU Model:       {torch.cuda.get_device_name(0)}")
        print(f"Total VRAM:      {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print("-" * 80)

    problems = [item["problem"] for item in UNSEEN_OLYMPIAD_PROBLEMS]

    # 1. Perturbation Engine
    print("\n[1] Meaning-Preserving Equivalence Perturbations (x -> x'):")
    print("-" * 80)
    for item in UNSEEN_OLYMPIAD_PROBLEMS:
        orig = item["problem"]
        variant = generate_contrastive_variant(orig)
        item["variant"] = variant
        print(f"[{item['id']} - {item['domain']}]")
        print(f"  Original:  {orig}")
        print(f"  Variant:   {variant}\n")

    # 2. Symbolic Structural Complexity Profiling
    print("[2] Symbolic Structural Complexity Profiling:")
    print("-" * 80)
    df_feats = extract_feature_dataframe(problems)
    summary_cols = [
        "char_count",
        "math_char_ratio",
        "fraction_count",
        "distinct_variable_count",
        "max_number_magnitude",
        "has_algebra_keyword",
        "has_number_theory_keyword",
        "has_combinatorics_keyword",
        "has_geometry_keyword",
    ]
    df_subset = df_feats[summary_cols].copy()
    df_subset.insert(0, "Problem_ID", [item["id"] for item in UNSEEN_OLYMPIAD_PROBLEMS])
    print(df_subset.to_string(index=False))

    # 3. Active Neural GPU Forward Passes
    if model_name:
        print(f"\n[3] Active Neural GPU Inference on '{model_name}':")
        print("-" * 80)
        dtype = torch.float16 if device == "cuda" else torch.float32
        print(f"Loading '{model_name}' onto {device.upper()}...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype).to(device)
        model.eval()

        print(f"\n{'Problem ID':<20} | {'Cosine Sim':<12} | {'Latent Drift':<14} | {'Neural Stability':<18}")
        print("-" * 72)

        neural_stabilities = []
        for item in UNSEEN_OLYMPIAD_PROBLEMS:
            inp_orig = tokenizer(item["problem"], return_tensors="pt").to(device)
            inp_var = tokenizer(item["variant"], return_tensors="pt").to(device)

            with torch.no_grad():
                out_orig = model(**inp_orig, output_hidden_states=True, return_dict=True, use_cache=False)
                out_var = model(**inp_var, output_hidden_states=True, return_dict=True, use_cache=False)

            # Final layer hidden representation
            v_orig = out_orig.hidden_states[-1][0, -1, :].detach().float().cpu().numpy()
            v_var = out_var.hidden_states[-1][0, -1, :].detach().float().cpu().numpy()

            norm_orig = np.linalg.norm(v_orig)
            norm_var = np.linalg.norm(v_var)
            cos_sim = float(np.dot(v_orig, v_var) / max(1e-8, (norm_orig * norm_var)))
            cos_sim = max(-1.0, min(1.0, cos_sim))
            drift = 1.0 - cos_sim

            if drift < 0.03:
                status = "STABLE"
                is_stable = True
            else:
                status = "HIGH DRIFT"
                is_stable = False

            neural_stabilities.append((cos_sim, drift, is_stable))
            print(f"{item['id']:<20} | {cos_sim:<12.4f} | {drift:<14.4f} | {status:<18}")

    # 4. Final Solution Decisions
    print("\n[4] Final Decision Output (Combined Neuro-Symbolic / Contrastive):")
    print("-" * 80)
    predictions = predict_contrastive_robustness(
        model_id=model_name or "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
        reasoning_effort="default",
        problems=problems,
    )

    for item, pred in zip(UNSEEN_OLYMPIAD_PROBLEMS, predictions):
        status = "ROBUST (True)" if pred else "FRAGILE (False)"
        print(f"  * {item['id']:<20} [{item['domain']:<32}] => {status}")

    print("\n" + "=" * 80)
    print(f" Total Problems Evaluated: {len(problems)}")
    print(f" Predicted Robust:         {predictions.count(True)} / {len(problems)} ({predictions.count(True)/len(problems):.1%})")
    print(f" Predicted Fragile:        {predictions.count(False)} / {len(problems)} ({predictions.count(False)/len(problems):.1%})")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--neural", action="store_true", help="Run active GPU neural forward passes")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    args = parser.parse_args()

    active_model = args.model if args.neural else None
    evaluate_unseen_olympiad(active_model)
