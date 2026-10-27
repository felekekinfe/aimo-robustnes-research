"""Comprehensive dataset and submission inspection for STEP 0."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

def main() -> None:
    # 1. val-sample-raw (28 rows)
    val_raw_path = DATA_DIR / "val-sample-raw.jsonl"
    if val_raw_path.is_file():
        val_raw = [json.loads(l) for l in val_raw_path.open(encoding="utf-8")]
        print("=== 1. VAL-SAMPLE RAW (28 rows from Hugging Face aimo-interp/val-sample) ===")
        print("Columns:", sorted(list(val_raw[0].keys())))
        print("Models:", dict(Counter(r["model_id"] for r in val_raw)))
        print("Distinct original_problem:", len(set(r["original_problem"] for r in val_raw)))
        print("Distinct problem_id:", len(set(r["problem_id"] for r in val_raw)))
        print("Label counts (model_is_robust):", dict(Counter(r["model_is_robust"] for r in val_raw)))
        print("\nVal-sample raw: Label counts per model:")
        for m in sorted(set(r["model_id"] for r in val_raw)):
            m_rows = [r for r in val_raw if r["model_id"] == m]
            print(f"  {m}: {dict(Counter(r['model_is_robust'] for r in m_rows))} (total {len(m_rows)})")
        print("\nVal-sample raw: Label counts per Delta (permutation_type):")
        for d, count in Counter(r["permutation_type"] for r in val_raw).most_common():
            d_rows = [r for r in val_raw if r["permutation_type"] == d]
            print(f"  {d}: {dict(Counter(r['model_is_robust'] for r in d_rows))} (total {len(d_rows)})")

    # 2. val-sample Codabench ingestion format (24 rows)
    cases_path = DATA_DIR / "val-sample" / "input" / "cases.jsonl"
    labels_path = DATA_DIR / "val-sample" / "reference" / "labels.jsonl"
    val_cases = [json.loads(l) for l in cases_path.open(encoding="utf-8")]
    val_labels = {json.loads(l)["id"]: json.loads(l)["is_robust"] for l in labels_path.open(encoding="utf-8")}
    print("\n=== 2. VAL-SAMPLE CODABENCH INGESTION FORMAT (24 rows) ===")
    print("Cases columns:", sorted(list(val_cases[0].keys())))
    print("Labels columns:", ["id", "is_robust"])
    print("Models:", dict(Counter(r["model_id"] for r in val_cases)))
    print("Distinct problem strings:", len(set(r["problem"] for r in val_cases)))
    print("Overall label counts:", dict(Counter(val_labels.values())))
    print("Label counts per model:")
    for m in sorted(set(r["model_id"] for r in val_cases)):
        m_cases = [c for c in val_cases if c["model_id"] == m]
        print(f"  {m}: {dict(Counter(val_labels[c['id']] for c in m_cases))} (total {len(m_cases)})")

    # 3. train-main-v2 (94 rows)
    train_path = DATA_DIR / "train-main-v2.jsonl"
    train_v2 = [json.loads(l) for l in train_path.open(encoding="utf-8")]
    print("\n=== 3. TRAIN-MAIN-V2 (94 rows) ===")
    print("Columns:", sorted(list(train_v2[0].keys())))
    print("Models:", dict(Counter(r["model_id"] for r in train_v2)))
    print("Distinct problem strings:", len(set(r["problem"] for r in train_v2)))
    print("Overall label counts:", dict(Counter(r["is_robust"] for r in train_v2)))
    print("Label counts per model:")
    for m in sorted(set(r["model_id"] for r in train_v2)):
        m_rows = [r for r in train_v2 if r["model_id"] == m]
        print(f"  {m}: {dict(Counter(r['is_robust'] for r in m_rows))} (total {len(m_rows)})")

    # 4. augmented-sample-math-agg (141 rows)
    aug_path = DATA_DIR / "augmented-sample-math-agg.jsonl"
    aug = [json.loads(l) for l in aug_path.open(encoding="utf-8")]
    print("\n=== 4. AUGMENTED-SAMPLE-MATH-AGG (141 rows) ===")
    print("Columns:", sorted(list(aug[0].keys())))
    print("Models:", dict(Counter(r["model_id"] for r in aug)))
    print("Distinct original_problem:", len(set(r["original_problem"] for r in aug)))
    print("Distinct problem_id:", len(set(r["problem_id"] for r in aug)))
    print("Overall label counts:", dict(Counter(r["model_is_robust"] for r in aug)))
    print("Label counts per Delta (top 10):")
    for d, count in Counter(str(r["permutation_type"]) for r in aug).most_common(10):
        d_rows = [r for r in aug if str(r["permutation_type"]) == d]
        print(f"  {d}: {dict(Counter(r['model_is_robust'] for r in d_rows))} (total {len(d_rows)})")

    # 5. Overlaps across datasets
    val_probs = set(c["problem"].strip() for c in val_cases)
    train_probs = set(c["problem"].strip() for c in train_v2)
    aug_probs = set(c["original_problem"].strip() for c in aug)
    print("\n=== 5. CROSS-DATASET PROBLEM TEXT OVERLAP ===")
    print("val-sample distinct problems:", len(val_probs))
    print("train-main-v2 distinct problems:", len(train_probs))
    print("augmented-sample-math-agg distinct problems:", len(aug_probs))
    print("Overlap val-sample & train-main-v2:", len(val_probs & train_probs))
    print("Overlap val-sample & augmented:", len(val_probs & aug_probs))
    print("Overlap train-main-v2 & augmented:", len(train_probs & aug_probs))

if __name__ == "__main__":
    main()
