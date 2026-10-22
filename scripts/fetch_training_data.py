"""Fetch public upstream training datasets from Hugging Face datasets-server.

Datasets:
1. aimo-interp/train-main-v2 (official training set for main track, 94 examples)
2. aimo-interp/augmented-sample-math-agg (141 examples used by the uncertainty baseline)
"""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"


def fetch_all_rows(dataset: str, config: str, split: str) -> list[dict]:
    """Fetch all rows from Hugging Face datasets-server via pagination."""
    rows = []
    offset = 0
    limit = 100
    while True:
        url = (
            f"https://datasets-server.huggingface.co/rows"
            f"?dataset={dataset}&config={config}&split={split}&offset={offset}&limit={limit}"
        )
        req = urllib.request.Request(url, headers={"User-Agent": "aimo-research-pipeline/0.1"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.load(resp)
        batch = [item["row"] for item in data.get("rows", [])]
        if not batch:
            break
        rows.extend(batch)
        offset += len(batch)
        if offset >= data.get("num_rows_total", offset):
            break
    return rows


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # 1. aimo-interp/train-main-v2
    print("Fetching aimo-interp/train-main-v2...")
    train_rows = fetch_all_rows("aimo-interp/train-main-v2", "default", "train")
    train_path = DATA_DIR / "train-main-v2.jsonl"
    with train_path.open("w", encoding="utf-8") as f:
        for r in train_rows:
            f.write(json.dumps(r, sort_keys=True) + "\n")
    print(f"Saved {len(train_rows)} rows to {train_path}")

    # 2. aimo-interp/augmented-sample-math-agg
    print("Fetching aimo-interp/augmented-sample-math-agg...")
    agg_rows = fetch_all_rows("aimo-interp/augmented-sample-math-agg", "default", "validation")
    agg_path = DATA_DIR / "augmented-sample-math-agg.jsonl"
    with agg_path.open("w", encoding="utf-8") as f:
        for r in agg_rows:
            f.write(json.dumps(r, sort_keys=True) + "\n")
    print(f"Saved {len(agg_rows)} rows to {agg_path}")


if __name__ == "__main__":
    main()
