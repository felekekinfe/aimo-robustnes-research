import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "solutions" / "synthetic-tree"))
from solution import are_robust

with open(ROOT / "data" / "train-main-v2.jsonl", "r", encoding="utf-8") as f:
    records = [json.loads(line) for line in f if line.strip()]

groups = {}
for i, r in enumerate(records):
    key = (r["model_id"], r.get("reasoning_effort", "default"))
    groups.setdefault(key, []).append((i, r))

preds = [None] * len(records)
for (model_id, effort), items in groups.items():
    idxs = [it[0] for it in items]
    probs = [it[1]["problem"] for it in items]
    out = are_robust(model_id=model_id, reasoning_effort=effort, problems=probs)
    for idx, p in zip(idxs, out):
        preds[idx] = p

print("=== ALL MISCLASSIFICATIONS (TOTAL 94) ===")
long_mistakes = []
for i, (r, p) in enumerate(zip(records, preds)):
    act = bool(r["is_robust"])
    if p != act:
        l = len(r["problem"])
        m_id = r["model_id"]
        eff = r.get("reasoning_effort")
        drop = r.get("max_drop")
        snippet = r["problem"][:100].replace("\n", " ")
        print(f"[{i:02d}] Len={l:4d} | Act={str(act):5s} Pred={str(p):5s} | Model={m_id} (effort={eff}) | Drop={drop}")
        print(f"     Text: {snippet}...")
        if l >= 450:
            long_mistakes.append(i)

print(f"\nTotal mistakes: {sum(p != bool(r['is_robust']) for r, p in zip(records, preds))}/94")
print(f"Mistakes on len >= 450: {len(long_mistakes)} / {sum(1 for r in records if len(r['problem']) >= 450)}")
