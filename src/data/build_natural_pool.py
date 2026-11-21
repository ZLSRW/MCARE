"""Build the parent-isolated natural dialogue training pool (ESConv-train + ED-train).

Selects 1,000 ESConv-train and 1,000 ED-train histories (parent-isolated against all
validation and test splits) for Stage-1/Stage-2 training and matched-pool baselines.
Selection is strictly deterministic using SHA-256 hash ordering:
  - ESConv: all 910 training parents contribute one history, plus 90 deterministically ranked parents contribute a second disjoint turn prefix.
  - ED: 1,000 deterministically ranked training parents contribute one history each.
Outputs (outputs/natural_pool/): prompts.jsonl, sft_natural_2000.jsonl, manifest.json.
usage: PYTHONPATH=src python src/data/build_natural_pool.py"""
import hashlib
import json
from collections import Counter
from pathlib import Path

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
from common.config import get_paths

_CFG = get_paths()
ROOT = _CFG.root
POL = _CFG.policy_dir
OUT = _CFG.exp_dir / "natural_pool"
SEED = 20261011


def h(*parts):
    return hashlib.sha256("|".join(str(p) for p in (SEED,) + parts).encode()).hexdigest()


def load(name):
    return [json.loads(line) for line in open(POL / f"{name}.jsonl", encoding="utf-8")]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    forbidden = {name: {r["parent_id"] for r in load(name)} for name in ("esconv_dev", "esconv_test", "ed_valid", "ed_test")}
    train = {"esconv": load("esconv_train"), "ed": load("ed_train")}
    by_parent = {}
    for src, rows in train.items():
        for r in rows:
            assert r["messages"][-1]["role"] == "assistant" and r["messages"][-2]["role"] == "user" and r["target"] == r["messages"][-1]["content"]
            by_parent.setdefault((src, r["parent_id"]), []).append(r)
    picks = []                                                     # (source, example)
    es_parents = sorted((p for (s, p) in by_parent if s == "esconv"), key=lambda p: h("parent", p))
    for p in es_parents:
        picks.append(("esconv", min(by_parent[("esconv", p)], key=lambda e: h("ex", e["id"]))))
    for p in es_parents[:90]:
        ex = sorted(by_parent[("esconv", p)], key=lambda e: h("ex", e["id"]))
        if len(ex) > 1:
            picks.append(("esconv", ex[1]))
    es_n = sum(1 for s, _ in picks if s == "esconv")
    ed_parents = sorted((p for (s, p) in by_parent if s == "ed"), key=lambda p: h("parent", p))[:2000 - es_n]
    for p in ed_parents:
        picks.append(("ed", min(by_parent[("ed", p)], key=lambda e: h("ex", e["id"]))))
    assert len(picks) == 2000 and len({e["id"] for _, e in picks}) == 2000, len(picks)
    used = {s: {e["parent_id"] for ss, e in picks if ss == s} for s in ("esconv", "ed")}
    overlap = {"esconv_dev": len(used["esconv"] & forbidden["esconv_dev"]), "esconv_test": len(used["esconv"] & forbidden["esconv_test"]), "ed_valid": len(used["ed"] & forbidden["ed_valid"]), "ed_test": len(used["ed"] & forbidden["ed_test"])}
    assert all(v == 0 for v in overlap.values()), overlap
    with open(OUT / "prompts.jsonl", "w", encoding="utf-8") as hp, open(OUT / "sft_natural_2000.jsonl", "w", encoding="utf-8") as hs:
        for src, e in sorted(picks, key=lambda x: x[1]["id"]):
            hp.write(json.dumps({"id": "nat:" + e["id"], "source": src, "parent_id": e["parent_id"], "messages": e["messages"][:-1], "target": e["target"]}, ensure_ascii=False) + "\n")
            hs.write(json.dumps({"messages": e["messages"]}, ensure_ascii=False) + "\n")
    turns = Counter(len(e["messages"]) - 1 for _, e in picks)
    json.dump({"seed": SEED, "prompts": 2000, "by_source": dict(Counter(s for s, _ in picks)), "parents": {s: len(v) for s, v in used.items()}, "parent_overlap_with_dev_test_valid": overlap,
               "history_messages_incl_system": {str(k): v for k, v in sorted(turns.items())},
               "note": "natural train histories only; no appended requirement sentence, no rules; target = the original next reply of the public conversation; K=8 candidates are sampled from Qwen3.5-9B (T=1.0, top-p 0.95) in the next step"},
              open(OUT / "manifest.json", "w"), indent=1)
    print("prompts", len(picks), dict(Counter(s for s, _ in picks)), "overlap", overlap)


if __name__ == "__main__":
    main()
