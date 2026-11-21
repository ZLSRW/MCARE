"""Build PERM-format training prompts from the leak-free ED / ESConv training pool.

PERM's own training prompts are not public, so this is an adaptation: each item follows the schema that
external/PERM/perm/empathy_dataset.py reads (persona.identity, scenario, dialogue[0]) and is derived only from
dataset labels and text, never invented. One item per parent conversation, taken at the first turn whose latest
user message has enough content. Parents that appear in the evaluation panel are excluded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[2]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
from common.config import get_paths as _get_paths
_CFG = _get_paths()
ROOT = _CFG.root
POLICY = _CFG.policy_dir
PANEL = _CFG.eval_dir / "panel_all.jsonl"
MIN_WORDS = 6


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def to_item(row: dict) -> dict:
    meta = row["metadata"]
    turns = row["messages"][1:-1]  # the last message is the target reply itself
    query = turns[-1]["content"].strip()
    history = turns[:-1]
    situation = meta.get("situation", "").strip()
    scenario = f"Situation: {situation}" if situation else "The user wants to talk about something on their mind."
    if history:
        scenario += "\nConversation so far:\n" + "\n".join(
            f"{'User' if turn['role'] == 'user' else 'Assistant'}: {turn['content'].strip()}" for turn in history
        )
    if row["source"] == "esconv":
        identity = f"A person dealing with {meta['problem_type']} and feeling {meta['emotion_type']}."
    else:
        identity = f"A person feeling {meta['emotion']}."
    return {
        "id": row["id"], "source": row["source"], "parent_id": row["parent_id"],
        "persona": {"identity": identity}, "scenario": scenario, "dialogue": [query],
    }


def first_usable_per_parent(rows: list[dict], excluded: set[str]) -> list[dict]:
    seen, result = set(), []
    for row in rows:
        if row["parent_id"] in seen or row["parent_id"] in excluded:
            continue
        prompt_last = row["messages"][-2]
        if prompt_last["role"] != "user" or len(prompt_last["content"].split()) < MIN_WORDS:
            continue
        seen.add(row["parent_id"])
        result.append(row)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-ed", type=int, default=2100)
    parser.add_argument("--train-esconv", type=int, default=900)
    parser.add_argument("--valid-ed", type=int, default=100)
    parser.add_argument("--valid-esconv", type=int, default=60)
    parser.add_argument("--seed", type=int, default=20260930)
    args = parser.parse_args()
    args.output_dir = args.output_dir.resolve()

    panel = read_jsonl(PANEL)
    excluded = {row["family"].split(":", 1)[1] for row in panel} | {row["reference_parent_id"] for row in panel if row.get("reference_parent_id")}
    rng = random.Random(args.seed)
    plan = {"train": (("ed_train", args.train_ed), ("esconv_train", args.train_esconv)),
            "valid": (("ed_valid", args.valid_ed), ("esconv_dev", args.valid_esconv))}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"seed": args.seed, "min_query_words": MIN_WORDS, "panel_parents_excluded": len(excluded), "files": {}}
    for split, parts in plan.items():
        items = []
        for name, count in parts:
            pool = first_usable_per_parent(read_jsonl(POLICY / f"{name}.jsonl"), excluded)
            rng.shuffle(pool)
            if len(pool) < count:
                raise ValueError(f"{name}: only {len(pool)} usable parents for {count}")
            items += [to_item(row) for row in pool[:count]]
        rng.shuffle(items)
        overlap = {item["parent_id"] for item in items} & excluded
        if overlap:
            raise ValueError(f"panel overlap: {sorted(overlap)[:3]}")
        path = args.output_dir / f"{split}.jsonl"
        path.write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in items), encoding="utf-8")
        manifest["files"][split] = {
            "path": str(path.relative_to(ROOT)), "rows": len(items), "sha256": sha256(path),
            "by_source": {source: sum(item["source"] == source for item in items) for source in ("ed", "esconv")},
        }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
