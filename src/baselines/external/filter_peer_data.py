"""Leak-filter the English text part of the PEER release against the evaluation panel and every non-train split of the evaluation corpora.

A PEER row is dropped when its text contains, verbatim (lower-cased word tokens), an utterance of at least 8 words (user or assistant turn) that belongs to
  * an evaluation-panel parent conversation, or
  * any ESConv dev / test dialogue, or any EmpatheticDialogues valid / test dialogue,
or when it shares a 12-gram with an evaluation-panel user turn. ESConv-train utterances of non-panel parents are allowed. Same rule family as build_escrag_data.py
(there restricted to ESConv and user turns); here ED is included and counselor turns too because PEER rows pack whole transcripts.
usage: filter_peer_data.py [--src peer_repro/data] [--out peer_repro_clean/data]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[2]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
from baselines.external.esc_repro_common import E, PANEL, ROOT, panel_excluded_parents, read_jsonl

from common.config import get_paths as _get_paths
POLICY = _get_paths().policy_dir
WORD = re.compile(r"\w+")


def words(text: str) -> list[str]:
    return WORD.findall(text.lower())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=Path, default=E / "peer_repro/data")
    parser.add_argument("--out", type=Path, default=E / "peer_repro_clean/data")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    excluded = panel_excluded_parents()
    panel_grams = set()
    for row in read_jsonl(PANEL):
        for message in row["messages"]:
            if message["role"] == "user" and len(words(message["content"])) >= 12:
                tokens = words(message["content"])
                panel_grams |= {" ".join(tokens[i:i + 12]) for i in range(len(tokens) - 11)}

    # forbidden utterances, indexed by their first 8 words
    forbidden: dict[str, list[tuple[str, ...]]] = defaultdict(list)
    sources = [("esconv", s) for s in ("train", "dev", "test")] + [("ed", s) for s in ("train", "valid", "test")]
    count = 0
    for source, split in sources:
        for row in read_jsonl(POLICY / f"{source}_{split}.jsonl"):
            if split == "train" and row["parent_id"] not in excluded:
                continue
            for message in row["messages"][1:] + [{"role": "assistant", "content": row.get("target", "")}]:
                tokens = words(message["content"])
                if len(tokens) >= 8:
                    forbidden[" ".join(tokens[:8])].append(tuple(tokens))
                    count += 1
    print(f"forbidden utterances: {count}")

    def leaks(text: str) -> bool:
        tokens = words(text)
        for i in range(len(tokens) - 7):
            for candidate in forbidden.get(" ".join(tokens[i:i + 8]), ()):
                if tuple(tokens[i:i + len(candidate)]) == candidate:
                    return True
        return any(" ".join(tokens[i:i + 12]) in panel_grams for i in range(len(tokens) - 11))

    manifest = {"source": str(args.src), "rule": "drop rows containing a verbatim >=8-word utterance of a panel parent / ESConv dev,test / ED valid,test, or a 12-gram of a panel user turn", "files": {}}
    for name in ("sft_text_en.jsonl", "grpo_text_en.jsonl", "ser_en_text_train.jsonl", "ser_en_text_test.jsonl"):
        rows = read_jsonl(args.src / name)
        kept = [row for row in rows if not leaks(" ".join(m["content"] for m in row["messages"] if m["role"] != "system"))]
        text = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in kept)
        (args.out / name).write_text(text, encoding="utf-8")
        manifest["files"][name] = {"before": len(rows), "after": len(kept), "sha256": hashlib.sha256(text.encode()).hexdigest()}
        print(name, len(rows), "->", len(kept))
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
