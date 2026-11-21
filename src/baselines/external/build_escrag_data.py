"""Build the data files for the ESCRAG-R1 reproduction (arXiv 2608.21925, code Matcha-Liu/ESCRAG-R1 @76dae5c).

Released with the paper and used as is: 3,300 + 367 ESC-Preference pairs (reward model), the 3,659-entry retrieval database
with bge-m3 embeddings, both RAG prompts and the training scripts. Not released and therefore rebuilt here:
  * the cold-start SFT demonstrations (paper: 500 GPT-4o demos) -> 500 entries sampled from the released retrieval database, which
    are GPT-4o <think>/<response> outputs for the same kind of dialogue state (history -> think + response);
  * the RL prompts (their data/esconv-thinking/train.parquet) -> rebuilt with the layout of the released
    examples/data_preprocess/esconv.py from the ESConv train split of this project (identical to the official 910-dialogue train
    split; parents that appear in the evaluation panel are excluded).

Leakage control: the released pairs / database contain verbatim client utterances from ESConv dialogues that are dev or test parents in
this project (some are evaluation-panel parents). Every released item whose text contains an utterance of a dev/test parent, or shares a
12-gram with an evaluation-panel user turn, is dropped from the reward-model pairs, the SFT demos and the retrieval corpus.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[2]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
from baselines.external.esc_repro_common import (ESCRAG_REPO, E, POLICY, THERAPIST_SYSTEM, PANEL, panel_excluded_parents, read_jsonl, sha256, transcript,
                              write_jsonl)


def turn_index(row: dict) -> int:
    return int(row["id"].rsplit(":", 1)[1])


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower().strip())


def ngrams(text: str, n: int = 12) -> set[str]:
    words = text.lower().split()
    return {" ".join(words[i:i + n]) for i in range(max(0, len(words) - n + 1))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=E / "escrag_repro/data")
    parser.add_argument("--sft-demos", type=int, default=500)
    parser.add_argument("--grpo-prompts", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260930)
    args = parser.parse_args()
    rng = random.Random(args.seed)

    excluded = panel_excluded_parents()
    panel_grams = set()
    for row in read_jsonl(PANEL):
        for message in row["messages"]:
            if message["role"] == "user" and len(message["content"].split()) >= 12:
                panel_grams |= ngrams(message["content"])
    utterance_owner: dict[str, set[tuple[str, str]]] = {}
    for split in ("train", "dev", "test"):
        for row in read_jsonl(POLICY / f"esconv_{split}.jsonl"):
            for message in row["messages"][1:]:
                if message["role"] == "user" and len(message["content"].split()) >= 8:
                    utterance_owner.setdefault(norm(message["content"]), set()).add((row["parent_id"], split))

    def leaks(text: str) -> bool:
        flat = norm(text)
        for utterance, owners in utterance_owner.items():
            if utterance in flat and any(split != "train" or parent in excluded for parent, split in owners):
                return True
        return bool(ngrams(text) & panel_grams)

    # ---- released database -> filtered retrieval corpus and SFT demos
    entries = json.loads((ESCRAG_REPO / "verl/RAG_Source/rag_index/entries.json").read_text(encoding="utf-8"))
    embeddings = np.load(ESCRAG_REPO / "verl/RAG_Source/rag_index/embeddings.npy")
    assert len(entries) == 3659 == len(embeddings), (len(entries), len(embeddings))
    keep_idx = [i for i, e in enumerate(entries) if not leaks(e["history"] + "\n" + e["response"])]
    kept = [entries[i] for i in keep_idx]
    (args.out / "rag_index").mkdir(parents=True, exist_ok=True)
    (args.out / "rag_index/entries.json").write_text(json.dumps(kept, ensure_ascii=False), encoding="utf-8")
    np.save(args.out / "rag_index/embeddings.npy", embeddings[keep_idx])
    usable = [e for e in kept if e["history"].strip() and "<think>" in e["response"] and "</response>" in e["response"]]
    demos = rng.sample(usable, args.sft_demos)
    sft_rows = [{"messages": [{"role": "system", "content": THERAPIST_SYSTEM},
                              {"role": "user", "content": e["history"].rstrip("\n") + "\n"},
                              {"role": "assistant", "content": e["response"].strip()}]} for e in demos]
    write_jsonl(args.out / "sft_demos.jsonl", sft_rows)

    # ---- RL prompts from the ESConv train split, layout of esconv.py (middle counselor turns only)
    by_parent: dict[str, list[dict]] = defaultdict(list)
    for row in read_jsonl(POLICY / "esconv_train.jsonl"):
        by_parent[row["parent_id"]].append(row)
    overlap = excluded & set(by_parent)
    pool = []
    for parent, rows in by_parent.items():
        if parent in excluded:
            continue
        rows.sort(key=turn_index)
        for row in rows[1:-1]:  # esconv.py skips the first and the last counselor turn of every dialogue
            turns = row["messages"][1:-1]
            if not turns or turns[-1]["role"] != "user":
                continue
            history = transcript(turns)
            meta = row["metadata"]
            pool.append({
                "id": row["id"], "parent_id": parent, "history": history,
                "emotion_type": meta.get("emotion_type", ""), "problem_type": meta.get("problem_type", ""),
                "situation": meta.get("situation", ""),
                "messages": [{"role": "system", "content": THERAPIST_SYSTEM}, {"role": "user", "content": history}],
            })
    rng.shuffle(pool)
    grpo = pool[:args.grpo_prompts]
    write_jsonl(args.out / "grpo_prompts.jsonl", grpo)
    write_jsonl(args.out / "esconv_pool.jsonl", pool)  # all contexts, used by the RLFF-ESC simulation as well

    # ---- reward-model pairs (chosen / rejected share the same user turn)
    pair_counts = {}
    for split in ("train", "valid"):
        pairs = read_jsonl(ESCRAG_REPO / f"ESC-Preference/preference_pairs_{split}.jsonl")
        kept_pairs = [p for p in pairs if not leaks(p["chosen"][0]["content"])]
        write_jsonl(args.out / f"preference_pairs_{split}.jsonl", kept_pairs)
        pair_counts[split] = {"released": len(pairs), "kept": len(kept_pairs)}
    rl_hits = sum(1 for row in grpo if ngrams(row["history"]) & panel_grams)

    manifest = {
        "seed": args.seed, "sft_demos": len(sft_rows), "grpo_prompts": len(grpo), "esconv_pool_contexts": len(pool),
        "esconv_train_parents": len(by_parent), "panel_parents_in_train_split": sorted(overlap),
        "parents_used_in_grpo_prompts": len({row["parent_id"] for row in grpo}),
        "grpo_prompts_with_12gram_overlap_with_panel_user_turns": rl_hits,
        "retrieval_corpus": {"released": len(entries), "kept_after_leak_filter": len(kept)},
        "preference_pairs": pair_counts,
        "sources": {
            "database.json_sha256": sha256(ESCRAG_REPO / "verl/RAG_Source/database.json"),
            "embeddings.npy_sha256": sha256(ESCRAG_REPO / "verl/RAG_Source/rag_index/embeddings.npy"),
            "repo_commit": "76dae5cebd56075a97877ec95f04f2b76aac01ea",
        },
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
