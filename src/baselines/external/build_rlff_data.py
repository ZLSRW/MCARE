"""Contexts for the RLFF-ESC reproduction (arXiv 2508.12935): ESConv train-split dialogue states.

Two disjoint sets of counselor turns (first and last counselor turn of every dialogue excluded, as in the ESCRAG-R1 preprocessing so both
reproductions draw on the same pool; parents that appear in the evaluation panel are excluded):
  sim_contexts.jsonl  contexts for the multi-agent simulation that produces the reward-model dataset D_r (paper: all ESConv train + ExTES train)
  rl_prompts.jsonl    the 1,000 GRPO prompts (budget aligned with the other reproduced RL baselines)
ExTES (paper's second dataset) is not available offline here and is not used.
"""

from __future__ import annotations

import argparse
import random
from collections import defaultdict
from pathlib import Path

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[2]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
from baselines.external.esc_repro_common import E, POLICY, THERAPIST_SYSTEM, panel_excluded_parents, read_jsonl, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=E / "rlff_repro/data")
    parser.add_argument("--sim-contexts", type=int, default=2000)
    parser.add_argument("--rl-prompts", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260931)
    args = parser.parse_args()
    excluded = panel_excluded_parents()
    by_parent = defaultdict(list)
    for row in read_jsonl(POLICY / "esconv_train.jsonl"):
        by_parent[row["parent_id"]].append(row)
    pool = []
    for parent, rows in by_parent.items():
        if parent in excluded:
            continue
        rows.sort(key=lambda r: int(r["id"].rsplit(":", 1)[1]))
        for row in rows[1:-1]:
            turns = [{"role": m["role"], "content": m["content"].strip()} for m in row["messages"][1:-1]]
            if turns and turns[-1]["role"] == "user":
                meta = row["metadata"]
                pool.append({"id": row["id"], "parent_id": parent, "turns": turns, "emotion_type": meta.get("emotion_type", ""),
                             "problem_type": meta.get("problem_type", ""), "situation": meta.get("situation", "")})
    rng = random.Random(args.seed)
    rng.shuffle(pool)
    sim, rl = pool[:args.sim_contexts], pool[args.sim_contexts:args.sim_contexts + args.rl_prompts]
    write_jsonl(args.out / "sim_contexts.jsonl", sim)
    def history(turns):  # 'Patient:/Therapist:' transcript, the reward-model input layout (scripts/rlff_rm.py)
        return "\n".join(f"{'Patient' if t['role'] == 'user' else 'Therapist'}: {t['content'].strip()}" for t in turns)

    def chat(turns):  # the chat encoder pairs user/assistant turns: a session that opens with the counselor gets a neutral seeker opener in front (history keeps the original)
        return ([{"role": "user", "content": "Hello."}] if turns[0]["role"] == "assistant" else []) + turns

    rl_rows = [{"messages": [{"role": "system", "content": THERAPIST_SYSTEM}] + chat(c["turns"]), "history": history(c["turns"]), "id": c["id"],
                "parent_id": c["parent_id"]} for c in rl]
    write_jsonl(args.out / "rl_prompts.jsonl", rl_rows)
    print({"pool": len(pool), "sim": len(sim), "rl": len(rl), "sim_parents": len({c['parent_id'] for c in sim}), "rl_parents": len({c['parent_id'] for c in rl})})


if __name__ == "__main__":
    main()
