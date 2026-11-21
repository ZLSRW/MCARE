"""Stage 1 Dataset Compiler: Combine P_base with Five Structured Preference Sets.

Implements Eq. (17) in Section III-B4:
  D^(1) = P_base (+) 2*P_rev (+) 3*P_need (+) 3*P_warm (+) 2*P_req (+) 1*P_acc

Asserts zero parent overlap with validation/test splits and verifies that
every training sequence fits within max_length=4096.
"""

from __future__ import annotations

import argparse
import collections
import json
import random
import sys
from pathlib import Path
from typing import Any

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from common.config import get_paths
from common.io_utils import dump_json, key, load_heldout_parents, read_jsonl, sha256_file, write_jsonl

STRUCTURED_SPECS = [
    ("rev", "P_rev.jsonl", 2),
    ("need", "P_need.jsonl", 3),
    ("warm", "P_warm.jsonl", 3),
    ("req", "P_req.jsonl", 2),
    ("acc", "P_acc.jsonl", 1),
]
MAX_TOKENS = 4096
COMPLETION_RESERVE = 512


def _view_to_dpo_row(tag: str, row: dict[str, Any], view_idx: int, view: dict[str, Any]) -> dict[str, Any]:
    responses = view["responses"]
    assert len(responses) == 2 and responses[0].strip() != responses[1].strip()
    return {
        "id": f"struct:{tag}:{row['id']}:{view_idx}",
        "source": row["source"],
        "parent_id": row["parent_id"],
        "kind": tag,
        "messages": view["messages"] + [{"role": "assistant", "content": responses[0]}],
        "rejected_response": responses[1],
    }


def _check_token_length(tok: Any, row: dict[str, Any], max_tokens: int) -> bool:
    for resp in (row["messages"][-1]["content"], row["rejected_response"]):
        msgs = row["messages"][:-1] + [{"role": "assistant", "content": resp}]
        enc = tok.apply_chat_template(msgs, tokenize=True, enable_thinking=False)
        ids = enc["input_ids"] if hasattr(enc, "__getitem__") and "input_ids" in enc else enc
        if len(ids) > max_tokens:
            return False
    return True


def main() -> None:
    cfg = get_paths()
    parser = argparse.ArgumentParser(description="Compile Stage-1 dataset D^(1) from P_base and structured preference sets.")
    parser.add_argument("--base-pairs", type=Path, default=cfg.stage1_dir / "base_pool" / "stage1_base_pairs.jsonl")
    parser.add_argument("--structured-dir", type=Path, default=cfg.specialist_dir / "data" / "objectives")
    parser.add_argument("--eval-inputs", type=Path, default=cfg.eval_dir / "generation_inputs.jsonl")
    parser.add_argument("--dev-inputs", type=Path, default=cfg.exp_dir / "dev" / "generation_inputs.jsonl")
    parser.add_argument("--out-path", type=Path, default=cfg.stage1_dir / "stage1_dataset.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-length-check", action="store_true")
    args = parser.parse_args()

    forbidden = load_heldout_parents(cfg.policy_dir, (args.eval_inputs, args.dev_inputs))
    base_rows = read_jsonl(args.base_pairs)
    assert base_rows, f"Empty base pairs: {args.base_pairs}"

    tok = None
    if not args.skip_length_check:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(cfg.base_model, local_files_only=True)

    rows: list[dict[str, Any]] = []
    dropped = collections.Counter()
    counts_by_group = collections.Counter()

    for r in base_rows:
        if r["parent_id"] in forbidden:
            dropped["forbidden_parent"] += 1
            continue
        if tok is not None and not _check_token_length(tok, r, MAX_TOKENS - COMPLETION_RESERVE):
            dropped["exceeds_max_tokens"] += 1
            continue
        rows.append(r)
        counts_by_group["base"] += 1

    for tag, filename, repeat in STRUCTURED_SPECS:
        src_path = args.structured_dir / filename
        if not src_path.exists():
            continue
        for r in read_jsonl(src_path):
            if r["parent_id"] in forbidden:
                dropped[f"{tag}:forbidden_parent"] += 1
                continue
            for v_idx, view in enumerate(r.get("views", [])):
                dpo_row = _view_to_dpo_row(tag, r, v_idx, view)
                if tok is not None and not _check_token_length(tok, dpo_row, MAX_TOKENS):
                    dropped[f"{tag}:exceeds_max_tokens"] += 1
                    continue
                for rep in range(repeat):
                    rows.append({**dpo_row, "id": f"{dpo_row['id']}:r{rep}"})
                    counts_by_group[tag] += 1

    rng = random.Random(args.seed)
    rng.shuffle(rows)
    assert not ({r["parent_id"] for r in rows} & forbidden)

    write_jsonl(args.out_path, rows)
    summary = {
        "rows": len(rows),
        "updates": -(-len(rows) // 16),
        "counts_by_group": dict(counts_by_group),
        "dropped": dict(dropped),
        "sha256": sha256_file(args.out_path),
    }
    dump_json(args.out_path.with_suffix(".summary.json"), summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
