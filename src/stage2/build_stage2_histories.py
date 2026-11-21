"""Select natural and controlled/request Stage-2 histories from the training pool.

Reproduces the 3,734-history Stage-2 pool (2,714 natural support histories and
1,020 controlled/explicit-request contexts) while asserting zero parent overlap
with validation and sealed test splits.
"""

from __future__ import annotations

import argparse
import collections
import sys
from pathlib import Path
from typing import Any

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from common.config import get_paths
from common.io_utils import dump_json, key, load_heldout_parents, read_jsonl, sha256_file, write_jsonl


def forbidden_parents(eval_inputs: Path, dev_inputs: Path) -> set[str]:
    return load_heldout_parents(get_paths().policy_dir, (eval_inputs, dev_inputs))


def main() -> None:
    cfg = get_paths()
    parser = argparse.ArgumentParser(description="Partition Stage-2 training histories into natural and controlled sets.")
    parser.add_argument("--pool", type=Path, default=cfg.stage1_dir / "base_pool" / "train_candidates.jsonl")
    parser.add_argument("--eval-inputs", type=Path, default=cfg.eval_dir / "generation_inputs.jsonl")
    parser.add_argument("--dev-inputs", type=Path, default=cfg.exp_dir / "dev" / "generation_inputs.jsonl")
    parser.add_argument("--out-dir", type=Path, default=cfg.stage2_dir / "data")
    parser.add_argument("--max-natural", type=int, default=2714)
    args = parser.parse_args()

    forbidden = forbidden_parents(args.eval_inputs, args.dev_inputs)
    pool = read_jsonl(args.pool)
    natural: list[dict[str, Any]] = []
    controlled: list[dict[str, Any]] = []

    for row in pool:
        assert row["parent_id"] not in forbidden, f"Parent leakage detected: {row['parent_id']}"
        is_conditioned = bool(row.get("instruction") or row.get("feedback") or row.get("kind", "natural") != "natural")
        entry = {
            "id": row["id"],
            "source": row["source"],
            "parent_id": row["parent_id"],
            "kind": row.get("kind", "natural"),
            "family": row.get("family"),
            "messages": row["messages"],
            "instruction": row.get("instruction"),
            "feedback": row.get("feedback"),
            "reference": row.get("reference") or "",
            "repeat": int(row.get("repeat") or (1 if not is_conditioned else 3)),
            "identity": bool(row.get("identity")),
            "request": bool(row.get("request")),
        }
        if is_conditioned or entry["repeat"] > 1:
            controlled.append(entry)
        else:
            natural.append(entry)

    natural.sort(key=lambda r: key(["stage2_natural", r["id"]]))
    if args.max_natural > 0:
        natural = natural[: args.max_natural]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    nat_path = args.out_dir / "stage2_natural_histories.jsonl"
    ctl_path = args.out_dir / "stage2_controlled_histories.jsonl"
    write_jsonl(nat_path, natural)
    write_jsonl(ctl_path, controlled)

    summary = {
        "natural_histories": len(natural),
        "natural_by_source": dict(collections.Counter(r["source"] for r in natural)),
        "controlled_histories": len(controlled),
        "controlled_by_kind": dict(collections.Counter(r["kind"] for r in controlled)),
        "natural_sha256": sha256_file(nat_path),
        "controlled_sha256": sha256_file(ctl_path),
        "overlap_with_dev_test": 0,
    }
    dump_json(args.out_dir / "stage2_histories_summary.json", summary)
    print(f"Stage-2 histories built: natural={len(natural)}, controlled={len(controlled)}")


if __name__ == "__main__":
    main()
