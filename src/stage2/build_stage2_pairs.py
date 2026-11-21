"""Build Stage-2 MCARE preference pairs (Section III-C Eqs. 25-27, Section IV-C Eq. 32) and Table VI ablations.

Inputs and offline signals:
  - Qwen3.8-27B five-flag compliance gate (c_hat == 0, warmth >= w_0 = 3)
  - e_i: Qwen3.8-27B fine-grained expected empathy score
  - h_i: Qwen3.8-27B / HelpSteer2 general helpfulness score
  - o_i: Support-utility reward model probability

Scoring (Eq. 32):
  - Natural histories:    s_i^E = 0.50 P_x(e_i) + 0.25 P_x(h_i) + 0.25 P_x(o_i)
  - Controlled histories: s_i^E = 0.50 [P_x(e_i) + P_x(h_i)]

Modes:
  --mode mcare      Build Stage-2 MCARE training pairs (stage2_pairs.jsonl)
  --mode ablations  Build Table VI self-sampled (gated & no-natural-gate) ablations
"""

from __future__ import annotations

import argparse
import collections
import json
import statistics
import sys
from pathlib import Path
from typing import Any

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from common.config import get_paths
from common.io_utils import dump_json, key, load_heldout_parents, read_jsonl, sha256_file, write_jsonl
from stage2.annotate_candidates import BOOLS, PROMPT, SCHEMA
from stage2.pair_selection import percentile, pick, values_for

UNIT_WEIGHT = {"quality": 1.0, "warmth": 0.0, "support": 0.0, "reference": 0.0}


def _forbidden_parents() -> set[str]:
    cfg = get_paths()
    return load_heldout_parents(
        cfg.policy_dir,
        (cfg.eval_dir / "generation_inputs.jsonl", cfg.exp_dir / "dev" / "generation_inputs.jsonl"),
    )


def _load_candidate_map(path: Path, score_field: str) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    for r in read_jsonl(path):
        out[r["candidate_id"]] = r.get(score_field)
    return out


def _load_support_map(path: Path) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    if path.is_file():
        for r in read_jsonl(path):
            if "candidate_id" in r:
                out[r["candidate_id"]] = r.get("p_E", r.get("score"))
            else:
                for s in r.get("scores", []):
                    out[s["candidate_id"]] = s.get("p_E", s.get("score"))
    elif path.is_dir():
        for sfile in sorted(path.glob("shard*.jsonl")):
            for r in read_jsonl(sfile):
                out[r["candidate_id"]] = r.get("p_E", r.get("score"))
    return out


def _build_split_pairs(
    rows: list[dict[str, Any]],
    empathy_map: dict[str, float | None],
    helpfulness_map: dict[str, float | None],
    support_map: dict[str, float | None] | None,
    annotations_dir: Path,
    weights: dict[str, float],
    apply_gate: bool = True,
    warmth_min: int = 3,
    length_window: float = 0.25,
    kind_tag: str = "stage2",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    proto_hash = key([PROMPT, SCHEMA])
    out: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    exclusions = collections.Counter()

    for row in rows:
        cs = row["candidates"]
        n_cand = len(cs)
        ann_path = annotations_dir / f"{key(row['id'])}.json"
        annotated = ann_path.exists()
        if annotated:
            ann = json.loads(ann_path.read_text(encoding="utf-8"))
            assert ann["protocol_hash"] == proto_hash
            checks = ann["checks"]
            costs = [sum(c[b] for b in BOOLS) for c in checks]
        else:
            checks = [{"warmth": warmth_min} for _ in cs]
            costs = [0] * n_cand

        raw_e = [empathy_map.get(c["candidate_id"]) for c in cs]
        raw_h = [helpfulness_map.get(c["candidate_id"]) for c in cs]
        raw_o = [support_map.get(c["candidate_id"]) for c in cs] if support_map is not None else [0.0] * n_cand

        pct: dict[str, list[float] | None] = {
            "e": percentile([float(x) for x in raw_e]) if all(x is not None for x in raw_e) else None,
            "h": percentile([float(x) for x in raw_h]) if all(x is not None for x in raw_h) else None,
            "o": percentile([float(x) for x in raw_o]) if ("o" in weights and all(x is not None for x in raw_o)) else None,
        }
        if any(pct[k] is None for k in weights):
            exclusions["training_scorer_unparsed"] += 1
            continue

        valid = [i for i in range(n_cand) if cs[i]["text"].strip() and not cs[i].get("hit_cap", False)]
        effective_costs = costs if apply_gate else [0] * n_cand
        if apply_gate:
            if not annotated:
                exclusions["annotation_unknown"] += 1
                continue
            good = [i for i in valid if costs[i] == 0 and checks[i]["warmth"] >= warmth_min]
        else:
            good = valid

        if not good:
            exclusions["no_compliant_warm_positive"] += 1
            continue

        reward = [sum(w * pct[k][i] for k, w in weights.items()) for i in range(n_cand)]  # type: ignore[index]
        h = {
            "cs": cs,
            "costs": effective_costs,
            "has_reference": False,
            "warmth": [0.0] * n_cand,
            "future": [0.0] * n_cand,
            "reference": [0.0] * n_cand,
            "quality": reward,
            "valid": valid,
        }
        values = values_for(h, UNIT_WEIGHT)
        chosen = max(good, key=lambda i: (values[i], -i))
        rejected, reason = pick(h, values, length_window, chosen)
        if rejected is None:
            exclusions[reason] += 1
            continue

        repeat = int(row.get("repeat", 1))
        for k in range(repeat):
            out.append(
                {
                    "id": f"{kind_tag}:{row['id']}:repeat{k}",
                    "source": row["source"],
                    "parent_id": row["parent_id"],
                    "kind": kind_tag,
                    "source_id": row["id"],
                    "messages": row["messages"] + [{"role": "assistant", "content": cs[chosen]["text"]}],
                    "rejected_response": cs[rejected]["text"],
                }
            )
        audits.append(
            {
                "chosen_words": len(cs[chosen]["text"].split()),
                "rejected_words": len(cs[rejected]["text"].split()),
                "rejected_cost": costs[rejected],
                "d_empathy": float(raw_e[chosen]) - float(raw_e[rejected]),  # type: ignore[arg-type]
                "d_helpfulness": float(raw_h[chosen]) - float(raw_h[rejected]),  # type: ignore[arg-type]
            }
        )

    summary = {
        "rows": len(out),
        "histories_used": len(audits),
        "exclusions": dict(exclusions),
        "violating_rejected_fraction": statistics.mean(a["rejected_cost"] > 0 for a in audits) if audits else 0.0,
        "mean_d_empathy": statistics.mean(a["d_empathy"] for a in audits) if audits else 0.0,
        "mean_d_helpfulness": statistics.mean(a["d_helpfulness"] for a in audits) if audits else 0.0,
    }
    return out, summary


def main() -> None:
    cfg = get_paths()
    parser = argparse.ArgumentParser(description="Build Stage-2 MCARE preference pairs and Table VI ablations.")
    parser.add_argument("--mode", choices=["mcare", "ablations"], default="mcare")
    parser.add_argument("--data-dir", type=Path, default=cfg.stage2_dir / "data")
    parser.add_argument("--annotations-dir", type=Path, default=cfg.stage2_dir / "annotations")
    parser.add_argument("--out-path", type=Path, default=None)
    args = parser.parse_args()

    forbidden = _forbidden_parents()
    if args.mode == "mcare":
        nat_rows = read_jsonl(args.data_dir / "stage2_natural_candidates.jsonl")
        ctl_rows = read_jsonl(args.data_dir / "stage2_controlled_candidates.jsonl")
        emp = _load_candidate_map(args.data_dir / "train_empathy_scores.jsonl", "empathy")
        hlp = _load_candidate_map(args.data_dir / "train_helpfulness_scores.jsonl", "helpfulness")
        sup = _load_support_map(args.data_dir / "train_support_scores.jsonl")

        nat_pairs, s_nat = _build_split_pairs(
            nat_rows, emp, hlp, sup, args.annotations_dir, {"e": 0.50, "h": 0.25, "o": 0.25}, apply_gate=True, kind_tag="stage2_nat"
        )
        ctl_pairs, s_ctl = _build_split_pairs(
            ctl_rows, emp, hlp, None, args.annotations_dir, {"e": 0.50, "h": 0.50}, apply_gate=True, kind_tag="stage2_ctl"
        )
        combined = sorted(nat_pairs + ctl_pairs, key=lambda r: key(["stage2", r["id"]]))
        assert not ({r["parent_id"] for r in combined} & forbidden)

        out_path = args.out_path or (args.data_dir / "stage2_pairs.jsonl")
        write_jsonl(out_path, combined)
        dump_json(
            out_path.with_suffix(".summary.json"),
            {
                "rows": len(combined),
                "updates": -(-len(combined) // 16),
                "sha256": sha256_file(out_path),
                "signals": "training-only Qwen3.8-27B (gate, warmth, empathy, helpfulness) + training-split support RM",
                "natural": s_nat,
                "controlled": s_ctl,
            },
        )
        print(f"Wrote {len(combined)} Stage-2 MCARE pairs to {out_path}")
        return

    # Table VI ablations: self-sampled candidates from pi_1, with and without natural-history gate
    self_rows = read_jsonl(args.data_dir / "stage2_self_sampled_candidates.jsonl")
    emp = _load_candidate_map(args.data_dir / "self_empathy_scores.jsonl", "empathy")
    hlp = _load_candidate_map(args.data_dir / "self_helpfulness_scores.jsonl", "helpfulness")
    sup = _load_support_map(args.data_dir / "self_support_scores.jsonl")
    nat_self = [r for r in self_rows if r.get("kind", "natural") == "natural" and int(r.get("repeat", 1)) == 1]
    ctl_self = [r for r in self_rows if r not in nat_self]

    for name, nat_gate, nat_weights in [
        ("ablation_self_sampled", True, {"e": 0.50, "h": 0.25, "o": 0.25}),
        ("ablation_self_sampled_nogate", False, {"e": 1.0 / 3.0, "h": 1.0 / 3.0, "o": 1.0 / 3.0}),
    ]:
        p_nat, sn = _build_split_pairs(
            nat_self, emp, hlp, sup, args.annotations_dir, nat_weights, apply_gate=nat_gate, kind_tag=name
        )
        p_ctl, sc = _build_split_pairs(
            ctl_self, emp, hlp, None, args.annotations_dir, {"e": 0.50, "h": 0.50}, apply_gate=True, kind_tag=name
        )
        combined = sorted(p_nat + p_ctl, key=lambda r: key([name, r["id"]]))
        assert not ({r["parent_id"] for r in combined} & forbidden)
        out_p = args.data_dir / f"{name}.jsonl"
        write_jsonl(out_p, combined)
        dump_json(out_p.with_suffix(".summary.json"), {"rows": len(combined), "natural": sn, "controlled": sc})
        print(f"Wrote {len(combined)} {name} pairs to {out_p}")


if __name__ == "__main__":
    main()
