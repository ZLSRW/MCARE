"""Stage 1 Base Preference-Pair Builder (P_base, Section III-B2 Eqs. 8-11 & Section IV-C Eq. 31).

Builds compliance-gated, length-matched preference pairs from offline K-candidate
samples of the frozen base policy (pi_0) using offline annotation and reward signals:
  - Semantic violation flags c_hat_i and warmth w_hat_i from Qwen3.8-27B review
  - Within-group percentile general quality Q_i (training quality critic)
  - Within-group percentile support utility S_i (training-split support RM)
  - Within-group percentile training reference similarity H_i (renormalized when absent)
  - Composite score Eq. (31): s_i = 0.50 Q_i + 0.25 W_i + 0.10 S_i + 0.15 H_i

Positive threshold w_0 = 3; negative selection matches positive length within
delta = 0.25 and prioritizes the highest-scoring violating candidate.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
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

WEIGHTS = {"quality": 0.50, "warmth": 0.25, "support": 0.10, "reference": 0.15}
DEFAULT_REPEATS = {"identity": 6, "request": 3, "controlled": 3, "natural": 1}


def percentile(values: list[float]) -> list[float]:
    n = len(values)
    assert n >= 2 and all(math.isfinite(v) for v in values)
    denom = float(n - 1)
    return [
        (sum(y < x - 1e-10 for y in values) + 0.5 * (sum(abs(y - x) <= 1e-10 for y in values) - 1)) / denom
        for x in values
    ]


def score_candidate_pool(
    cs: list[dict[str, Any]],
    checks: list[dict[str, Any]],
    quality_raw: list[float],
    support_raw: list[float],
    ref_raw: list[float] | None = None,
) -> tuple[list[float], bool]:
    q_pct = percentile(quality_raw)
    w_norm = [(float(c["warmth"]) - 1.0) / 4.0 for c in checks]
    s_pct = percentile(support_raw)
    has_ref = ref_raw is not None and len(ref_raw) == len(cs)
    h_pct = percentile(ref_raw) if has_ref else [0.0] * len(cs)

    denom = 1.0 if has_ref else (1.0 - WEIGHTS["reference"])
    scores = [
        (
            WEIGHTS["quality"] * q_pct[i]
            + WEIGHTS["warmth"] * w_norm[i]
            + WEIGHTS["support"] * s_pct[i]
            + WEIGHTS["reference"] * h_pct[i]
        )
        / denom
        for i in range(len(cs))
    ]
    return scores, has_ref


def build_base_pairs(
    candidates_path: Path,
    annotations_dir: Path,
    quality_path: Path,
    support_path: Path,
    out_dir: Path,
    reference_scores_path: Path | None = None,
    warmth_min: int = 3,
    length_window: float = 0.25,
    forbidden_parents: set[str] | None = None,
) -> dict[str, Any]:
    rows = read_jsonl(candidates_path)
    quality_map = {r["id"]: r for r in read_jsonl(quality_path)}
    support_map = {r["id"]: r for r in read_jsonl(support_path)}
    ref_map = (
        {r["id"]: r for r in read_jsonl(reference_scores_path)}
        if reference_scores_path and reference_scores_path.exists()
        else {}
    )
    forbidden = forbidden_parents or set()

    pairs: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    proto_hash = key([PROMPT, SCHEMA])

    for row in rows:
        gid = row["id"]
        if str(row["parent_id"]) in forbidden:
            exclusions.append({"id": gid, "reason": "heldout_parent_excluded"})
            continue
        ann_path = annotations_dir / f"{key(gid)}.json"
        if not ann_path.exists():
            exclusions.append({"id": gid, "reason": "annotation_missing"})
            continue
        ann = json.loads(ann_path.read_text(encoding="utf-8"))
        assert ann["protocol_hash"] == proto_hash
        cs = row["candidates"]
        checks = ann["checks"]
        assert len(cs) == len(checks)
        costs = [sum(c[b] for b in BOOLS) for c in checks]

        if gid not in quality_map or gid not in support_map:
            exclusions.append({"id": gid, "reason": "training_signal_missing"})
            continue

        q_raw = [float(x["score"]) for x in quality_map[gid]["scores"]]
        s_raw = [float(x["p_E"]) for x in support_map[gid]["scores"]]
        has_ref = bool((row.get("reference") or "").strip()) and gid in ref_map
        r_raw = [float(x["score"]) for x in ref_map[gid]["scores"]] if has_ref else None

        values, _ = score_candidate_pool(cs, checks, q_raw, s_raw, r_raw)
        valid = [i for i in range(len(cs)) if cs[i]["text"].strip() and not cs[i].get("hit_cap", False)]
        good = [i for i in valid if costs[i] == 0 and checks[i]["warmth"] >= warmth_min]
        if not good:
            exclusions.append({"id": gid, "reason": "no_compliant_warm_positive"})
            continue

        chosen = max(good, key=lambda i: (values[i], -i))
        length = len(cs[chosen]["text"].split())
        near = [
            i
            for i in valid
            if i != chosen
            and cs[i]["text"] != cs[chosen]["text"]
            and (1.0 - length_window) * length <= len(cs[i]["text"].split()) <= (1.0 + length_window) * length
        ]
        bad = [i for i in near if costs[i] > 0]
        if not near:
            exclusions.append({"id": gid, "reason": "no_length_matched_negative"})
            continue
        rejected = max(bad, key=lambda i: (values[i], -i)) if bad else min(near, key=lambda i: (values[i], i))
        if not bad and values[chosen] <= values[rejected] + 1e-10:
            exclusions.append({"id": gid, "reason": "no_strict_score_gap"})
            continue

        category = row.get("repeat_category") or ("identity" if row.get("identity") else "request" if row.get("request") else "controlled" if row.get("kind", "natural") != "natural" else "natural")
        copies = int(row.get("repeat") or DEFAULT_REPEATS.get(category, 1))
        base_meta = {k: row[k] for k in ("id", "source", "parent_id") if k in row}
        base_meta["kind"] = row.get("kind", "natural")
        for rep in range(copies):
            ident = f"{gid}:repeat{rep}"
            pairs.append(
                {
                    **base_meta,
                    "id": ident,
                    "source_id": gid,
                    "messages": row["messages"] + [{"role": "assistant", "content": cs[chosen]["text"]}],
                    "rejected_response": cs[rejected]["text"],
                }
            )
            audits.append(
                {
                    "id": ident,
                    "source_id": gid,
                    "chosen_index": chosen,
                    "rejected_index": rejected,
                    "chosen_cost": costs[chosen],
                    "rejected_cost": costs[rejected],
                    "chosen_words": length,
                    "rejected_words": len(cs[rejected]["text"].split()),
                }
            )

    out_dir.mkdir(parents=True, exist_ok=True)
    pairs_path = out_dir / "stage1_base_pairs.jsonl"
    write_jsonl(pairs_path, pairs)
    write_jsonl(out_dir / "stage1_base_audit.jsonl", audits)
    write_jsonl(out_dir / "stage1_base_exclusions.jsonl", exclusions)

    summary = {
        "pairs": len(pairs),
        "unique_contexts": len({r["source_id"] for r in pairs}),
        "sha256": sha256_file(pairs_path),
        "kinds": dict(collections.Counter(r["kind"] for r in pairs)),
        "violating_rejected_fraction": statistics.mean(a["rejected_cost"] > 0 for a in audits) if audits else 0.0,
        "mean_chosen_words": statistics.mean(a["chosen_words"] for a in audits) if audits else 0.0,
        "mean_rejected_words": statistics.mean(a["rejected_words"] for a in audits) if audits else 0.0,
        "exclusions": len(exclusions),
    }
    dump_json(out_dir / "stage1_base_summary.json", summary)
    return summary


def main() -> None:
    cfg = get_paths()
    default_dir = cfg.stage1_dir / "base_pool"
    parser = argparse.ArgumentParser(description="Build Stage-1 base preference pairs (P_base).")
    parser.add_argument("--candidates", type=Path, default=default_dir / "train_candidates.jsonl")
    parser.add_argument("--annotations-dir", type=Path, default=default_dir / "annotations")
    parser.add_argument("--quality-scores", type=Path, default=default_dir / "quality_scores.jsonl")
    parser.add_argument("--support-scores", type=Path, default=default_dir / "support_scores.jsonl")
    parser.add_argument("--reference-scores", type=Path, default=default_dir / "reference_scores.jsonl")
    parser.add_argument("--out-dir", type=Path, default=default_dir)
    args = parser.parse_args()

    forbidden = load_heldout_parents(
        cfg.policy_dir,
        (cfg.eval_dir / "generation_inputs.jsonl", cfg.exp_dir / "dev" / "generation_inputs.jsonl"),
    )
    summary = build_base_pairs(
        args.candidates,
        args.annotations_dir,
        args.quality_scores,
        args.support_scores,
        args.out_dir,
        args.reference_scores,
        forbidden_parents=forbidden,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
