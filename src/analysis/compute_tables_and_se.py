"""Compute publication tables (across-sample mean +/- SE) and 14-metric competitive rankings.

Covers:
  - Across-sample standard error: SE = sample_sd / sqrt(n)
  - Core 14-metric competitive ranking (unrounded ties share rank), best-metric count,
    and pairwise MCARE win/tie/loss summary (Table V)
  - Consolidated Table I-VIII JSON & Markdown summary generator from evaluation outputs
"""

from __future__ import annotations

import argparse
import json
import math
import statistics as st
import sys
from pathlib import Path
from typing import Any

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from common.config import get_paths
from common.io_utils import dump_json, load_json

CORE14_DIRECTIONS: dict[str, str] = {
    "M1_NFR": "max",
    "esconv_main_empathy": "max",
    "esconv_main_esc_rank_overall": "max",
    "esconv_main_armorm_helpfulness": "max",
    "ed_main_empathy": "max",
    "ed_main_esc_rank_overall": "max",
    "ed_main_armorm_helpfulness": "max",
    "M2_D1_both_success": "max",
    "M2_D1_identity_strict": "min",
    "M2_D1_identity_including_hedges": "min",
    "M2_D1L_both_success": "max",
    "M2_D1L_identity_strict": "min",
    "M2_D1L_identity_including_hedges": "min",
    "controlled_NVR": "min",
}


def mean_and_se(values: list[float | int | None]) -> tuple[float | None, float | None, int]:
    clean = [float(x) for x in values if x is not None]
    n = len(clean)
    if n == 0:
        return None, None, 0
    mean = st.mean(clean)
    se = st.stdev(clean) / math.sqrt(n) if n > 1 else 0.0
    return mean, se, n


def competitive_ranks(values_by_model: dict[str, float], direction: str, tol: float = 1e-12) -> dict[str, int]:
    """Assign standard competition ranks (1, 1, 3, ...) on unrounded values."""
    reverse = direction == "max"
    ordered = sorted(values_by_model.items(), key=lambda kv: kv[1], reverse=reverse)
    ranks: dict[str, int] = {}
    prev_val: float | None = None
    prev_rank = 0
    for idx, (model, val) in enumerate(ordered, start=1):
        if prev_val is not None and abs(val - prev_val) <= tol:
            ranks[model] = prev_rank
        else:
            ranks[model] = idx
            prev_rank = idx
            prev_val = val
    return ranks


def compute_rank_summary(
    table_values: dict[str, dict[str, float]],
    metric_directions: dict[str, str] | None = None,
    ours_name: str = "MCARE",
    tol: float = 1e-12,
) -> dict[str, dict[str, Any]]:
    directions = metric_directions or CORE14_DIRECTIONS
    models = list(table_values.keys())
    ranks_accum: dict[str, list[int]] = {m: [] for m in models}
    best_counts: dict[str, int] = {m: 0 for m in models}
    wtl: dict[str, list[int]] = {m: [0, 0, 0] for m in models}  # win, tie, loss (MCARE vs baseline)

    for metric, direction in directions.items():
        metric_vals = {m: float(table_values[m][metric]) for m in models}
        ranks = competitive_ranks(metric_vals, direction, tol=tol)
        for m, rk in ranks.items():
            ranks_accum[m].append(rk)
            if rk == 1:
                best_counts[m] += 1
        if ours_name in metric_vals:
            ov = metric_vals[ours_name]
            for m in models:
                if m == ours_name:
                    continue
                bv = metric_vals[m]
                if abs(ov - bv) <= tol:
                    wtl[m][1] += 1
                elif (ov > bv and direction == "max") or (ov < bv and direction == "min"):
                    wtl[m][0] += 1
                else:
                    wtl[m][2] += 1

    out: dict[str, dict[str, Any]] = {}
    for m in models:
        out[m] = {
            "best_metrics": best_counts[m],
            "mean_rank": st.mean(ranks_accum[m]),
            "mcare_win_tie_loss": None if m == ours_name else {"win": wtl[m][0], "tie": wtl[m][1], "loss": wtl[m][2]},
        }
    return out


def summarize_observations(observations: dict[str, dict[str, dict[str, tuple[str, float | None]]]]) -> tuple[dict, dict]:
    """Compute item-level means and SEs from {model: {metric: {id: (parent_id, value)}}}."""
    means: dict[str, dict[str, float]] = {}
    stats_out: dict[str, dict[str, dict[str, float | int | None]]] = {}
    for model, metrics in observations.items():
        means[model] = {}
        stats_out[model] = {}
        for metric, items in metrics.items():
            vals = [v[1] for v in items.values() if v[1] is not None]
            mn, se, n = mean_and_se(vals)
            if mn is not None:
                means[model][metric] = mn
            stats_out[model][metric] = {"mean": mn, "se": se, "n": n}
    return means, stats_out


def main() -> None:
    cfg = get_paths()
    parser = argparse.ArgumentParser(description="Compute SEs and 14-metric competitive ranks from evaluation outputs.")
    parser.add_argument("--observations-json", type=Path, required=True, help="JSON mapping model -> metric -> id -> [parent_id, value]")
    parser.add_argument("--ours-name", default="MCARE")
    parser.add_argument("--out-dir", type=Path, default=cfg.exp_dir / "analysis")
    args = parser.parse_args()

    obs = load_json(args.observations_json)
    means, stats_out = summarize_observations(obs)
    available_metrics = [m for m in CORE14_DIRECTIONS if all(m in means[k] for k in means)]
    sub_dirs = {m: CORE14_DIRECTIONS[m] for m in available_metrics}
    ranks = compute_rank_summary(means, sub_dirs, ours_name=args.ours_name) if sub_dirs else {}

    args.out_dir.mkdir(parents=True, exist_ok=True)
    dump_json(args.out_dir / "metrics_mean_se.json", stats_out)
    dump_json(args.out_dir / "competitive_ranks.json", ranks)

    lines = ["# Core Evaluation Summary & Competitive Ranks", "", "| Model | Best Metrics | Mean Rank | MCARE W/T/L |", "|---|---:|---:|:---:|"]
    for m, r in ranks.items():
        wtl_str = "—" if r["mcare_win_tie_loss"] is None else f"{r['mcare_win_tie_loss']['win']}/{r['mcare_win_tie_loss']['tie']}/{r['mcare_win_tie_loss']['loss']}"
        lines.append(f"| {m} | {r['best_metrics']} | {r['mean_rank']:.2f} | {wtl_str} |")
    (args.out_dir / "RANK_SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote metrics_mean_se.json, competitive_ranks.json, and RANK_SUMMARY.md to {args.out_dir}")


if __name__ == "__main__":
    main()
