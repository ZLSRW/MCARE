"""Generate vector PDF figures for the MCARE IEEE TAFFC manuscript.

Reads summarized metric JSONs produced by evaluation/analysis scripts and renders:
  - paired_core.pdf: Paired parent-cluster bootstrap distributions vs. MCARE (Fig. 3)
  - empathy_distribution.pdf: Empirical CDFs of ESConv & ED empathy scores (Fig. 4)
  - core_tradeoff.pdf: NFR (%) vs. ESConv empathy with +/- 1 SE error bars (Fig. 5)
  - m2_position.pdf: D1 vs. D1L retention & strict identity attribution (+/- 1 SE) (Fig. 6)
  - m2_joint_empathy.pdf: D1L joint outcomes & empathy D0 -> D1L (+/- 1 SE) (Fig. 7)
  - stage_tradeoff.pdf: Attribution-empathy plane across Stage 1/2/ablation policies (Fig. 8)
  - fg_probes.pdf: Probe F adaptation rates & Probe G updated-turn empathy (+/- 1 SE) (Fig. 9)
"""

from __future__ import annotations

import argparse
import json
import math
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from analysis.compute_tables_and_se import mean_and_se
from common.io_utils import load_json

ORANGE, BLUE, GRAY, GREEN, INK = "#B5562A", "#285B7D", "#8A949C", "#3E7A55", "#2B2B2B"


def setup_style(font_dir: Path | None = None) -> None:
    if font_dir and font_dir.exists():
        for p in font_dir.glob("*.TTF"):
            font_manager.fontManager.addfont(str(p))
        for p in font_dir.glob("*.ttf"):
            font_manager.fontManager.addfont(str(p))
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif", "Liberation Serif"],
            "mathtext.fontset": "stix",
            "font.size": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.3,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 0.6,
        }
    )


def style_axis(ax: plt.Axes) -> None:
    ax.spines[["right", "top"]].set_visible(False)
    ax.tick_params(length=2.5, width=0.6)
    ax.grid(color="#E3E7EA", lw=0.5)
    ax.set_axisbelow(True)


def save_fig(fig: plt.Figure, out_dir: Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{name}.pdf", bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def plot_core_tradeoff(models: list[str], values: dict, se_map: dict, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(3.45, 2.3))
    offsets = {
        "MCARE": (7, 5),
        "Base": (-30, 10),
        "RLFF": (-16, -15),
        "RLVER": (-44, 4),
        "Vanilla-SFT": (16, -1),
        "Quality-DPO": (-54, -10),
        "Joint-DPO": (-54, -2),
        "ESCRAG": (15, -1),
        "PEER-RUR": (-50, -3),
    }
    for n in models:
        x, y = 100.0 * values[n]["M1_NFR"], values[n]["esconv_main_empathy"]
        xse, yse = 100.0 * se_map[n]["M1_NFR"], se_map[n]["esconv_main_empathy"]
        ours, base = n == "MCARE", n == "Base"
        col = ORANGE if ours else BLUE if base else GRAY
        ax.errorbar(
            x,
            y,
            xerr=xse,
            yerr=yse,
            fmt="none",
            ecolor=col,
            elinewidth=0.75 if ours else 0.55,
            capsize=1.8 if ours else 1.3,
            capthick=0.65 if ours else 0.5,
            alpha=0.85 if ours else 0.55,
            zorder=3,
        )
        ax.scatter(
            x,
            y,
            marker="*" if ours else "D" if base else "o",
            s=110 if ours else 26 if base else 22,
            c=col,
            ec="white",
            lw=0.45,
            zorder=4,
        )
        if n in offsets:
            ax.annotate(
                n,
                (x, y),
                xytext=offsets[n],
                textcoords="offset points",
                fontsize=6.8,
                va="center",
                weight="bold" if ours else "normal",
                color=ORANGE if ours else "#3F4A54",
                arrowprops={"arrowstyle": "-", "lw": 0.4, "color": "#A4ADB4", "shrinkB": 3},
            )
    style_axis(ax)
    ax.set(xlabel="Natural request fulfilment, NFR (%)", ylabel="ESConv empathy", xlim=(53, 100.8), ylim=(2.25, 4.32))
    save_fig(fig, out_dir, "core_tradeoff")


def plot_m2_position(models: list[str], values: dict, se_map: dict, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.48), sharey=True)
    yy = np.arange(len(models))
    specs = [
        ("M2_D1_both_success", "M2_D1L_both_success", "(a) Paired need retention", "Paired success (%)"),
        ("M2_D1_identity_strict", "M2_D1L_identity_strict", "(b) Strict identity attribution", "Unsupported attribution (%)"),
    ]
    for ax, (a, b, title, xlabel) in zip(axes, specs):
        for y, n in enumerate(models):
            xa, xb = 100.0 * values[n][a], 100.0 * values[n][b]
            sea, seb = 100.0 * se_map[n][a], 100.0 * se_map[n][b]
            ours = n == "MCARE"
            ax.plot([xa, xb], [y, y], color=ORANGE if ours else "#B3BDC5", lw=1.5 if ours else 0.9, zorder=1)
            ax.errorbar(xa, y, xerr=sea, fmt="none", ecolor=BLUE, elinewidth=0.65, capsize=1.6, capthick=0.55, alpha=0.65, zorder=2)
            ax.errorbar(xb, y, xerr=seb, fmt="none", ecolor=ORANGE, elinewidth=0.65, capsize=1.6, capthick=0.55, alpha=0.65, zorder=2)
        ax.scatter([100.0 * values[n][a] for n in models], yy, color=BLUE, s=18, label=r"D1: early disclosure ($\pm$1 SE)", zorder=3)
        ax.scatter(
            [100.0 * values[n][b] for n in models],
            yy,
            facecolors="white",
            edgecolors=ORANGE,
            s=20,
            marker="s",
            lw=0.9,
            label=r"D1L: latest message ($\pm$1 SE)",
            zorder=3,
        )
        ax.axhspan(len(models) - 1.45, len(models) - 0.55, color=ORANGE, alpha=0.08, zorder=0)
        style_axis(ax)
        ax.set_yticks(yy, models)
        for tl in ax.get_yticklabels():
            if tl.get_text() == "MCARE":
                tl.set_weight("bold")
                tl.set_color(ORANGE)
        ax.set_ylim(len(models) - 0.4, -0.7)
        ax.set_title(title, loc="left", fontsize=8.3)
        ax.set_xlabel(xlabel)
    axes[0].legend(loc="lower left", frameon=False, fontsize=6.8)
    fig.subplots_adjust(left=0.11, right=0.99, top=0.92, bottom=0.15, wspace=0.1)
    save_fig(fig, out_dir, "m2_position")


def plot_empathy_ecdf(observations: dict, out_dir: Path) -> None:
    show_models = [("Base", GRAY, "--"), ("RLVER", "#6B8E9B", "-."), ("Joint-DPO", BLUE, ":"), ("SafeDPO", GREEN, "-."), ("MCARE", ORANGE, "-")]
    fig, axes = plt.subplots(1, 2, figsize=(3.55, 2.0), sharey=True)
    for ax, (metric, title) in zip(axes, [("esconv_main_empathy", "(a) ESConv"), ("ed_main_empathy", "(b) ED")]):
        for name, col, ls in show_models:
            if name not in observations or metric not in observations[name]:
                continue
            vals = sorted(float(v[1]) for v in observations[name][metric].values() if v[1] is not None)
            if not vals:
                continue
            y = np.arange(1, len(vals) + 1) / len(vals)
            ax.step(vals, y, where="post", color=col, ls=ls, lw=1.4 if name == "MCARE" else 0.95, label=name)
        style_axis(ax)
        ax.set(xlabel="Empathy score (1-5)", xlim=(1.0, 5.0), ylim=(0, 1.02))
        ax.set_title(title, loc="left", fontsize=8.0)
    axes[0].set_ylabel("Empirical CDF")
    axes[0].legend(loc="upper left", frameon=False, fontsize=6.0)
    save_fig(fig, out_dir, "empathy_distribution")


def plot_paired_core(observations: dict, ours_name: str, baselines: list[str], out_dir: Path, n_boot: int = 4000) -> None:
    """Half-eye / interval plot of paired parent-cluster bootstrap differences (MCARE minus baseline)."""
    from common.prompts_and_schemas import ci_boot

    panels = [
        ("M1_NFR", "(a) Natural request fulfilment (%)", 100.0),
        ("esconv_main_empathy", "(b) ESConv empathy (1-5)", 1.0),
        ("ed_main_empathy", "(c) ED empathy (1-5)", 1.0),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.55), sharey=True)
    yy = np.arange(len(baselines))
    for ax, (metric, title, scale) in zip(axes, panels):
        a = {gid: (pid, float(v)) for gid, (pid, v) in observations.get(ours_name, {}).get(metric, {}).items() if v is not None}
        for y, bname in enumerate(baselines):
            b = {gid: (pid, float(v)) for gid, (pid, v) in observations.get(bname, {}).get(metric, {}).items() if v is not None}
            common_ids = sorted(set(a) & set(b))
            if not common_ids:
                continue
            by_parent: dict[str, list[float]] = defaultdict(list)
            for g in common_ids:
                by_parent[a[g][0]].append(scale * (a[g][1] - b[g][1]))
            diff, lo, hi = ci_boot(by_parent, lambda rows: st.mean(rows), n_boot=n_boot, seed=20261013)
            sig = lo > 0
            col = BLUE if sig else GRAY
            ax.plot([lo, hi], [y, y], color=col, lw=1.2, alpha=0.85, zorder=2)
            ax.scatter([diff], [y], color=ORANGE if sig else col, s=20, zorder=3)
        ax.axvline(0.0, color="#6B757D", lw=0.7, ls="--", zorder=1)
        style_axis(ax)
        ax.set_yticks(yy, baselines)
        ax.set_ylim(len(baselines) - 0.4, -0.6)
        ax.set_title(title, loc="left", fontsize=7.9)
        ax.set_xlabel(r"Paired difference $\Delta$ (MCARE $-$ baseline)")
    fig.subplots_adjust(left=0.14, right=0.99, top=0.88, bottom=0.17, wspace=0.14)
    save_fig(fig, out_dir, "paired_core")


def plot_m2_joint_empathy(models: list[str], m2_empathy: dict, m2_joint: dict, m2_se: dict, out_dir: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.16, 2.55), gridspec_kw={"width_ratios": [1.18, 1.0]})
    yy = np.arange(len(models))
    cats = [
        ("success=True,attr=False", BLUE, r"Retained & attribution-free ($J_{0,1L}$)"),
        ("success=True,attr=True", "#E09F67", "Retained, unsupported attribution"),
        ("success=False,attr=False", "#C5CED4", "Unfulfilled, attribution-free"),
        ("success=False,attr=True", ORANGE, "Unfulfilled, unsupported attribution"),
    ]
    left = np.zeros(len(models))
    j0_vals = np.zeros(len(models))
    j0_ses = np.zeros(len(models))
    for idx_c, (cname, col, lab) in enumerate(cats):
        vals = []
        for idx_m, n in enumerate(models):
            d = m2_joint[n]["joint_success_x_attribution"]["D1L"]
            denom = m2_joint[n]["J0c_n"]["D1L"]
            v = 100.0 * d.get(cname, 0) / denom
            vals.append(v)
            if idx_c == 0:
                j0_vals[idx_m] = v
                j0_ses[idx_m] = 100.0 * (m2_se.get(n, {}).get("J0_D1L_se") or 0.0)
        arr = np.array(vals)
        ax1.barh(yy, arr, left=left, height=0.64, color=col, edgecolor="white", lw=0.4, label=lab, zorder=3)
        left += arr
    ax1.errorbar(j0_vals, yy, xerr=j0_ses, fmt="none", ecolor="#18364A", elinewidth=0.75, capsize=1.6, capthick=0.6, zorder=5)
    ax1.set_yticks(yy, models)
    for tl in ax1.get_yticklabels():
        if tl.get_text() == "MCARE":
            tl.set_weight("bold")
            tl.set_color(ORANGE)
    ax1.set_ylim(len(models) - 0.4, -0.6)
    ax1.set_xlim(0, 100)
    ax1.set_xlabel("Proportion of M2 applicable pairs under D1L (%)")
    ax1.set_title("(a) Item-level joint status under D1L disclosure", loc="left", fontsize=8.2)
    style_axis(ax1)
    ax1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2, frameon=False, fontsize=6.7, handlelength=1.1, columnspacing=1.0)

    for n in models:
        x = 100.0 * m2_joint[n]["J0c_unsupported_free_paired_success"]["D1L"]
        xse = 100.0 * (m2_se.get(n, {}).get("J0_D1L_se") or 0.0)
        y0 = m2_empathy[n]["applicable_103"]["empathy_D0"]
        y1 = m2_empathy[n]["applicable_103"]["empathy_D1L"]
        y0se = m2_se.get(n, {}).get("emp_D0_se") or 0.0
        y1se = m2_se.get(n, {}).get("emp_D1L_se") or 0.0
        ours, base = n == "MCARE", n == "Base"
        col1 = ORANGE if ours else BLUE if base else GREEN
        ax2.plot([x, x], [y0, y1], color=ORANGE if ours else "#B8C2C8", lw=1.1 if ours else 0.7, ls=":", zorder=2)
        ax2.errorbar(x, y0, yerr=y0se, fmt="none", ecolor=ORANGE if ours else "#9AA4AC", elinewidth=0.5, capsize=1.2, capthick=0.45, alpha=0.55, zorder=2)
        ax2.errorbar(
            x, y1, xerr=xse, yerr=y1se, fmt="none", ecolor=col1, elinewidth=0.75 if ours else 0.55,
            capsize=1.8 if ours else 1.3, capthick=0.6 if ours else 0.45, alpha=0.85 if ours else 0.55, zorder=3,
        )
        ax2.scatter(x, y0, marker="o", s=12, facecolors="white", edgecolors=ORANGE if ours else "#8A949C", lw=0.7, zorder=3)
        ax2.scatter(x, y1, marker="*" if ours else "D" if base else "o", s=115 if ours else 26 if base else 22, c=col1, ec="white", lw=0.45, zorder=4)
    style_axis(ax2)
    ax2.set_xlim(14, 90)
    ax2.set_ylim(1.45, 4.42)
    ax2.set_xlabel(r"Attribution-free paired retention under D1L, $J_{0,1L}$ (%)")
    ax2.set_ylabel("M2 same-context empathy score")
    ax2.set_title("(b) Same-context empathy (D0 -> D1L) vs. $J_{0,1L}$", loc="left", fontsize=8.2)
    fig.subplots_adjust(left=0.10, right=0.99, top=0.90, bottom=0.25, wspace=0.22)
    save_fig(fig, out_dir, "m2_joint_empathy")


def plot_stage_tradeoff(stage_points: dict, anchor_points: dict, out_dir: Path) -> None:
    """Plot attribution-empathy trajectory across Stage 1, Specialist, and Stage 2."""
    fig, ax = plt.subplots(figsize=(3.45, 2.45))
    for n, pt in anchor_points.items():
        base = n == "Base"
        if base:
            ax.errorbar(pt["attr"], pt["emp"], xerr=pt.get("attr_se", 0.0), yerr=pt.get("emp_se", 0.0), fmt="none", ecolor=GRAY, elinewidth=0.55, capsize=1.3, alpha=0.65, zorder=2)
        ax.scatter(pt["attr"], pt["emp"], marker="D" if base else "o", s=20 if base else 14, c=GRAY if base else "#CBD2D7", ec="none", zorder=2)
    pts_xy = {}
    for label, pt in stage_points.items():
        ours = label == "MCARE"
        start = label in ("Stage 1 only", "Specialist")
        col = ORANGE if ours else BLUE
        ax.errorbar(
            pt["attr"], pt["emp"], xerr=pt.get("attr_se", 0.0), yerr=pt.get("emp_se", 0.0), fmt="none", ecolor=col,
            elinewidth=0.75 if ours else 0.6, capsize=1.8 if ours else 1.4, alpha=0.85 if ours else 0.65, zorder=3,
        )
        ax.scatter(pt["attr"], pt["emp"], marker="*" if ours else "s" if start else "o", s=110 if ours else 24, c=col, ec="white", lw=0.45, zorder=4)
        ax.annotate(label, (pt["attr"], pt["emp"]), xytext=tuple(pt.get("offset", (6, -4))), textcoords="offset points", fontsize=6.6, weight="bold" if ours else "normal", color=ORANGE if ours else INK)
        pts_xy[label] = (pt["attr"], pt["emp"])
    if "Stage 1 only" in pts_xy and "Specialist" in pts_xy:
        p1, pe = pts_xy["Stage 1 only"], pts_xy["Specialist"]
        ax.plot([p1[0], pe[0]], [p1[1], pe[1]], color="#7D8890", lw=0.7, ls=(0, (3, 2)), zorder=1)
        if "MCARE" in pts_xy:
            ax.annotate("", xy=pts_xy["MCARE"], xytext=p1, arrowprops={"arrowstyle": "-|>", "color": ORANGE, "lw": 0.9, "shrinkA": 4, "shrinkB": 6})
    style_axis(ax)
    ax.set(xlabel="Strict identity attribution, mean of D1 and D1L (%)", ylabel="Empathy, mean of ESConv and ED", xlim=(0, 70), ylim=(2.6, 4.38))
    save_fig(fig, out_dir, "stage_tradeoff")


def plot_fg_probes(f_summary: dict, g_summary: dict, models: list[str], labels: list[str], out_dir: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.16, 2.42), gridspec_kw={"width_ratios": [1.08, 1.0]})
    xx = np.arange(len(models))
    w = 0.24
    ek = dict(elinewidth=0.65, capsize=1.5, capthick=0.55, ecolor=INK)
    ax1.bar(xx - w, [f_summary[m]["fact_used"] for m in models], yerr=[f_summary[m]["fact_used_se"] for m in models], error_kw=ek, width=w, color=BLUE, edgecolor="white", lw=0.4, label=r"Fact used in +Evidence ($\uparrow$)", zorder=3)
    ax1.bar(xx, [f_summary[m]["fact_fab"] for m in models], yerr=[f_summary[m]["fact_fab_se"] for m in models], error_kw=ek, width=w, color="#D98C5F", edgecolor="white", lw=0.4, label=r"Fact hallucinated in Base ($\downarrow$)", zorder=3)
    ax1.bar(xx + w, [f_summary[m]["attr_base"] for m in models], yerr=[f_summary[m]["attr_base_se"] for m in models], error_kw=ek, width=w, color=GRAY, edgecolor="white", lw=0.4, label=r"Unsupported attr. in Base ($\downarrow$)", zorder=3)
    ax1.set_xticks(xx, labels)
    ax1.set_ylim(0, 108)
    ax1.set_ylabel("Rate (%)")
    ax1.set_title("(a) Personal evidence adaptation probe (F, n = 50)", loc="left", fontsize=8.2)
    style_axis(ax1)
    ax1_r = ax1.twinx()
    ax1_r.errorbar(xx, [f_summary[m]["emp_base"] for m in models], yerr=[f_summary[m]["emp_base_se"] for m in models], fmt="-o", color=ORANGE, lw=1.1, ms=4.0, elinewidth=0.65, capsize=1.6, label="Empathy (Base)", zorder=5)
    ax1_r.errorbar(xx, [f_summary[m]["emp_evid"] for m in models], yerr=[f_summary[m]["emp_evid_se"] for m in models], fmt="--s", color=GREEN, lw=1.0, ms=3.8, elinewidth=0.65, capsize=1.6, label="Empathy (+Evidence)", zorder=5)
    ax1_r.set_ylim(2.4, 4.18)
    ax1_r.set_ylabel("Empathy score (1-5)", color=INK)
    ax1_r.spines["top"].set_visible(False)

    dirs = [("listen_to_action", "Listen -> Action"), ("action_to_listen", "Action -> Listen"), ("both", "Combined")]
    gx = np.arange(len(dirs))
    gw = 0.15
    cols = [GRAY, "#6B8E9B", "#8E86AD", BLUE, ORANGE]
    for i, (m, lab, col) in enumerate(zip(models, labels, cols)):
        vals = [g_summary[m][d]["mean"] for d, _ in dirs]
        ses = [g_summary[m][d]["se"] for d, _ in dirs]
        ax2.bar(gx + (i - 2) * gw, vals, yerr=ses, error_kw=dict(elinewidth=0.6, capsize=1.3, ecolor=INK), width=gw, color=col, edgecolor="white", lw=0.4, label=lab, zorder=3)
    ax2.set_xticks(gx, [dl for _, dl in dirs])
    ax2.set_ylim(3.2, 5.35)
    ax2.set_ylabel("Updated-turn empathy score (1-5)")
    ax2.set_title("(b) Empathy under latest need update (G, n = 50)", loc="left", fontsize=8.2)
    style_axis(ax2)
    fig.subplots_adjust(left=0.07, right=0.93, top=0.89, bottom=0.26, wspace=0.34)
    save_fig(fig, out_dir, "fg_probes")


def main() -> None:
    parser = argparse.ArgumentParser(description="Render MCARE publication PDF figures.")
    parser.add_argument("--observations-json", type=Path, required=True, help="JSON mapping model -> metric -> id -> [parent_id, value]")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--font-dir", type=Path, default=None)
    parser.add_argument("--extra-json", type=Path, default=None, help="Optional JSON supplying m2_empathy, m2_joint, m2_se, stage_points, anchor_points, f_summary, g_summary")
    args = parser.parse_args()

    setup_style(args.font_dir)
    obs = load_json(args.observations_json)
    models = list(obs.keys())
    values: dict[str, dict[str, float]] = defaultdict(dict)
    se_map: dict[str, dict[str, float]] = defaultdict(dict)
    for m in models:
        for k, d in obs[m].items():
            mn, se, _ = mean_and_se([v[1] for v in d.values()])
            if mn is not None and se is not None:
                values[m][k] = mn
                se_map[m][k] = se

    if all("M1_NFR" in values[m] and "esconv_main_empathy" in values[m] for m in models):
        plot_core_tradeoff(models, values, se_map, args.out_dir)
    if all("M2_D1_both_success" in values[m] and "M2_D1L_identity_strict" in values[m] for m in models):
        plot_m2_position(models, values, se_map, args.out_dir)
    plot_empathy_ecdf(obs, args.out_dir)
    if "MCARE" in obs and len(models) > 1:
        baselines = [m for m in models if m != "MCARE"]
        plot_paired_core(obs, "MCARE", baselines, args.out_dir)
    if args.extra_json and args.extra_json.exists():
        extra = load_json(args.extra_json)
        if {"m2_empathy", "m2_joint", "m2_se"} <= set(extra):
            plot_m2_joint_empathy(models, extra["m2_empathy"], extra["m2_joint"], extra["m2_se"], args.out_dir)
        if {"stage_points", "anchor_points"} <= set(extra):
            plot_stage_tradeoff(extra["stage_points"], extra["anchor_points"], args.out_dir)
        if {"f_summary", "g_summary", "fg_models", "fg_labels"} <= set(extra):
            plot_fg_probes(extra["f_summary"], extra["g_summary"], extra["fg_models"], extra["fg_labels"], args.out_dir)
    print(f"Figures saved to {args.out_dir}")


if __name__ == "__main__":
    main()
