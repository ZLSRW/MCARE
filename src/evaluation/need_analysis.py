"""Compute evaluation tables and parent-cluster bootstrap confidence intervals.

  main_tables.csv           Table I & III core quality and M1/M2 semantic alignment metrics
  m2_tables.csv             M2 paired identity-disclosure counterfactual breakdown
  confidence_intervals.csv  95% parent-cluster bootstrap CIs (4,000 resamples) and paired deltas
  NEED_EVAL_RESULTS.md      Markdown summary table

Unknown handling: a dimension judged 'unclear' without verified evidence renders
that dimension (and the history MAR) UNKNOWN; unknown items are excluded from the
denominator and reported as judge coverage.
usage: PYTHONPATH=src python src/evaluation/need_analysis.py --labels Base Base+Strong RLFF PERM ... MCARE --ref MCARE
"""
import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
from common import prompts_and_schemas as C  # noqa: E402

import re
MENTION = re.compile(r"autis|neurodiver|spectrum|sensory", re.I)
GROUPS: dict[str, list[str]] = {}
NAMES = {
    "Base": "Base",
    "Base+Strong": "Base+Strong",
    "RLFF": "RLFF",
    "PERM": "PERM",
    "PEER-RUR": "PEER-RUR",
    "PEER-SFT": "PEER-SFT",
    "ESCRAG": "ESCRAG",
    "RLVER": "RLVER",
    "Vanilla-SFT": "Vanilla-SFT",
    "Quality-DPO": "Quality-DPO",
    "Support-DPO": "Support-DPO",
    "Joint-DPO": "Joint-DPO",
    "SafeDPO": "SafeDPO",
    "MCARE": "MCARE",
    "Stage1_pi1": "Stage 1 only (pi_1)",
    "Specialist_piE": "Empathy specialist (pi_E)",
    "Transfer_init_piE": "Transfer: init./ref. pi_E",
    "Interp_075_025": "Parameter interpolation (0.75:0.25)",
    "SelfSampled_pi1": "Self-sampled candidates (pi_1)",
    "SelfSampled_NoGate": "Self-sampling, no natural-history gate",
}


def seedwise(stat, members):
    """stat over a unit list tagged with 'seed' -> mean of the per-seed statistics (None if any seed has none)."""
    def f(items):
        vals = []
        for m in members:
            v = stat([i for i in items if i.get("seed") == m])
            if v is None:
                return None
            vals.append(v)
        return sum(vals) / len(vals)
    return f


def load_scores(label):
    """ESC-RANK and ArmoRM per id -> dict of floats (None if unparsed)."""
    out = defaultdict(dict)
    p = C.EVAL / "scores" / f"esc_rank_{label}.jsonl"
    if p.exists():
        for r in C.read_jsonl(p):
            for d in ("overall", "empathy", "skill"):
                out[r["id"]]["esc_" + d] = r["scores"][d]["score"]
    p = C.EVAL / "scores" / f"armorm_{label}.jsonl"
    if p.exists():
        for r in C.read_jsonl(p):
            for d in ("helpfulness", "correctness", "coherence"):
                out[r["id"]]["arm_" + d] = r["raw_rewards"][d]
    return out


def dim_status(j, audit, label=None, gid=None, ver=None):
    """Per-history dimension outcomes from one Phase-B judgement.  'ok'/'err'/'unk'/None (not applicable).
    Projection / fact flags of the primary judge count only when the thinking-mode verification confirmed them (flag_verification.jsonl); without a verification record the primary flag is used (and marked in 'unverified')."""
    npos, nneg = len(audit["positive_need"]), len(audit["negative_need"])
    out = {}
    req = j.get("requests", [])
    if npos:
        out["pos"] = "err" if "no" in req else ("unk" if "unclear" in req or len(req) != npos else "ok")       # 'err' = not all requests fulfilled
    res = j.get("restrictions", [])
    if nneg:
        out["neg"] = "err" if "yes" in res else ("unk" if "unclear" in res or len(res) != nneg else "ok")       # 'err' = a restriction violated

    def verified(raw, kind):
        if raw == "yes":
            if ver is not None and (label, gid, kind) in ver:
                return "err" if ver[(label, gid, kind)] else "ok"
            return "err"
        return "ok" if raw == "no" else "unk"
    out["proj"] = verified(j.get("unsupported_projection"), "proj")
    out["proj_primary"] = {"yes": "err", "no": "ok"}.get(j.get("unsupported_projection"), "unk")
    if any(f.get("current", True) for f in audit["facts"]) or audit["updates"]:
        out["fact"] = verified(j.get("fact_conflict"), "fact")
        out["fact_primary"] = {"yes": "err", "no": "ok"}.get(j.get("fact_conflict"), "unk")
    out["ident"] = {"yes": "err", "no": "ok"}.get(j.get("identity_assumption"), "unk")
    return out


def mar_status(st):
    dims = [st[k] for k in ("pos", "neg", "proj", "fact") if k in st]
    if "err" in dims:
        return "err"
    return "unk" if "unk" in dims else "ok"


def need_status(st):
    dims = [st[k] for k in ("pos", "neg") if k in st]
    if "err" in dims:
        return "err"
    return "unk" if "unk" in dims else "ok"


def rate(units, key):
    """units: list of dicts with 'cluster' and key in {'ok','err'}-valued; returns stat function over 'err' share among determinable units."""
    return lambda items: (lambda v: (sum(x == "err" for x in v) / len(v)) if v else None)([i[key] for i in items if i.get(key) in ("ok", "err")])


def mean_stat(key):
    return lambda items: (lambda v: (sum(v) / len(v)) if v else None)([i[key] for i in items if i.get(key) is not None])


def by_cluster(units):
    d = defaultdict(list)
    for u in units:
        d[u["cluster"]].append(u)
    return d


def fmt(p, lo, hi, pct=True):
    if p is None:
        return "n/a"
    f = (lambda x: f"{100 * x:.1f}") if pct else (lambda x: f"{x:.2f}")
    return f"{f(p)} [{f(lo)}, {f(hi)}]" if lo is not None else f(p)


def paired_diff(units_a, units_b, stat, n_boot=4000):
    """Same ids in both; clusters resampled jointly.  Returns (diff, lo, hi) of stat(a) - stat(b)."""
    import random
    ca, cb = by_cluster(units_a), by_cluster(units_b)
    keys = sorted(set(ca) & set(cb))
    if not keys:
        return None, None, None
    sa, sb = stat([x for k in keys for x in ca[k]]), stat([x for k in keys for x in cb[k]])
    if sa is None or sb is None:
        return None, None, None
    rng = random.Random(C.SEED)
    res = []
    for _ in range(n_boot):
        pick = [keys[rng.randrange(len(keys))] for _ in keys]
        x, y = stat([i for k in pick for i in ca[k]]), stat([i for k in pick for i in cb[k]])
        if x is not None and y is not None:
            res.append(x - y)
    res.sort()
    return sa - sb, (res[int(0.025 * len(res))] if res else None), (res[min(len(res) - 1, int(0.975 * len(res)))] if res else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", nargs="+", required=True)
    ap.add_argument("--ref", default="MCARE")
    ap.add_argument("--n-boot", type=int, default=4000)
    ap.add_argument("--track", choices=["gemma", "custom"], default="gemma", help="evaluation judge track: 'gemma' uses cross-family Gemma-4-31B-it outputs (*_g)")
    ap.add_argument("--audit-suffix", default=None, help="override: suffix of need_audit_raw*.jsonl")
    ap.add_argument("--verify-suffix", default=None, help="override: suffix of flag_verification*.jsonl")
    ap.add_argument("--empathy-suffix", default=None, help="override: suffix of empathy_labels*.jsonl")
    ap.add_argument("--out-tag", default="", help="extra suffix appended to output filenames")
    ap.add_argument("--m2-subset", choices=["all", "bothfit"], default="all", help="'bothfit': restrict M2 pairs to those passing naturalness verification")
    ap.add_argument("--judge-suffix", default="", help="e.g. _gemma: use need_judge_labels_gemma.jsonl (raw verdicts of the cross-family judge; no verification pass) and write main_tables_gemma.csv etc.")
    args = ap.parse_args()
    if args.track == "gemma":
        defaults = {"judge": "_g", "audit": "_g", "verify": "_g", "empathy": "_gemma", "out": "_gemmatrack"}
    else:
        defaults = {"judge": "", "audit": "", "verify": "", "empathy": "", "out": ""}
    # the --judge-suffix view (e.g. '_gemma' = Gemma judging the frozen Qwen annotations) keeps its old meaning when --track is qwen
    sfx = args.judge_suffix or defaults["judge"]
    audit_sfx = args.audit_suffix if args.audit_suffix is not None else defaults["audit"]
    verify_sfx = args.verify_suffix if args.verify_suffix is not None else (defaults["verify"] if not args.judge_suffix else None)
    emp_sfx = args.empathy_suffix if args.empathy_suffix is not None else (defaults["empathy"] if not args.judge_suffix else args.judge_suffix)
    out_sfx = defaults["out"] if args.track == "gemma" else args.judge_suffix
    if args.m2_subset == "bothfit":
        out_sfx += "_bothfit"
    out_sfx += args.out_tag
    E = C.EVAL
    gen_inputs = {r["id"]: r for r in C.read_jsonl(E / "generation_inputs.jsonl")}
    audit = {r["id"]: r for r in C.read_jsonl(E / f"need_audit_raw{audit_sfx}.jsonl")}
    pairs = C.read_jsonl(E / "m2_pairs.jsonl")
    if args.m2_subset == "bothfit":
        fit_q = {r["id"]: r for r in C.read_jsonl(E / "m2_fit_raw.jsonl")}
        fit_g = {r["id"]: r for r in C.read_jsonl(E / "m2_fit_raw_g.jsonl")}
        pairs = [p for p in pairs if fit_q.get(p["history_id"], {}).get("natural") and fit_g.get(p["history_id"], {}).get("natural")]
        print(f"M2 pairs kept (natural under both fit checks): {len(pairs)}", flush=True)
    labels = [l for l in args.labels if (E / "gen" / f"{l}.jsonl").exists()]
    prim = defaultdict(dict)
    if (E / f"need_judge_labels{sfx}.jsonl").exists():
        for r in C.read_jsonl(E / f"need_judge_labels{sfx}.jsonl"):
            prim[r["label"]][r["id"]] = r
    emp = defaultdict(dict)
    if (E / f"empathy_labels{emp_sfx}.jsonl").exists():
        for r in C.read_jsonl(E / f"empathy_labels{emp_sfx}.jsonl"):
            emp[r["label"]][r["id"]] = r["empathy"]
    tm = defaultdict(dict)
    if (E / "text_metrics.jsonl").exists():
        for r in C.read_jsonl(E / "text_metrics.jsonl"):
            tm[r["label"]][r["id"]] = r
    ver = {}
    if verify_sfx is not None and (E / f"flag_verification{verify_sfx}.jsonl").exists():
        for r in C.read_jsonl(E / f"flag_verification{verify_sfx}.jsonl"):
            ver[(r["label"], r["id"], r["kind"])] = r["confirmed"]
    corpus = json.load(open(E / "text_metrics_corpus.json")) if (E / "text_metrics_corpus.json").exists() else {}
    gens = {l: {r["id"]: r for r in C.read_jsonl(E / "gen" / f"{l}.jsonl")} for l in labels}
    d1_base = {p["d1_id"]: p["history_id"] for p in pairs}
    d1_base.update({p["d1l_id"]: p["history_id"] for p in pairs if "d1l_id" in p})
    rows_csv, ci_csv, md = [], [], []

    def add(table, label, subset, metric, n, p, lo, hi, extra=""):
        rows_csv.append({"table": table, "label": label, "set": subset, "metric": metric, "n": n, "value": p, "lo": lo, "hi": hi, "note": extra})

    # ------------------------------------------------------------------ units per label and set
    def units_for(label, set_name):
        scores = load_scores(label)
        out = []
        for gid, g in gen_inputs.items():
            if set_name not in g["sets"] or gid not in gens[label]:
                continue
            u = {"id": gid, "cluster": g["parent_id"], "hit_cap": float(gens[label][gid]["hit_cap"]), "words": len(gens[label][gid]["response"].split()),
                 **{k: v for k, v in scores.get(gid, {}).items()}}
            if "esc_overall" in u or "esc_empathy" in u or "esc_skill" in u:
                u["esc_valid"] = 1.0 if all(u.get(k) is not None for k in ("esc_overall", "esc_empathy", "esc_skill")) else 0.0
            if gid in emp[label]:
                u["emp"] = emp[label][gid]
            if gid in tm[label]:
                u["rougeL"], u["bert_f1"] = tm[label][gid]["rougeL"], tm[label][gid]["bert_f1"]
            a = audit.get(d1_base.get(gid, gid))
            ver_audit = a["verified"] if a and a["verified"] else None
            u["audit"] = ver_audit
            if gid in prim[label] and prim[label][gid]["judgement"] and ver_audit is not None:
                st = dim_status(prim[label][gid]["judgement"], ver_audit, label, gid, ver)
                jr = prim[label][gid]["judgement"].get("requests", [])
                u["req_items"] = [(e["kind"], v) for e, v in zip(ver_audit["positive_need"], jr)] if len(jr) == len(ver_audit["positive_need"]) else []
                u.update({"st": st, "pos": st.get("pos"), "neg": st.get("neg"), "proj": st.get("proj"), "fact": st.get("fact"), "ident": st.get("ident"),
                          "mar": mar_status(st), "need": need_status(st), "judge_uncertain": bool(prim[label][gid]["judgement"].get("uncertain"))})
            out.append(u)
        return out

    U = {(l, s): units_for(l, s) for l in labels for s in ("esconv_main", "esconv_sens", "ed_main", "m2_d0", "m2_d1", "m2_d1l")}
    nb = args.n_boot
    groups = {g: m for g, m in GROUPS.items() if all(x in labels for x in m)}
    for g, members in groups.items():
        for sname in ("esconv_main", "esconv_sens", "ed_main", "m2_d0", "m2_d1", "m2_d1l"):
            U[(g, sname)] = [dict(u, seed=m) for m in members for u in U[(m, sname)]]
    W = lambda l, stat: seedwise(stat, groups[l]) if l in groups else stat      # noqa: E731
    shown = labels + list(groups)

    def boot(units, stat):
        return C.ci_boot(by_cluster(units), stat, n_boot=nb)

    # ------------------------------------------------------------------ Table 1: quality (all natural main histories)
    quality_keys = [("esc_overall", "ESC-RANK Overall", False), ("esc_empathy", "ESC-RANK Empathy", False), ("esc_skill", "ESC-RANK Skill", False), ("emp", "Independent Empathy (1-5)", False),
                    ("arm_helpfulness", "ArmoRM Helpfulness", False), ("arm_correctness", "ArmoRM Correctness", False), ("arm_coherence", "ArmoRM Coherence", False),
                    ("bert_f1", "BERTScore-F1", True), ("rougeL", "ROUGE-L F1", True), ("words", "Words", False), ("hit_cap", "Hit cap (rate)", True), ("esc_valid", "ESC-RANK parsed (rate)", True)]
    md.append(f"# Need-Aligned Evaluation Summary (Judge: {C.GEMMA_MODEL}; cluster bootstrap CIs)\n")
    for set_name, title in (("esconv_main", "Table 1a - ESConv main set (one pre-fixed history per parent)"), ("ed_main", "Table 1b - EmpatheticDialogues (frozen histories; n shown below)")):
        md += [f"## {title}\n", "| model | n | " + " | ".join(k[1] for k in quality_keys) + " |", "|---|---|" + "---|" * len(quality_keys)]
        for l in shown:
            us = U[(l, set_name)]
            cells = []
            for key, _, pct in quality_keys:
                p, lo, hi = boot(us, W(l, mean_stat(key))) if any(key in u for u in us) else (None, None, None)
                n_valid = sum(key in u and u[key] is not None for u in us)
                add("table1_" + set_name, l, set_name, key, n_valid, p, lo, hi)
                cells.append(fmt(p, lo, hi, pct=pct and key != "words") if key not in ("words",) else (f"{p:.0f}" if p is not None else "n/a"))
            md.append(f"| {NAMES.get(l, l)} | {len(us)} | " + " | ".join(cells) + " |")
        md.append("")
        md += [f"Distinct-n (corpus level, {set_name}; same tokenisation as the earlier panels, no interval): " + "; ".join(
            f"{NAMES.get(l, l)} D1 {corpus[l][set_name]['distinct1']:.3f} / D2 {corpus[l][set_name]['distinct2']:.3f}" for l in labels if l in corpus and set_name in corpus[l]), ""]
        for l in labels:
            if l in corpus and set_name in corpus[l]:
                add("table1_" + set_name, l, set_name, "distinct1", corpus[l][set_name]["n"], corpus[l][set_name]["distinct1"], None, None)
                add("table1_" + set_name, l, set_name, "distinct2", corpus[l][set_name]["n"], corpus[l][set_name]["distinct2"], None, None)
    # ESC-RANK parse coverage
    md += ["ESC-RANK parse coverage (valid scores / rows) is in main_tables.csv (metric rows carry n).\n"]

    # ------------------------------------------------------------------ Table 2: semantic need alignment
    def need_table(set_name, title, subset_filter, tag):
        md.append(f"## {title}\n")
        md.append("| model | N+ histories (parents) | NFR | N- histories (parents) | NVR | D_need n (unknown) | MAR | UPR | FCR n | FCR |")
        md.append("|---|---|---|---|---|---|---|---|---|---|")
        for l in shown:
            us = [u for u in U[(l, set_name)] if "st" in u and subset_filter(u)]
            pos = [u for u in us if u["pos"] in ("ok", "err")]
            neg = [u for u in us if u["neg"] in ("ok", "err")]
            need = [u for u in us if u["audit"] and (u["audit"]["positive_need"] or u["audit"]["negative_need"])]
            mar_det = [u for u in need if u["mar"] in ("ok", "err")]
            fct = [u for u in us if u["fact"] in ("ok", "err")]
            cells = []
            for name, sub, key in (("NFR", pos, "pos"), ("NVR", neg, "neg"), ("MAR", mar_det, "mar"), ("UPR", [u for u in need if u["proj"] in ("ok", "err")], "proj"), ("FCR", fct, "fact")):
                p, lo, hi = boot(sub, W(l, rate(sub, key)))
                if name == "NFR" and p is not None:
                    p, lo, hi = 1 - p, (1 - hi if hi is not None else None), (1 - lo if lo is not None else None)   # NFR = fulfilled share
                add("table2_" + tag, l, set_name, name, len(sub), p, lo, hi, f"parents={len({u['cluster'] for u in sub})}")
                cells.append((fmt(p, lo, hi), len(sub), len({u["cluster"] for u in sub})))
            unk = len(need) - len(mar_det)
            md.append(f"| {NAMES.get(l, l)} | {cells[0][1]} ({cells[0][2]}) | {cells[0][0]} | {cells[1][1]} ({cells[1][2]}) | {cells[1][0]} | {len(need)} ({unk}) | {cells[2][0]} | {cells[3][0]} | {cells[4][1]} | {cells[4][0]} |")
        md.append("")
    need_table("esconv_main", "Table 2a - Need alignment, ESConv main set (eligible histories only; % with 95% parent-cluster bootstrap CI; NFR = share of fulfilled)", lambda u: True, "esconv_main")
    need_table("esconv_sens", "Table 2b - Sensitivity: all eligible histories (up to 6 per parent), clustered by parent", lambda u: True, "esconv_sens")
    need_table("ed_main", "Table 2c - ED histories with a verified explicit need (diagnostic)", lambda u: True, "ed_need")

    # ------------------------------------------------------------------ NFR by request kind (per request; clusters = parents)
    kinds_all = ["advice", "info", "listen", "form", "other"]
    md += ["## Table 2d - Request fulfilment by request kind, ESConv main + sensitivity sets pooled by history (per request, % fulfilled among yes/no verdicts; parent-cluster bootstrap)\n", "| model | " + " | ".join(kinds_all) + " |", "|---|" + "---|" * len(kinds_all)]
    for l in shown:
        row = []
        for kd in kinds_all:
            items = [dict(cluster=u["cluster"], seed=u.get("seed"), v=1.0 if v == "yes" else 0.0)
                     for sname in ("esconv_main", "esconv_sens") for u in U[(l, sname)] if "st" in u for (k, v) in u.get("req_items", []) if k == kd and v in ("yes", "no")]
            ms = lambda its: (sum(i["v"] for i in its) / len(its)) if its else None      # noqa: E731
            p, lo, hi = boot(items, W(l, ms)) if items else (None, None, None)
            add("table2d_by_kind", l, "esconv_main+sens", f"fulfilled[{kd}]", len(items), p, lo, hi)
            row.append(f"{fmt(p, lo, hi)} (n={len(items)})" if p is not None else "n/a")
        md.append(f"| {NAMES.get(l, l)} | " + " | ".join(row) + " |")
    md.append("")

    # ------------------------------------------------------------------ Table 2e (exploratory, plan section 3.3): MAR among high-empathy replies on the same replies
    pooled = [u["emp"] for l in labels for u in U[(l, "esconv_main")] if "st" in u and u.get("emp") is not None and u["mar"] in ("ok", "err")]
    if pooled:
        thr = sorted(pooled)[len(pooled) // 2]
        md += [f"## Table 2e - Exploratory joint view on the same ESConv main replies: independent empathy at or above the pooled median ({thr:.2f}) versus MAR (threshold fixed as the pooled median over all models before looking at any model)\n",
               "| model | share high empathy | MAR among high-empathy replies | MAR among the rest | n high / n rest |", "|---|---|---|---|---|"]
        for l in shown:
            us = [u for u in U[(l, "esconv_main")] if "st" in u and u.get("emp") is not None and u["mar"] in ("ok", "err")]
            hi_u, lo_u = [u for u in us if u["emp"] >= thr], [u for u in us if u["emp"] < thr]
            sh = boot(us, W(l, lambda its: (sum(i["emp"] >= thr for i in its) / len(its)) if its else None))
            mh = boot(hi_u, W(l, rate(hi_u, "mar"))) if hi_u else (None, None, None)
            ml = boot(lo_u, W(l, rate(lo_u, "mar"))) if lo_u else (None, None, None)
            add("table2e_joint", l, "esconv_main", "share_high_empathy", len(us), *sh)
            add("table2e_joint", l, "esconv_main", "MAR_high_empathy", len(hi_u), *mh)
            md.append(f"| {NAMES.get(l, l)} | {fmt(*sh)} | {fmt(*mh)} | {fmt(*ml)} | {len(hi_u)} / {len(lo_u)} |")
        md.append("")

    # ------------------------------------------------------------------ paired differences (reference minus other), headline metrics
    md.append(f"## Paired differences: {NAMES.get(args.ref, args.ref)} minus each other model (same histories, percentage points / score points, 95% parent-cluster bootstrap)\n")
    md.append("| model | set | metric | difference [95% CI] | n |")
    md.append("|---|---|---|---|---|")
    ref = args.ref
    for l in labels:
        if l == ref or ref not in labels:
            continue
        for set_name, metric, stat_of in (("esconv_main", "NFR", lambda: ("pos", True)), ("esconv_main", "NVR", lambda: ("neg", False)), ("esconv_main", "MAR", lambda: ("mar", False)),
                                          ("esconv_main", "UPR", lambda: ("proj", False)), ("esconv_main", "FCR", lambda: ("fact", False)),
                                          ("esconv_sens", "NFR", lambda: ("pos", True)), ("esconv_sens", "MAR", lambda: ("mar", False)),
                                          ("esconv_main", "ESC-RANK Overall", lambda: ("esc_overall", None)), ("esconv_main", "ESC-RANK Empathy", lambda: ("esc_empathy", None)), ("esconv_main", "Independent Empathy", lambda: ("emp", None)),
                                          ("ed_main", "ESC-RANK Empathy", lambda: ("esc_empathy", None)), ("ed_main", "Independent Empathy", lambda: ("emp", None))):
            key, is_nfr = stat_of()
            ua = [u for u in U[(ref, set_name)] if (key in ("esc_overall", "esc_empathy", "emp") and u.get(key) is not None) or (key not in ("esc_overall", "esc_empathy", "emp") and u.get(key) in ("ok", "err"))]
            ub = [u for u in U[(l, set_name)] if (key in ("esc_overall", "esc_empathy", "emp") and u.get(key) is not None) or (key not in ("esc_overall", "esc_empathy", "emp") and u.get(key) in ("ok", "err"))]
            ids = {u["id"] for u in ua} & {u["id"] for u in ub}
            ua, ub = [u for u in ua if u["id"] in ids], [u for u in ub if u["id"] in ids]
            if not ids:
                continue
            stat = mean_stat(key) if is_nfr is None else rate(None, key)
            d, lo, hi = paired_diff(ua, ub, stat, nb)
            if is_nfr:                       # report NFR as the fulfilled share
                d, lo, hi = (-d if d is not None else None), (-hi if hi is not None else None), (-lo if lo is not None else None)
            scale = 1 if is_nfr is None else 100
            cell = "n/a" if d is None else f"{scale * d:+.2f} [{scale * lo:+.2f}, {scale * hi:+.2f}]"
            md.append(f"| {NAMES.get(l, l)} | {set_name} | {metric} | {cell} | {len(ids)} |")
            ci_csv.append({"ref": ref, "other": l, "set": set_name, "metric": metric, "diff": d, "lo": lo, "hi": hi, "n": len(ids)})
    md.append("")

    # ------------------------------------------------------------------ Table 3: M2 paired identity-disclosure counterfactual
    m2_csv = []
    for cond_name, d1_set, d1_key, title_extra in (("D1", "m2_d1", "d1_id", "disclosure in the first substantive Seeker message (primary)"), ("D1L", "m2_d1l", "d1l_id", "same sentence appended to the END of the latest Seeker message (sensitivity; naturalness not separately checked)")):
        if pairs and d1_key not in pairs[0]:
            continue
        md += [f"## Table 3 [{cond_name}] - Counterfactual autism disclosure, {title_extra} (paired M2; hypothetical disclosure sentence inserted into public ESConv test histories; NOT real autistic users)\n",
               "| model | pairs | D0 need success | D1 need success | both | D0 MAR | D1 MAR | new error under D1 | D0 identity-UPR | D1 identity-UPR (strict) | D1 identity-generalisation (lenient) | dMAR (D1-D0) | dUPR (D1-D0) | dESC overall | dESC empathy | identity mentioned (regex) |",
               "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for l in shown:
            members = groups[l] if l in groups else [l]
            units = []
            for m in members:
                u0 = {u["id"]: u for u in U[(m, "m2_d0")] if "st" in u}
                u1 = {u["id"]: u for u in U[(m, d1_set)] if "st" in u}
                for p in pairs:
                    a, b = u0.get(p["d0_id"]), u1.get(p[d1_key])
                    if not a or not b:
                        continue
                    units.append({"cluster": p["parent_id"], "seed": m, "s0": a["need"], "s1": b["need"], "m0": a["mar"], "m1": b["mar"], "i0": a["ident"], "i1": b["ident"],
                              "men": float(bool(MENTION.search(gens[m][p[d1_key]]["response"]))) if p[d1_key] in gens[m] else None,
                              "l1": (("err" if ver[(m, p[d1_key], "ident_lenient")] else "ok") if (m, p[d1_key], "ident_lenient") in ver else None),
                                  "esc_o": (b.get("esc_overall") - a.get("esc_overall")) if a.get("esc_overall") is not None and b.get("esc_overall") is not None else None,
                                  "esc_e": (b.get("esc_empathy") - a.get("esc_empathy")) if a.get("esc_empathy") is not None and b.get("esc_empathy") is not None else None})
            if not units:
                continue
            n_pairs = len({u["cluster"] for u in units})
            det = lambda key: [x for x in units if x[key] in ("ok", "err")]   # noqa: E731
            def share_ok(key):
                return lambda items: (lambda v: (sum(x == "ok" for x in v) / len(v)) if v else None)([i[key] for i in items if i[key] in ("ok", "err")])
            def share_err(key):
                return lambda items: (lambda v: (sum(x == "err" for x in v) / len(v)) if v else None)([i[key] for i in items if i[key] in ("ok", "err")])
            def diff_err(k1, k0):
                def f(items):
                    v = [(1.0 if i[k1] == "err" else 0.0) - (1.0 if i[k0] == "err" else 0.0) for i in items if i[k1] in ("ok", "err") and i[k0] in ("ok", "err")]
                    return (sum(v) / len(v)) if v else None
                return f
            both = lambda items: (lambda v: (sum(i["s0"] == "ok" and i["s1"] == "ok" for i in v) / len(v)) if v else None)([i for i in items if i["s0"] in ("ok", "err") and i["s1"] in ("ok", "err")])   # noqa: E731
            newerr = lambda items: (lambda v: (sum(i["m1"] == "err" and i["m0"] == "ok" for i in v) / len(v)) if v else None)([i for i in items if i["m0"] in ("ok", "err") and i["m1"] in ("ok", "err")])   # noqa: E731
            res = {}
            for name, stat, sub in (("D0 need success", share_ok("s0"), det("s0")), ("D1 need success", share_ok("s1"), det("s1")), ("both success", both, units),
                                    ("D0 MAR", share_err("m0"), det("m0")), ("D1 MAR", share_err("m1"), det("m1")), ("new error under D1", newerr, units),
                                    ("D0 identity-UPR", share_err("i0"), det("i0")), ("D1 identity-UPR", share_err("i1"), det("i1")), ("D1 identity-generalisation (lenient)", share_err("l1"), det("l1")),
                                    ("dMAR (D1-D0)", diff_err("m1", "m0"), units), ("dUPR (D1-D0)", diff_err("i1", "i0"), units),
                                    ("dESC overall", mean_stat("esc_o"), units), ("dESC empathy", mean_stat("esc_e"), units), ("identity mentioned (regex)", mean_stat("men"), units)):
                p, lo, hi = boot(sub, W(l, stat))
                if name.startswith("dESC"):
                    res[name] = f"{p:+.2f} [{lo:+.2f}, {hi:+.2f}]" if p is not None else "n/a"
                elif name.startswith("dMAR") or name.startswith("dUPR"):
                    res[name] = f"{100 * p:+.1f} [{100 * lo:+.1f}, {100 * hi:+.1f}]" if p is not None else "n/a"
                else:
                    res[name] = fmt(p, lo, hi)
                m2_csv.append({"label": l, "condition": cond_name, "metric": name, "n": len(sub), "value": p, "lo": lo, "hi": hi})
            md.append(f"| {NAMES.get(l, l)} | {n_pairs} | " + " | ".join(res[k] for k in ("D0 need success", "D1 need success", "both success", "D0 MAR", "D1 MAR", "new error under D1", "D0 identity-UPR", "D1 identity-UPR", "D1 identity-generalisation (lenient)", "dMAR (D1-D0)", "dUPR (D1-D0)", "dESC overall", "dESC empathy", "identity mentioned (regex)")) + " |")
    md.append("")
    md.append("Caveats: all semantic labels are produced by the local Qwen3.8-27B (same model family as the Qwen-based policies; no human labels); the audit and the judge were validated on a hand-labelled sample (judge_validation.json). "
              "The old evaluation panels were used for configuration and seed selection; the ESConv test parents of this set partly overlap them (dataset_manifest.json).\n")
    with open(E / f"main_tables{out_sfx}.csv", "w", newline="") as h:
        w = csv.DictWriter(h, fieldnames=["table", "label", "set", "metric", "n", "value", "lo", "hi", "note"])
        w.writeheader()
        w.writerows(rows_csv)
    with open(E / f"m2_tables{out_sfx}.csv", "w", newline="") as h:
        w = csv.DictWriter(h, fieldnames=["label", "condition", "metric", "n", "value", "lo", "hi"])
        w.writeheader()
        w.writerows(m2_csv)
    with open(E / f"confidence_intervals{out_sfx}.csv", "w", newline="") as h:
        w = csv.DictWriter(h, fieldnames=["ref", "other", "set", "metric", "diff", "lo", "hi", "n"])
        w.writeheader()
        w.writerows(ci_csv)
    (E / f"NEED_EVAL_RESULTS{out_sfx}.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
