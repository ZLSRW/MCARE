#!/usr/bin/env python3
"""Tables VII & VIII: Probe F (personal-evidence adaptation) and Probe G (latest-need-update response).

Uses 4,000-resample parent-cluster bootstrap confidence intervals.
"""
import json
import statistics as st
from collections import defaultdict
from pathlib import Path
import numpy as np
import sys

_SRC = Path(__file__).resolve().parents[2]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
from common.config import get_paths as _get_paths
_CFG = _get_paths()
ROOT = _CFG.root
OUT = _CFG.exp_dir / 'probes_fg'

MODELS = ['Base', 'Base+Strong', 'Joint-DPO', 'Stage1_pi1', 'MCARE']


def read_jsonl(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def boot_ci(vals_by_parent, n_boot=4000, seed=20261104):
    rng = np.random.default_rng(seed)
    keys = sorted(vals_by_parent)
    sums = np.array([sum(vals_by_parent[k]) for k in keys])
    counts = np.array([len(vals_by_parent[k]) for k in keys])
    point = sums.sum() / counts.sum()
    idx = rng.integers(0, len(keys), (n_boot, len(keys)))
    bs = sums[idx].sum(1) / counts[idx].sum(1)
    return point, float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def rate_and_ci(flags_by_parent):
    """flags_by_parent: {parent_id: [0/1,...]} over DETERMINABLE items only -> (rate, lo, hi, n)."""
    n = sum(len(v) for v in flags_by_parent.values())
    if n == 0:
        return None, None, None, 0
    m, lo, hi = boot_ci(flags_by_parent)
    return m, lo, hi, n


# ============================== F ==============================
def compute_f():
    parents = {p['parent_id']: p for p in json.loads((OUT / 'F_inputs.json').read_text())}
    judge = defaultdict(dict)   # model -> id -> judgement
    for r in read_jsonl(OUT / 'F_judge_labels_g.jsonl'):
        if r['parse_ok']:
            judge[r['model']][r['id']] = r
    emp = defaultdict(dict)
    for r in read_jsonl(OUT / 'F_empathy_labels_g.jsonl'):
        if r.get('empathy') is not None:
            emp[r['model']][r['id']] = r['empathy']

    results = {}
    for m in MODELS:
        delivered_base, delivered_evid = defaultdict(list), defaultdict(list)
        paired_delivered = defaultdict(list)
        fact_used = defaultdict(list)
        fact_fabricated = defaultdict(list)
        identity_mention, identity_forbid = defaultdict(list), defaultdict(list)
        attr_base, attr_evid = defaultdict(list), defaultdict(list)
        emp_base, emp_evid, emp_delta = [], [], defaultdict(list)
        A_i = defaultdict(list)

        for pid, p in parents.items():
            base_id, evid_id = p['base']['id'], p['evidence']['id']
            jb, je = judge[m].get(base_id), judge[m].get(evid_id)
            db = dc = None
            if jb:
                jg = jb['judgement']
                if jg['delivered'] in ('yes', 'no'):
                    db = jg['delivered'] == 'yes'
                    delivered_base[pid].append(float(db))
                if jg['fact_fabricated'] in ('yes', 'no'):
                    fact_fabricated[pid].append(float(jg['fact_fabricated'] == 'yes'))
                group = identity_mention if p['identity_requirement'] == 'mention' else identity_forbid
                group[pid].append(float(jg['identity_followed'] == 'yes'))
                if jg['identity_assumption'] in ('yes', 'no'):
                    attr_base[pid].append(float(jg['identity_assumption'] == 'yes'))
            if je:
                jg = je['judgement']
                if jg['delivered'] in ('yes', 'no'):
                    dc = jg['delivered'] == 'yes'
                    delivered_evid[pid].append(float(dc))
                if jg['fact_used'] in ('yes', 'no'):
                    fact_used[pid].append(float(jg['fact_used'] == 'yes'))
                group = identity_mention if p['identity_requirement'] == 'mention' else identity_forbid
                group[pid].append(float(jg['identity_followed'] == 'yes'))
                if jg['identity_assumption'] in ('yes', 'no'):
                    attr_evid[pid].append(float(jg['identity_assumption'] == 'yes'))
            if db is not None and dc is not None:
                paired_delivered[pid].append(float(db and dc))
            if base_id in emp[m]:
                emp_base.append(emp[m][base_id])
            if evid_id in emp[m]:
                emp_evid.append(emp[m][evid_id])
            if base_id in emp[m] and evid_id in emp[m]:
                emp_delta[pid].append(emp[m][evid_id] - emp[m][base_id])
            if jb and je:
                jgb, jge = jb['judgement'], je['judgement']
                a = (jge.get('fact_used') == 'yes' and jgb.get('fact_fabricated') == 'no' and
                     jgb.get('delivered') == 'yes' and jge.get('delivered') == 'yes' and
                     jgb.get('identity_followed') == 'yes' and jge.get('identity_followed') == 'yes')
                A_i[pid].append(float(a))

        def r(d):
            return rate_and_ci(d)

        results[m] = {
            'delivered_base': r(delivered_base), 'delivered_evidence': r(delivered_evid),
            'paired_delivered': r(paired_delivered), 'fact_used_evidence': r(fact_used),
            'fact_fabricated_base': r(fact_fabricated), 'identity_followed_mention_group': r(identity_mention),
            'identity_followed_forbid_group': r(identity_forbid), 'attribution_base': r(attr_base),
            'attribution_evidence': r(attr_evid), 'empathy_base_mean': (st.mean(emp_base) if emp_base else None),
            'empathy_evidence_mean': (st.mean(emp_evid) if emp_evid else None),
            'empathy_evidence_minus_base': r(emp_delta), 'A_evidence_adaptation_success': r(A_i),
            'n_parents': len(parents),
        }
    (OUT / 'ProbeF_results.json').write_text(json.dumps(results, ensure_ascii=False, indent=1))

    lines = ['# Table VII (Probe F): Support Adaptation under Personal Evidence', '',
              '| Model | Base Delivered | Evidence Delivered | Paired Delivered | Fact Used | Base Fabricated | ID Mention OK | ID Forbid OK | Attr (Base) | Attr (Evidence) | Empathy (Base) | Empathy (Evidence) | Adaptation Success A |',
              '|---|---|---|---|---|---|---|---|---|---|---|---|---|']

    def fmt(x):
        m, lo, hi, n = x
        return f'{m:.3f}[{lo:.3f},{hi:.3f}] n={n}' if m is not None else 'N/A'

    for m, r in results.items():
        lines.append(f"| {m} | {fmt(r['delivered_base'])} | {fmt(r['delivered_evidence'])} | {fmt(r['paired_delivered'])} | "
                      f"{fmt(r['fact_used_evidence'])} | {fmt(r['fact_fabricated_base'])} | {fmt(r['identity_followed_mention_group'])} | "
                      f"{fmt(r['identity_followed_forbid_group'])} | {fmt(r['attribution_base'])} | {fmt(r['attribution_evidence'])} | "
                      f"{r['empathy_base_mean']:.3f} | {r['empathy_evidence_mean']:.3f} | {fmt(r['A_evidence_adaptation_success'])} |")
    (OUT / 'ProbeF_results.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


# ============================== G ==============================
def compute_g():
    parents = {p['parent_id']: p for p in json.loads((OUT / 'G_inputs.json').read_text())}
    judge = defaultdict(dict)
    for r in read_jsonl(OUT / 'G_judge_labels_g.jsonl'):
        if r['parse_ok']:
            judge[r['model']][r['id']] = r
    emp = defaultdict(dict)
    for r in read_jsonl(OUT / 'G_empathy_labels_g.jsonl'):
        if r.get('empathy') is not None:
            emp[r['model']][r['id']] = r['empathy']

    dirs = ['listen_to_action', 'action_to_listen', 'both']
    results = {}
    for m in MODELS:
        results[m] = {}
        for d in dirs:
            s_keep, s_update, u_paired, residue, attr_keep, attr_update = (defaultdict(list) for _ in range(6))
            emp_keep, emp_update = [], []
            for pid, p in parents.items():
                if d != 'both' and p['direction'] != d:
                    continue
                keep_id, upd_id = p['keep']['id'], p['update']['id']
                jk, ju = judge[m].get(keep_id), judge[m].get(upd_id)
                sk = su = None
                if jk:
                    jg = jk['judgement']
                    if jg['current_style_delivered'] in ('yes', 'no'):
                        sk = jg['current_style_delivered'] == 'yes'
                        s_keep[pid].append(float(sk))
                    if jg['identity_assumption'] in ('yes', 'no'):
                        attr_keep[pid].append(float(jg['identity_assumption'] == 'yes'))
                if ju:
                    jg = ju['judgement']
                    if jg['current_style_delivered'] in ('yes', 'no'):
                        su = jg['current_style_delivered'] == 'yes'
                        s_update[pid].append(float(su))
                    if jg['old_pattern_residue'] in ('yes', 'no'):
                        residue[pid].append(float(jg['old_pattern_residue'] == 'yes'))
                    if jg['identity_assumption'] in ('yes', 'no'):
                        attr_update[pid].append(float(jg['identity_assumption'] == 'yes'))
                if sk is not None and su is not None:
                    u_paired[pid].append(float(sk and su))
                if keep_id in emp[m]:
                    emp_keep.append(emp[m][keep_id])
                if upd_id in emp[m]:
                    emp_update.append(emp[m][upd_id])

            def r(dd):
                return rate_and_ci(dd)

            results[m][d] = {
                'S_keep': r(s_keep), 'S_update': r(s_update), 'U_paired': r(u_paired),
                'old_pattern_residue_update': r(residue), 'attribution_keep': r(attr_keep),
                'attribution_update': r(attr_update), 'empathy_keep_mean': (st.mean(emp_keep) if emp_keep else None),
                'empathy_update_mean': (st.mean(emp_update) if emp_update else None),
            }
    (OUT / 'ProbeG_results.json').write_text(json.dumps(results, ensure_ascii=False, indent=1))

    def fmt(x):
        m, lo, hi, n = x
        return f'{m:.3f}[{lo:.3f},{hi:.3f}] n={n}' if m is not None else 'N/A'

    lines = ['# Table VIII (Probe G): Response Adaptation under Latest-Need Updates', '']
    for d in dirs:
        lines.append(f'## Direction: {d}')
        lines.append('| Model | S_keep | S_update | Paired U | Prior Residue | Attr (keep) | Attr (update) | Empathy (keep) | Empathy (update) |')
        lines.append('|---|---|---|---|---|---|---|---|---|')
        for m in MODELS:
            r = results[m][d]
            lines.append(f"| {m} | {fmt(r['S_keep'])} | {fmt(r['S_update'])} | {fmt(r['U_paired'])} | {fmt(r['old_pattern_residue_update'])} | "
                          f"{fmt(r['attribution_keep'])} | {fmt(r['attribution_update'])} | {r['empathy_keep_mean']:.3f} | {r['empathy_update_mean']:.3f} |")
        lines.append('')
    (OUT / 'ProbeG_results.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    compute_f()
    compute_g()
