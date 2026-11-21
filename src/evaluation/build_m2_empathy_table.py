#!/usr/bin/env python3
"""Table III & VI M2 empathy table: join M2 empathy scores with need-success and attribution labels.

Evaluates the 103-parent need-applicable subset (plus 120-full and 73-natural subsets)
and computes D1-D0 / D1L-D0 empathy deltas with 4,000-resample parent-cluster bootstrap CIs.
"""
import json
import statistics as st
from collections import defaultdict
from pathlib import Path
import sys
import numpy as np

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
from common.config import get_paths as _get_paths
_CFG = _get_paths()
ROOT = _CFG.root
EVAL_DIR = _CFG.eval_dir
OUT = _CFG.exp_dir / 'analysis'

from evaluation.need_analysis import dim_status, need_status  # noqa: E402

LABEL_SOURCE = {
    'Base': (EVAL_DIR, 'Base'),
    'Base+Strong': (EVAL_DIR, 'Base_Strong'),
    'RLFF': (EVAL_DIR, 'RLFF'),
    'PERM': (EVAL_DIR, 'PERM'),
    'PEER-RUR': (EVAL_DIR, 'PEER_RUR'),
    'PEER-SFT': (EVAL_DIR, 'PEER_SFT'),
    'ESCRAG': (EVAL_DIR, 'ESCRAG'),
    'RLVER': (EVAL_DIR, 'RLVER'),
    'Vanilla-SFT': (EVAL_DIR, 'Vanilla_SFT'),
    'Quality-DPO': (EVAL_DIR, 'Quality_DPO'),
    'Support-DPO': (EVAL_DIR, 'Support_DPO'),
    'Joint-DPO': (EVAL_DIR, 'Joint_DPO'),
    'SafeDPO': (EVAL_DIR, 'SafeDPO'),
    'MCARE': (EVAL_DIR / 'runs/MCARE', 'MCARE'),
    'Stage1_pi1': (EVAL_DIR / 'runs/Stage1_pi1', 'Stage1_pi1'),
    'Specialist_piE': (EVAL_DIR / 'runs/Specialist_piE', 'Specialist_piE'),
    'Transfer_init_piE': (EVAL_DIR / 'runs/Transfer_init_piE', 'Transfer_init_piE'),
    'Interp_075_025': (EVAL_DIR / 'runs/Interp_075_025', 'Interp_075_025'),
    'SelfSampled_pi1': (EVAL_DIR / 'runs/SelfSampled_pi1', 'SelfSampled_pi1'),
    'SelfSampled_NoGate': (EVAL_DIR / 'runs/SelfSampled_NoGate', 'SelfSampled_NoGate'),
}


def read_jsonl(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    audit = {r['id']: r for r in read_jsonl(EVAL_DIR / 'need_audit_raw_g.jsonl')}
    pairs = read_jsonl(EVAL_DIR / 'm2_pairs.jsonl')
    fits = {r['id']: r for r in read_jsonl(EVAL_DIR / 'm2_fit_raw_g.jsonl')}
    disclosed = {p[k]: p['history_id'] for p in pairs for k in ('d1_id', 'd1l_id')}
    emp_rows = read_jsonl(OUT / 'm2_empathy_labels_g.jsonl')
    empathy = defaultdict(dict)  # label -> id -> score
    for r in emp_rows:
        if r.get('empathy') is not None:
            empathy[r['label']][r['id']] = r['empathy']

    eligible_pairs = [p for p in pairs if (audit.get(p['history_id']) or {}).get('flags', {}).get('eligible_strict')]
    assert len(eligible_pairs) == 103, len(eligible_pairs)


    def natural_of(p):
        f = fits.get(p['history_id'], {})
        return bool(f.get('parse_ok') and f.get('natural') and f['template'] == p['template'] and f['insert_index'] == p['insert_index'])


    natural_pairs = [p for p in eligible_pairs if natural_of(p)]


    def boot_ci(vals_by_parent, n_boot=4000, seed=20261104):
        """vals_by_parent: {parent_id: [values]} -> (mean, lo, hi) via parent-cluster bootstrap."""
        rng = np.random.default_rng(seed)
        keys = sorted(vals_by_parent)
        sums = np.array([sum(vals_by_parent[k]) for k in keys])
        counts = np.array([len(vals_by_parent[k]) for k in keys])
        point = sums.sum() / counts.sum()
        idx = rng.integers(0, len(keys), (n_boot, len(keys)))
        bs = sums[idx].sum(1) / counts[idx].sum(1)
        return point, float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


    results = {}
    for label, (directory, actual) in LABEL_SOURCE.items():
        primary = {r['id']: r for r in read_jsonl(directory / 'need_judge_labels_g.jsonl') if r['label'] == actual}
        statuses = {}
        for gid, row in primary.items():
            if not (row['parse_ok'] and row['judgement'] is not None):
                continue
            a = audit.get(disclosed.get(gid, gid))
            if a is None:
                continue
            s = dim_status(row['judgement'], a['verified'], actual, gid, None)
            statuses[gid] = {**s, 'need': need_status(s)}
        emp = empathy.get(label, {})

        def agg(subset_pairs):
            s01 = {'D1': [], 'D1L': []}
            attr = {'D1': [], 'D1L': []}
            empD0, empD1, empD1L = [], [], []
            d1_minus_d0_by_parent, d1l_minus_d0_by_parent = defaultdict(list), defaultdict(list)
            for p in subset_pairs:
                d0 = statuses.get(p['d0_id'], {}).get('need')
                for tag, key in (('D1', 'd1_id'), ('D1L', 'd1l_id')):
                    dc_id = p[key]
                    dc = statuses.get(dc_id, {}).get('need')
                    if d0 in ('ok', 'err') and dc in ('ok', 'err'):
                        s01[tag].append(1.0 if (d0 == 'ok' and dc == 'ok') else 0.0)
                    row = primary.get(dc_id)
                    if row and row['parse_ok'] and row['judgement'] is not None:
                        q = row['judgement'].get('identity_assumption')
                        if q in ('yes', 'no'):
                            attr[tag].append(1.0 if q == 'yes' else 0.0)
                if p['d0_id'] in emp:
                    empD0.append(emp[p['d0_id']])
                if p['d1_id'] in emp:
                    empD1.append(emp[p['d1_id']])
                if p['d1l_id'] in emp:
                    empD1L.append(emp[p['d1l_id']])
                if p['d0_id'] in emp and p['d1_id'] in emp:
                    d1_minus_d0_by_parent[p['parent_id']].append(emp[p['d1_id']] - emp[p['d0_id']])
                if p['d0_id'] in emp and p['d1l_id'] in emp:
                    d1l_minus_d0_by_parent[p['parent_id']].append(emp[p['d1l_id']] - emp[p['d0_id']])
            out = {
                'n': len(subset_pairs),
                'S0_D1': st.mean(s01['D1']) if s01['D1'] else None, 'S0_D1L': st.mean(s01['D1L']) if s01['D1L'] else None,
                'attr_D1': st.mean(attr['D1']) if attr['D1'] else None, 'attr_D1L': st.mean(attr['D1L']) if attr['D1L'] else None,
                'empathy_D0': st.mean(empD0) if empD0 else None, 'empathy_D1': st.mean(empD1) if empD1 else None,
                'empathy_D1L': st.mean(empD1L) if empD1L else None,
            }
            if d1_minus_d0_by_parent:
                m, lo, hi = boot_ci(d1_minus_d0_by_parent)
                out['empathy_D1_minus_D0'] = {'mean': m, 'ci95': [lo, hi]}
            if d1l_minus_d0_by_parent:
                m, lo, hi = boot_ci(d1l_minus_d0_by_parent)
                out['empathy_D1L_minus_D0'] = {'mean': m, 'ci95': [lo, hi]}
            return out

        results[label] = {'applicable_103': agg(eligible_pairs), 'natural_73': agg(natural_pairs)}

    (OUT / 'm2_empathy_table.json').write_text(json.dumps(results, ensure_ascii=False, indent=1))

    lines = ['# M2 Same-Context Need Retention, Attribution, and Empathy (n = 103 eligible parents)', '',
             '| Model | S0,1 | S0,1L | Attr D1 | Attr D1L | Empathy D0 | Empathy D1 | Empathy D1L | D1-D0 Delta [95% CI] | D1L-D0 Delta [95% CI] |',
             '|---|---|---|---|---|---|---|---|---|---|']
    for label, r in results.items():
        a = r['applicable_103']
        d1d = a.get('empathy_D1_minus_D0', {})
        d1ld = a.get('empathy_D1L_minus_D0', {})
        fmt = lambda x: f'{x:.4f}' if x is not None else 'N/A'
        fmtci = lambda d: (f"{d['mean']:+.3f} [{d['ci95'][0]:+.3f},{d['ci95'][1]:+.3f}]" if d else 'N/A')
        lines.append(f"| {label} | {fmt(a['S0_D1'])} | {fmt(a['S0_D1L'])} | {fmt(a['attr_D1'])} | {fmt(a['attr_D1L'])} | "
                      f"{fmt(a['empathy_D0'])} | {fmt(a['empathy_D1'])} | {fmt(a['empathy_D1L'])} | {fmtci(d1d)} | {fmtci(d1ld)} |")
    (OUT / 'm2_empathy_table.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))



if __name__ == '__main__':
    main()
