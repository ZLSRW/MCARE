#!/usr/bin/env python3
"""Table IV: Controlled negative-boundary NVR overall, exact two-sided McNemar tests
versus Base and Base+Strong, and five-category error breakdown (30 items per category).
"""
import json
import math
from pathlib import Path
import sys

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
from common.config import get_paths as _get_paths
_CFG = _get_paths()
ROOT = _CFG.root
EVAL_DIR = _CFG.eval_dir
OUT = _CFG.exp_dir / 'analysis'

from evaluation.need_analysis import dim_status  # noqa: E402

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


def exact_mcnemar_p(n01, n10):
    """Two-sided exact binomial McNemar p-value on the discordant pairs."""
    n = n01 + n10
    if n == 0:
        return 1.0
    k = min(n01, n10)

    def pmf(i):
        return math.comb(n, i) * (0.5 ** n)
    p_le_k = sum(pmf(i) for i in range(0, k + 1))
    return min(1.0, 2 * p_le_k)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    controlled = read_jsonl(EVAL_DIR / 'controlled_negative.jsonl')
    audit = {r['id']: r for r in read_jsonl(EVAL_DIR / 'need_audit_raw_g.jsonl')}
    cat_of = {r['id']: r['kind'] for r in controlled}
    ids150 = [r['id'] for r in controlled]
    assert len(ids150) == 150 and len(set(cat_of.values())) == 5


    def neg_status(label, directory, actual):
        primary = {r['id']: r for r in read_jsonl(directory / 'need_judge_labels_g.jsonl') if r['label'] == actual}
        out = {}
        for gid in ids150:
            row = primary.get(gid)
            if not row or not (row['parse_ok'] and row['judgement'] is not None):
                continue
            a = audit.get(gid)
            if a is None:
                continue
            s = dim_status(row['judgement'], a['verified'], actual, gid, None)
            out[gid] = s.get('neg')  # 'ok' (no violation) / 'err' (violated) / 'unk'
        return out


    status_by_label = {label: neg_status(label, d, a) for label, (d, a) in LABEL_SOURCE.items()}

    # ---- Overall NVR and exact two-sided McNemar tests vs. Base and Base+Strong ----



    def pair_counts(a_status, b_status):
        """a=err,b=ok ; a=ok,b=err counts over ids determinable for BOTH."""
        n_a_only = n_b_only = n_both = n_neither = 0
        for gid in ids150:
            sa, sb = a_status.get(gid), b_status.get(gid)
            if sa not in ('ok', 'err') or sb not in ('ok', 'err'):
                continue
            ea, eb = sa == 'err', sb == 'err'
            if ea and not eb:
                n_a_only += 1
            elif eb and not ea:
                n_b_only += 1
            elif ea and eb:
                n_both += 1
            else:
                n_neither += 1
        return n_a_only, n_b_only, n_both, n_neither


    overall = {}
    for label, st in status_by_label.items():
        determ = [v for v in st.values() if v in ('ok', 'err')]
        n_err = sum(v == 'err' for v in determ)
        overall[label] = {'n_determinable': len(determ), 'n_violations': n_err, 'nvr_pct': 100 * n_err / len(determ) if determ else None}

    mcnemar = {}
    for ref in ('Base', 'Base+Strong'):
        mcnemar[f'MCARE_vs_{ref}'] = {}
        a_only, b_only, both, neither = pair_counts(status_by_label['MCARE'], status_by_label[ref])
        p = exact_mcnemar_p(a_only, b_only)
        mcnemar[f'MCARE_vs_{ref}'] = {'MCARE_only_violates': a_only, f'{ref}_only_violates': b_only,
                                       'both_violate': both, 'neither_violates': neither, 'exact_mcnemar_p': p}

    # ---- C(2): per-category breakdown ----
    per_cat = {}
    for label, st in status_by_label.items():
        per_cat[label] = {}
        for cat in sorted(set(cat_of.values())):
            ids_cat = [gid for gid in ids150 if cat_of[gid] == cat]
            determ = [st.get(gid) for gid in ids_cat if st.get(gid) in ('ok', 'err')]
            n_err = sum(v == 'err' for v in determ)
            per_cat[label][cat] = f'{n_err}/{len(determ)}'

    (OUT / 'boundary_breakdown_results.json').write_text(json.dumps(
        {'overall_NVR': overall, 'exact_mcnemar': mcnemar, 'per_category_errors': per_cat}, ensure_ascii=False, indent=1))

    lines = ['# Controlled Negative-Boundary Violation Rate (NVR), Exact McNemar Tests, and Category Breakdown', '']
    lines.append('## Overall NVR and Exact Paired Tests')
    lines.append('| Model | Determinable n | Violations | NVR% |')
    lines.append('|---|---|---|---|')
    for label, o in overall.items():
        lines.append(f"| {label} | {o['n_determinable']} | {o['n_violations']} | {o['nvr_pct']:.2f} |")
    lines.append('')
    for k, v in mcnemar.items():
        other_k = [x for x in v if x.endswith('_only_violates') and 'MCARE' not in x][0]
        lines.append(
            f"**{k}**: MCARE-only violations = {v.get('MCARE_only_violates', v.get(list(v)[0]))}, "
            f"baseline-only violations = {v[other_k]}, both violate = {v['both_violate']}, "
            f"neither violates = {v['neither_violates']}, exact McNemar p = {v['exact_mcnemar_p']:.4f}"
        )
    lines.append('')
    lines.append('## Five-Category Error Counts (errors / 30)')
    cats = sorted(set(cat_of.values()))
    lines.append('| Model | ' + ' | '.join(cats) + ' |')
    lines.append('|---|' + '---|' * len(cats))
    for label in LABEL_SOURCE:
        lines.append(f"| {label} | " + ' | '.join(per_cat[label][c] for c in cats) + ' |')
    (OUT / 'boundary_breakdown_results.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))



if __name__ == '__main__':
    main()
