#!/usr/bin/env python3
"""Table VI & Fig. 8 mechanism transfer & candidate-source/gate diagnostics.

Computes direct paired gain-differences (delta^(1) = E(MCARE) - E(pi_1) vs
delta^(E) = E(Transfer_init_piE) - E(Specialist_piE)) and candidate-source /
natural-gate comparisons using M1 empathy labels and precomputed M2 tables.
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

M1_EMPATHY_SOURCE = {
    'Stage1_pi1': (EVAL_DIR / 'runs/Stage1_pi1', 'Stage1_pi1', 'empathy_labels_g.jsonl'),
    'MCARE': (EVAL_DIR / 'runs/MCARE', 'MCARE', 'empathy_labels_g.jsonl'),
    'Specialist_piE': (EVAL_DIR / 'runs/Specialist_piE', 'Specialist_piE', 'empathy_labels_g.jsonl'),
    'Transfer_init_piE': (EVAL_DIR / 'runs/Transfer_init_piE', 'Transfer_init_piE', 'empathy_labels_g.jsonl'),
    'SelfSampled_pi1': (EVAL_DIR / 'runs/SelfSampled_pi1', 'SelfSampled_pi1', 'empathy_labels_g.jsonl'),
    'SelfSampled_NoGate': (EVAL_DIR / 'runs/SelfSampled_NoGate', 'SelfSampled_NoGate', 'empathy_labels_g.jsonl'),
}


def read_jsonl(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def load_m1_empathy(label):
    d, actual, fname = M1_EMPATHY_SOURCE[label]
    return {r['id']: r['empathy'] for r in read_jsonl(d / fname) if r['label'] == actual and r.get('empathy') is not None}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    inputs = {r['id']: r for r in read_jsonl(EVAL_DIR / 'generation_inputs.jsonl')}
    m1_ids = [iid for iid, r in inputs.items() if 'esconv_main' in r['sets'] or 'ed_main' in r['sets']]
    assert len(m1_ids) == 495


    def source_of(iid):
        return 'ESConv' if iid.split(':')[0] == 'esconv' else 'ED'


    def boot_ci(vals_by_parent, n_boot=4000, seed=20261104):
        rng = np.random.default_rng(seed)
        keys = sorted(vals_by_parent)
        sums = np.array([sum(vals_by_parent[k]) for k in keys])
        counts = np.array([len(vals_by_parent[k]) for k in keys])
        point = sums.sum() / counts.sum()
        idx = rng.integers(0, len(keys), (n_boot, len(keys)))
        bs = sums[idx].sum(1) / counts[idx].sum(1)
        return point, float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


    # ---- Initialization & Reference Anchor Analysis ----
    emp = {label: load_m1_empathy(label) for label in ('Stage1_pi1', 'MCARE', 'Specialist_piE', 'Transfer_init_piE')}

    delta1_by_parent, deltaE_by_parent, D_by_parent = defaultdict(list), defaultdict(list), defaultdict(list)
    delta1_by_source, deltaE_by_source = defaultdict(list), defaultdict(list)
    n_joint = 0
    for iid in m1_ids:
        if not all(iid in emp[l] for l in emp):
            continue
        n_joint += 1
        d1 = emp['MCARE'][iid] - emp['Stage1_pi1'][iid]
        dE = emp['Transfer_init_piE'][iid] - emp['Specialist_piE'][iid]
        pid = inputs[iid]['parent_id']
        delta1_by_parent[pid].append(d1)
        deltaE_by_parent[pid].append(dE)
        D_by_parent[pid].append(d1 - dE)
        delta1_by_source[source_of(iid)].append(d1)
        deltaE_by_source[source_of(iid)].append(dE)

    e1_delta1 = boot_ci(delta1_by_parent)
    e1_deltaE = boot_ci(deltaE_by_parent)
    e1_D = boot_ci(D_by_parent)

    e1 = {
        'n_joint_items': n_joint,
        'delta1_pi1_to_MCARE': {'mean': e1_delta1[0], 'ci95': [e1_delta1[1], e1_delta1[2]]},
        'deltaE_piE_to_transfer_init_piE': {'mean': e1_deltaE[0], 'ci95': [e1_deltaE[1], e1_deltaE[2]]},
        'D_delta1_minus_deltaE': {'mean': e1_D[0], 'ci95': [e1_D[1], e1_D[2]]},
        'delta1_by_source_mean': {k: st.mean(v) for k, v in delta1_by_source.items()},
        'deltaE_by_source_mean': {k: st.mean(v) for k, v in deltaE_by_source.items()},
    }

    m2_emp = json.loads((OUT / 'm2_empathy_table.json').read_text())
    m2_joint = json.loads((OUT / 'm2_joint_transition_results.json').read_text())
    e1_labels = ['Stage1_pi1', 'MCARE', 'Specialist_piE', 'Transfer_init_piE']
    e1['m2_empathy_applicable_103'] = {l: m2_emp[l]['applicable_103'] for l in e1_labels}
    e1['m2_need_transition_eligible'] = {l: m2_joint[l]['transition_eligible'] for l in e1_labels}
    e1['m2_joint_success_x_attribution'] = {l: m2_joint[l]['joint_success_x_attribution'] for l in e1_labels}
    e1['m2_J0c'] = {l: m2_joint[l]['J0c_unsupported_free_paired_success'] for l in e1_labels}

    e1['boundary_transition_note'] = ('Only 1/120 M2 parent histories carry a nonempty negative_need; a '
                                       'request/restriction-boundary transition table on M2 is not meaningful '
                                       'at this n and is reported qualitatively only, not as a table.')

    # ---- Candidate Source & Natural-History Gate Comparison ----
    e3_labels = ['MCARE', 'SelfSampled_pi1', 'SelfSampled_NoGate']
    for l in e3_labels:
        if l not in emp:
            emp[l] = load_m1_empathy(l)

    e3_m1 = {}
    for l in e3_labels:
        by_source = defaultdict(list)
        for iid in m1_ids:
            if iid in emp[l]:
                by_source[source_of(iid)].append(emp[l][iid])
        e3_m1[l] = {src: st.mean(vals) for src, vals in by_source.items()}

    e3 = {
        'M1_empathy_by_source_recomputed': e3_m1,
        'm2_empathy_applicable_103': {l: m2_emp[l]['applicable_103'] for l in e3_labels},
        'm2_need_transition_eligible': {l: m2_joint[l]['transition_eligible'] for l in e3_labels},
        'm2_joint_success_x_attribution': {l: m2_joint[l]['joint_success_x_attribution'] for l in e3_labels},
        'm2_J0c': {l: m2_joint[l]['J0c_unsupported_free_paired_success'] for l in e3_labels},
    }

    (OUT / 'mechanism_inheritance.json').write_text(json.dumps(e1, ensure_ascii=False, indent=1))
    (OUT / 'mechanism_candidate_source_gate.json').write_text(json.dumps(e3, ensure_ascii=False, indent=1))

    lines = ['# Mechanism Analysis: Initialization & Reference Anchor Inheritance', '',
             f"n (shared M1 items) = {n_joint}", '',
             f"delta^(1) = E(MCARE) - E(Stage1_pi1): mean {e1_delta1[0]:+.4f}, 95% CI [{e1_delta1[1]:+.4f}, {e1_delta1[2]:+.4f}]",
             f"delta^(E) = E(Transfer_init_piE) - E(Specialist_piE): mean {e1_deltaE[0]:+.4f}, 95% CI [{e1_deltaE[1]:+.4f}, {e1_deltaE[2]:+.4f}]",
             f"D = delta^(1) - delta^(E): mean {e1_D[0]:+.4f}, 95% CI [{e1_D[1]:+.4f}, {e1_D[2]:+.4f}]", '',
             'Mean delta by dataset source:',
             f"  delta^(1): ESConv={e1['delta1_by_source_mean'].get('ESConv'):+.4f}  ED={e1['delta1_by_source_mean'].get('ED'):+.4f}",
             f"  delta^(E): ESConv={e1['deltaE_by_source_mean'].get('ESConv'):+.4f}  ED={e1['deltaE_by_source_mean'].get('ED'):+.4f}",
             '',
             '## M2 Same-Context Facets (n = 103 eligible parents)', '',
             '| Model | Empathy D0 | Empathy D1 | D1-D0 Delta [95% CI] | ok->ok (D1) | ok->err | err->ok | err->err | Paired+NoAttr | Paired+Attr |',
             '|---|---|---|---|---|---|---|---|---|---|']
    for l in e1_labels:
        a = m2_emp[l]['applicable_103']
        d1d = a.get('empathy_D1_minus_D0', {})
        tr = m2_joint[l]['transition_eligible']['D1']
        jt = m2_joint[l]['joint_success_x_attribution']['D1']
        fmtci = (f"{d1d['mean']:+.3f} [{d1d['ci95'][0]:+.3f},{d1d['ci95'][1]:+.3f}]" if d1d else 'N/A')
        lines.append(f"| {l} | {a['empathy_D0']:.3f} | {a['empathy_D1']:.3f} | {fmtci} | {tr.get('ok->ok',0)} | "
                      f"{tr.get('ok->err',0)} | {tr.get('err->ok',0)} | {tr.get('err->err',0)} | "
                      f"{jt.get('success=True,attr=False',0)} | {jt.get('success=True,attr=True',0)} |")
    lines.append('')
    lines.append(e1['boundary_transition_note'])
    (OUT / 'mechanism_inheritance.md').write_text('\n'.join(lines) + '\n')

    lines3 = ['# Mechanism Analysis: Specialist Candidate Supply & Natural-History Gate', '',
              '## M1 Empathy by Corpus', '',
              '| Model | ESConv Empathy | ED Empathy |', '|---|---|---|']
    for l in e3_labels:
        v = e3_m1[l]
        lines3.append(f"| {l} | {v.get('ESConv'):.3f} | {v.get('ED'):.3f} |")
    lines3 += ['', '## M2 Empathy, Transitions, and Joint State (n = 103 eligible parents)', '',
               '| Model | Empathy D0 | Empathy D1 | D1-D0 Delta [95% CI] | S0,1 | Attr D1 | ok->ok | ok->err | err->ok | err->err | J0,1 |',
               '|---|---|---|---|---|---|---|---|---|---|---|']
    for l in e3_labels:
        a = m2_emp[l]['applicable_103']
        d1d = a.get('empathy_D1_minus_D0', {})
        tr = m2_joint[l]['transition_eligible']['D1']
        j0c = m2_joint[l]['J0c_unsupported_free_paired_success']['D1']
        fmtci = (f"{d1d['mean']:+.3f} [{d1d['ci95'][0]:+.3f},{d1d['ci95'][1]:+.3f}]" if d1d else 'N/A')
        lines3.append(f"| {l} | {a['empathy_D0']:.3f} | {a['empathy_D1']:.3f} | {fmtci} | {a['S0_D1']:.4f} | "
                      f"{a['attr_D1']:.4f} | {tr.get('ok->ok',0)} | {tr.get('ok->err',0)} | {tr.get('err->ok',0)} | "
                      f"{tr.get('err->err',0)} | {j0c:.4f} |")
    (OUT / 'mechanism_candidate_source_gate.md').write_text('\n'.join(lines3) + '\n')

    print('\n'.join(lines))
    print()
    print('\n'.join(lines3))



if __name__ == '__main__':
    main()
