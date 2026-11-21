#!/usr/bin/env python3
"""Table III & Fig. 7 M2 need transitions, 2x2 joint compliance x attribution states,
and unsupported-attribution-free paired success J_{0,c} = mean(s0 * sc * (1 - zc)) over the
103-parent need-applicable subset.
"""
import json
import statistics as st
from collections import defaultdict
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
    # the audit file is keyed by the D0/history id only; D1/D1L ids must be remapped to it (same as prepare_data.py)
    disclosed = {p[k]: p['history_id'] for p in pairs for k in ('d1_id', 'd1l_id')}

    results = {}
    for label, (directory, actual) in LABEL_SOURCE.items():
        primary = {r['id']: r for r in read_jsonl(directory / 'need_judge_labels_g.jsonl') if r['label'] == actual}
        statuses = {}
        for gid, row in primary.items():
            if not (row['parse_ok'] and row['judgement'] is not None):
                continue
            a = audit.get(disclosed.get(gid, gid))  # D1/D1L ids map back to their D0/history audit record
            verified = a['verified'] if a else None
            if verified is None:
                continue
            s = dim_status(row['judgement'], verified, actual, gid, None)
            statuses[gid] = {**s, 'need': need_status(s)}

        trans = {'D1': defaultdict(int), 'D1L': defaultdict(int)}
        joint = {'D1': defaultdict(int), 'D1L': defaultdict(int)}
        j0c_vals = {'D1': [], 'D1L': []}
        full_joint = {'D1': defaultdict(int), 'D1L': defaultdict(int)}  # all 120, not just eligible_strict/applicable
        n_eligible = {'D1': 0, 'D1L': 0}
        n_natural = {'D1': 0, 'D1L': 0}

        for p in pairs:
            flags = (audit.get(p['history_id']) or {}).get('flags') or {}
            eligible = flags.get('eligible_strict')
            f = fits.get(p['history_id'], {})
            natural = (f.get('parse_ok') and f.get('natural') and f['template'] == p['template']
                       and f['insert_index'] == p['insert_index'])
            for tag, key in (('D1', 'd1_id'), ('D1L', 'd1l_id')):
                d0, dc = p['d0_id'], p[key]
                s0 = statuses.get(d0, {}).get('need')
                sc = statuses.get(dc, {}).get('need')
                # attribution (identity_assumption) read directly from this model's own judgement on the Dc id
                zc = None
                if dc in primary and primary[dc]['parse_ok'] and primary[dc]['judgement'] is not None:
                    q = primary[dc]['judgement'].get('identity_assumption')
                    zc = 1 if q == 'yes' else 0 if q == 'no' else None
                if s0 in ('ok', 'err') and sc in ('ok', 'err'):
                    full_joint[tag][(s0, sc)] += 1
                if not eligible:
                    continue
                n_eligible[tag] += 1
                if natural:
                    n_natural[tag] += 1
                if s0 in ('ok', 'err') and sc in ('ok', 'err'):
                    trans[tag][(s0, sc)] += 1
                    both = (s0 == 'ok' and sc == 'ok')
                    if zc is not None:
                        joint[tag][(both, bool(zc))] += 1
                        j0c_vals[tag].append(1.0 if (both and not zc) else 0.0)

        def strkeys(d):
            return {f'{k[0]}->{k[1]}': v for k, v in d.items()}

        results[label] = {
            'n_eligible': n_eligible, 'n_natural': n_natural,
            'transition_eligible': {t: strkeys(trans[t]) for t in trans},
            'transition_all120': {t: strkeys(full_joint[t]) for t in full_joint},
            'joint_success_x_attribution': {t: {f'success={k[0]},attr={k[1]}': v for k, v in joint[t].items()} for t in joint},
            'J0c_unsupported_free_paired_success': {t: (st.mean(j0c_vals[t]) if j0c_vals[t] else None) for t in j0c_vals},
            'J0c_n': {t: len(j0c_vals[t]) for t in j0c_vals},
        }

    (OUT / 'm2_joint_transition_results.json').write_text(json.dumps(results, ensure_ascii=False, indent=1))

    lines = ['# M2 Need Transitions, Joint Compliance x Attribution State, and J0,c Retention', '']
    lines.append('| Model | Cond. | Eligible n | ok->ok | ok->err | err->ok | err->err | Paired+NoAttr | Paired+Attr | Unpaired+NoAttr | Unpaired+Attr | J0,c |')
    lines.append('|---|---|---|---|---|---|---|---|---|---|---|---|')
    for label, r in results.items():
        for t in ('D1', 'D1L'):
            tr = r['transition_eligible'][t]
            jt = r['joint_success_x_attribution'][t]
            j0c = r['J0c_unsupported_free_paired_success'][t]
            j0c_str = f"{j0c:.4f}" if j0c is not None else "N/A"
            lines.append(f"| {label} | {t} | {r['n_eligible'][t]} | {tr.get('ok->ok',0)} | {tr.get('ok->err',0)} | "
                          f"{tr.get('err->ok',0)} | {tr.get('err->err',0)} | {jt.get('success=True,attr=False',0)} | "
                          f"{jt.get('success=True,attr=True',0)} | {jt.get('success=False,attr=False',0)} | "
                          f"{jt.get('success=False,attr=True',0)} | {j0c_str} |")
    (OUT / 'm2_joint_transition_results.md').write_text('\n'.join(lines) + '\n')
    print('wrote', OUT / 'm2_joint_transition_results.json', 'and .md')
    print('MCARE D1:', results['MCARE']['transition_eligible']['D1'], 'J0c=', results['MCARE']['J0c_unsupported_free_paired_success']['D1'])
    print('MCARE D1L:', results['MCARE']['transition_eligible']['D1L'], 'J0c=', results['MCARE']['J0c_unsupported_free_paired_success']['D1L'])
    print('Base D1:', results['Base']['transition_eligible']['D1'], 'J0c=', results['Base']['J0c_unsupported_free_paired_success']['D1'])



if __name__ == '__main__':
    main()
