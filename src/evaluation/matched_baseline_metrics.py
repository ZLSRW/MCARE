"""Complete reference metrics and unchanged frozen-panel statistics for one new weight."""
import argparse
import collections
import json
import os
import subprocess
import sys
from pathlib import Path
import numpy as np
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
from baselines.matched_pool_common import _CFG, ROOT, RUN, POLICY, read, dump, sha


def interval(values, inputs, total):
    grouped = collections.defaultdict(list)
    for gid, value in values.items():
        if value is not None:
            grouped[inputs[gid]['parent_id']].append(value)
    if not grouped:
        return {'value': None, 'lo': None, 'hi': None, 'n': 0, 'unknown': total}
    parents = sorted(grouped)
    num = np.array([sum(grouped[p]) for p in parents]); den = np.array([len(grouped[p]) for p in parents])
    picks = np.random.default_rng(20261012).integers(0, len(parents), (4000, len(parents)))
    reps = num[picks].sum(axis=1) / den[picks].sum(axis=1)
    return {'value': float(num.sum()/den.sum()), 'lo': float(np.quantile(reps, .025)),
            'hi': float(np.quantile(reps, .975)), 'n': int(den.sum()), 'parents': len(parents),
            'numerator': float(num.sum()), 'unknown': total-int(den.sum())}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--label', required=True)
    ap.add_argument('--device', default='cuda'); args = ap.parse_args()
    evaluation = RUN / 'eval'
    os.environ['NEED_EVAL_DIR'] = str(evaluation)
    inputs = {r['id']: r for r in read(evaluation / 'generation_inputs.jsonl')}
    targets = {r['id']: r['target'] for name in ['esconv_test', 'ed_test'] for r in read(POLICY / f'{name}.jsonl')}
    refs = {gid: targets[gid] for gid, g in inputs.items() if set(g['sets']) & {'esconv_main', 'ed_main'}}
    assert len(refs) == 495
    path = evaluation / 'references.json'
    if path.exists():
        assert json.loads(path.read_text()) == refs
    else:
        dump(path, refs)
    subprocess.run([sys.executable, str(_SRC / 'evaluation/text_metrics.py'),
                    '--labels', args.label, '--device', args.device], check=True)
    metrics = [r for r in read(evaluation / 'text_metrics.jsonl') if r['label'] == args.label]
    assert {r['id'] for r in metrics} == set(refs) and len(metrics) == 495
    subprocess.run([sys.executable, str(_SRC / 'evaluation/need_analysis.py'),
                    '--labels', args.label, '--ref', args.label, '--track', 'gemma',
                    '--empathy-suffix', '_g', '--out-tag', '_'+args.label], check=True)
    from evaluation.need_analysis import dim_status, need_status, mar_status
    primary = {r['id']: r for r in read(evaluation / 'need_judge_labels_g.jsonl') if r['label'] == args.label}
    flags = {(r['label'], r['id'], r['kind']): r['confirmed'] for r in read(evaluation / 'flag_verification_g.jsonl')}
    audit = {r['id']: r for r in read(evaluation / 'need_audit_raw_g.jsonl')}
    pairs = read(evaluation / 'm2_pairs.jsonl')
    disclosed = {p[key]: p['history_id'] for p in pairs for key in ['d1_id', 'd1l_id']}
    statuses = {}
    for gid, r in primary.items():
        assert r['parse_ok'] and r.get('judgement') is not None
        a = audit[disclosed.get(gid, gid)]['verified']
        st = dim_status(r['judgement'], a, args.label, gid, flags)
        statuses[gid] = {**st, 'need': need_status(st), 'mar': mar_status(st)}
    positive = json.loads((evaluation / 'm1_complete_ids.json').read_text())['positive_ids']
    controlled = read(evaluation / 'controlled_negative.jsonl')
    natural_neg = [gid for gid, g in inputs.items() if 'esconv_main' in g['sets'] and
                   (audit.get(gid, {}).get('verified') or {}).get('negative_need')]
    assert len(positive) == 255 and len(controlled) == 150 and len(natural_neg) == 3
    def success(st):
        return 1. if st == 'ok' else 0. if st == 'err' else None
    def failure(st):
        return 1. if st == 'err' else 0. if st == 'ok' else None
    out = {'label': args.label, 'seed': 42, 'resampling': '4000 parent-cluster bootstrap draws',
           'M1_all_positive_NFR': interval({gid: success(statuses[gid]['pos']) for gid in positive}, inputs, 255),
           'controlled_NVR': interval({r['id']: failure(statuses[r['id']]['neg']) for r in controlled}, inputs, 150),
           'natural_negative_NVR_diagnostic': interval({gid: failure(statuses[gid]['neg']) for gid in natural_neg}, inputs, 3)}
    fits = {r['id']: r for r in read(evaluation / 'm2_fit_raw_g.jsonl')}
    eligible = [p for p in pairs if (audit[p['history_id']].get('flags') or {}).get('eligible_strict')]
    approved = [p for p in eligible if (f := fits.get(p['history_id'])) and f['parse_ok'] and f['natural'] and
                f['template'] == p['template'] and f['insert_index'] == p['insert_index']]
    assert len(eligible) == 103 and len(approved) == 73
    for group, chosen in [('M2_all', eligible), ('M2_natural_approved', approved)]:
        for condition, key in [('D1', 'd1_id'), ('D1L', 'd1l_id')]:
            vals = {}
            for p in chosen:
                a, b = statuses[p['d0_id']]['need'], statuses[p[key]]['need']
                vals[p['d0_id']] = float(a == b == 'ok') if a in ['ok', 'err'] and b in ['ok', 'err'] else None
            out[group+'_'+condition+'_both_success'] = interval(vals, inputs, len(chosen))
    out['reference_metric_rows'] = 495
    out['inputs_sha256'] = sha(evaluation / 'generation_inputs.jsonl')
    out['generated_sha256'] = sha(evaluation / 'gen' / f'{args.label}.jsonl')
    dump(evaluation / f'{args.label}_RESULTS.json', out)
    print(json.dumps(out, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
