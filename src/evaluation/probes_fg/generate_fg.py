#!/usr/bin/env python3
"""Controlled Probe F & Probe G inference under the shared evaluation protocol.

Uses greedy decoding (max_new_tokens=768, non-thinking mode) across five
evaluated configurations (Base, Base+Strong, Joint-DPO, Stage-1 pi1, and MCARE)
on 200 paired probe inputs (Probe F: 100, Probe G: 100). Resumable by id.

usage: PYTHONPATH=src python src/evaluation/probes_fg/generate_fg.py --label {Base,Base+Strong,Joint-DPO,Stage1_pi1,MCARE}
"""
import argparse
import json
import os
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[2]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
from common.config import get_paths as _get_paths
from common.hf_worker import generate_batch as _hf_generate_batch, load as _hf_load
from common import prompts_and_schemas as C  # noqa: E402

_CFG = _get_paths()
EXP = _CFG.exp_dir
OUT = _CFG.exp_dir / 'probes_fg'

BASE_MODEL = _CFG.base_model
SPEC = {
    'Base': {'model': BASE_MODEL, 'adapter': None, 'strong': False},
    'Base+Strong': {'model': BASE_MODEL, 'adapter': None, 'strong': True},
    'Joint-DPO': {'model': str(_CFG.baselines_dir / 'merged/Joint-DPO'), 'adapter': None, 'strong': False},
    'Stage1_pi1': {'model': BASE_MODEL, 'adapter': os.environ.get('MCARE_STAGE1_ADAPTER', str(_CFG.stage1_dir / 'pi1_s42/final')), 'strong': False},
    'MCARE': {'model': BASE_MODEL, 'adapter': os.environ.get('MCARE_FINAL_ADAPTER', str(_CFG.stage2_dir / 'MCARE_s42/final')), 'strong': False},
}


def read_jsonl(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def flatten_inputs():
    items = []
    for exp, keys in (('F', ('base', 'evidence')), ('G', ('keep', 'update'))):
        parents = json.loads((OUT / f'{exp}_inputs.json').read_text())
        for p in parents:
            for k in keys:
                row = p[k]
                items.append({'id': row['id'], 'messages': row['messages'], 'exp': exp, 'version': k, 'parent_id': p['parent_id']})
    assert len({i['id'] for i in items}) == len(items)
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--label', required=True, choices=list(SPEC))
    ap.add_argument('--batch-size', type=int, default=8)
    args = ap.parse_args()
    spec = SPEC[args.label]

    model, tok = _hf_load(spec['model'])
    if spec['adapter']:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, spec['adapter'], is_trainable=False).eval()

    items = flatten_inputs()
    out_path = OUT / 'gen' / f'{args.label}.jsonl'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = {r['id']: r for r in read_jsonl(out_path)} if out_path.exists() else {}
    histories = [C.with_strong_prompt(it['messages']) if spec['strong'] else it['messages'] for it in items]

    for start in range(0, len(items), args.batch_size):
        batch = items[start:start + args.batch_size]
        if all(it['id'] in done for it in batch):
            continue
        out = _hf_generate_batch(model, tok, histories[start:start + args.batch_size], max_tokens=768)
        with out_path.open('a') as f:
            for it, g in zip(batch, out):
                if it['id'] in done:
                    continue
                if not g['text']:
                    raise ValueError('Empty generated reply for ' + it['id'])
                rec = {'id': it['id'], 'exp': it['exp'], 'version': it['version'], 'parent_id': it['parent_id'],
                       'response': g['text'], 'generated_tokens': g['tokens'], 'hit_cap': g['hit_cap'], 'finish_reason': g['finish_reason']}
                f.write(json.dumps(rec, ensure_ascii=False) + '\n')
                done[it['id']] = rec
        print(json.dumps({'label': args.label, 'generated': len(done), 'expected': len(items)}), flush=True)
    assert set(done) == {it['id'] for it in items}
    print('FINISHED', args.label, len(done), flush=True)


if __name__ == '__main__':
    main()
