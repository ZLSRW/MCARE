#!/usr/bin/env python3
"""Score M2 D0/D1/D1L evaluation replies for empathy via test-time judge logprobs.

Evaluates the 13 baseline models, MCARE, and the 6 Table-VI ablation/mechanism
variants on m2_d0/m2_d1/m2_d1l using EMP_SYSTEM/EMP_USER and 1-5 digit expected
empathy. Resumable: already-scored (label, id) pairs are skipped on restart.
"""
import json
import os
import sys
import threading
import time
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
from common.config import get_paths as _get_paths
_CFG = _get_paths()
ROOT = _CFG.root
EVAL_DIR = _CFG.eval_dir
OUT = _CFG.exp_dir / 'analysis'

JUDGE_URL = _CFG.eval_judge_url + '/chat/completions'

from common.prompts_and_schemas import EMP_SYSTEM, EMP_USER, render_dialogue, digit_expectation, prompt_hash  # noqa: E402

PROMPT_HASH = prompt_hash('emp')

MODELS = {
    'Base': EVAL_DIR / 'gen/Base.jsonl',
    'Base+Strong': EVAL_DIR / 'gen/Base_Strong.jsonl',
    'RLFF': EVAL_DIR / 'gen/RLFF.jsonl',
    'PERM': EVAL_DIR / 'gen/PERM.jsonl',
    'PEER-RUR': EVAL_DIR / 'gen/PEER_RUR.jsonl',
    'PEER-SFT': EVAL_DIR / 'gen/PEER_SFT.jsonl',
    'ESCRAG': EVAL_DIR / 'gen/ESCRAG.jsonl',
    'RLVER': EVAL_DIR / 'gen/RLVER.jsonl',
    'Vanilla-SFT': EVAL_DIR / 'gen/Vanilla_SFT.jsonl',
    'Quality-DPO': EVAL_DIR / 'gen/Quality_DPO.jsonl',
    'Support-DPO': EVAL_DIR / 'gen/Support_DPO.jsonl',
    'Joint-DPO': EVAL_DIR / 'gen/Joint_DPO.jsonl',
    'SafeDPO': EVAL_DIR / 'gen/SafeDPO.jsonl',
    'MCARE': EVAL_DIR / 'runs/MCARE/gen/MCARE.jsonl',
    'Stage1_pi1': EVAL_DIR / 'runs/Stage1_pi1/gen/Stage1_pi1.jsonl',
    'Specialist_piE': EVAL_DIR / 'runs/Specialist_piE/gen/Specialist_piE.jsonl',
    'Transfer_init_piE': EVAL_DIR / 'runs/Transfer_init_piE/gen/Transfer_init_piE.jsonl',
    'Interp_075_025': EVAL_DIR / 'runs/Interp_075_025/gen/Interp_075_025.jsonl',
    'SelfSampled_pi1': EVAL_DIR / 'runs/SelfSampled_pi1/gen/SelfSampled_pi1.jsonl',
    'SelfSampled_NoGate': EVAL_DIR / 'runs/SelfSampled_NoGate/gen/SelfSampled_NoGate.jsonl',
}
COND_OF_SET = {'m2_d0': 'D0', 'm2_d1': 'D1', 'm2_d1l': 'D1L'}


def read_jsonl(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def build_items():
    OUT.mkdir(parents=True, exist_ok=True)
    inputs = {r['id']: r for r in read_jsonl(EVAL_DIR / 'generation_inputs.jsonl')}
    items = []
    for label, gen_path in MODELS.items():
        gen = {r['id']: r['response'] for r in read_jsonl(gen_path)}
        n_found = 0
        for iid, row in inputs.items():
            cond = next((COND_OF_SET[s] for s in row['sets'] if s in COND_OF_SET), None)
            if cond is None:
                continue
            reply = gen.get(iid)
            if reply is None:
                continue
            n_found += 1
            items.append({'label': label, 'id': iid, 'parent_id': row['parent_id'], 'condition': cond,
                          'messages': row['messages'], 'reply': reply})
        expect = 120
        if n_found != 3 * expect:
            print(f'WARNING {label}: found {n_found} M2 rows, expected {3 * expect}', file=sys.stderr)
    return items


def call_judge(item, timeout=60, retries=3):
    body = json.dumps({
        'model': _CFG.judge_served_name,
        'messages': [{'role': 'system', 'content': EMP_SYSTEM},
                     {'role': 'user', 'content': EMP_USER.format(
                         dialogue=render_dialogue(item['messages'], mark_latest=False),
                         reply=item['reply'].strip() or '(empty reply)')}],
        'temperature': 0, 'max_tokens': 1, 'logprobs': True, 'top_logprobs': 20,
        'chat_template_kwargs': {'enable_thinking': False},
    }).encode()
    last_err = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(JUDGE_URL, data=body, headers={'Content-Type': 'application/json'})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                r = json.loads(resp.read())
            top = {e['token']: e['logprob'] for e in r['choices'][0]['logprobs']['content'][0]['top_logprobs']}
            score, mass = digit_expectation(top)
            return {'label': item['label'], 'id': item['id'], 'parent_id': item['parent_id'],
                    'condition': item['condition'], 'empathy': score, 'digit_mass': mass,
                    'prompt_hash': PROMPT_HASH}
        except Exception as e:
            last_err = e
            time.sleep(2 * (attempt + 1))
    return {'label': item['label'], 'id': item['id'], 'parent_id': item['parent_id'],
            'condition': item['condition'], 'empathy': None, 'digit_mass': None,
            'prompt_hash': PROMPT_HASH, 'error': repr(last_err)}


def main():
    out_path = OUT / 'm2_empathy_labels_g.jsonl'
    done = set()
    if out_path.exists():
        for r in read_jsonl(out_path):
            done.add((r['label'], r['id']))
    items = [it for it in build_items() if (it['label'], it['id']) not in done]
    print(f'todo: {len(items)} (already done: {len(done)})', flush=True)
    if not items:
        return
    lock = threading.Lock()
    n_done = 0
    n_err = 0
    t0 = time.time()
    with open(out_path, 'a') as fh, ThreadPoolExecutor(max_workers=48) as pool:
        futs = {pool.submit(call_judge, it): it for it in items}
        for fut in as_completed(futs):
            row = fut.result()
            with lock:
                fh.write(json.dumps(row, ensure_ascii=False) + '\n')
                fh.flush()
                n_done += 1
                if row.get('empathy') is None:
                    n_err += 1
                if n_done % 200 == 0 or n_done == len(items):
                    el = time.time() - t0
                    print(f'{n_done}/{len(items)} done, {n_err} errors, {el:.0f}s elapsed, {el/n_done*1000:.0f}ms/item avg', flush=True)
    print('FINISHED', flush=True)


if __name__ == '__main__':
    main()
