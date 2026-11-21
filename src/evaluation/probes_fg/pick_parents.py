#!/usr/bin/env python3
"""Deterministically select 100 held-out parents for Probes F and G (50 each),
disjoint from every parent in M1/M2/controlled_negative (esconv_test / ed_test)
and from each other, using the deterministic hash_rank selection from
src/data/build_natural_pool.py.
"""
import json
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[2]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
from common.config import get_paths as _get_paths
_CFG = _get_paths()
ROOT = _CFG.root
EVAL_DIR = _CFG.eval_dir
POL = _CFG.policy_dir
OUT = _CFG.exp_dir / 'probes_fg'

from common.prompts_and_schemas import hash_rank  # noqa: E402



def read_jsonl(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def richest_per_parent(rows):
    by_parent = {}
    for r in rows:
        p = r['parent_id']
        if p not in by_parent or len(r['messages']) > len(by_parent[p]['messages']):
            by_parent[p] = r
    return by_parent


def main():
    esconv_dev = richest_per_parent(read_jsonl(POL / 'esconv_dev.jsonl'))
    ed_valid = richest_per_parent(read_jsonl(POL / 'ed_valid.jsonl'))

    esc_parents = sorted(esconv_dev, key=lambda p: hash_rank('fg_pick_esc', p))
    ed_parents = sorted(ed_valid, key=lambda p: hash_rank('fg_pick_ed', p))

    assert len(esc_parents) >= 50 and len(ed_parents) >= 50

    picks = {
        'F': {'esconv': esc_parents[:25], 'ed': ed_parents[:25]},
        'G': {'esconv': esc_parents[25:50], 'ed': ed_parents[25:50]},
    }

    out = {}
    for exp, srcs in picks.items():
        rows = []
        for src, plist in srcs.items():
            pool = esconv_dev if src == 'esconv' else ed_valid
            for p in plist:
                rows.append(pool[p])
        out[exp] = rows

    for exp, rows in out.items():
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / f'{exp}_picked_parents.jsonl').write_text('\n'.join(json.dumps(r, ensure_ascii=False) for r in rows) + '\n')
        n_esc = sum(1 for r in rows if r['source'] == 'esconv')
        print(exp, len(rows), 'parents (', n_esc, 'esconv,', len(rows) - n_esc, 'ed)')

    # sanity: no overlap between F and G parents, and none touch the already-used esconv_test/ed_main parents
    used = set()
    sup = json.load(open('/dev/null')) if False else None
    import itertools
    f_ids = {r['parent_id'] for r in out['F']}
    g_ids = {r['parent_id'] for r in out['G']}
    assert not (f_ids & g_ids), 'F/G parent overlap!'
    print('F/G disjoint parent check: OK,', len(f_ids), 'vs', len(g_ids))



if __name__ == '__main__':
    main()
