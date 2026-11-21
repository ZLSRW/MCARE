"""Prepare source histories, candidate caches, and counterfactual reversal pairs
for Stage-1 structured preference sets and empathy specialist (pi_E) training.
"""
import copy
import json
import re
from collections import Counter
from pathlib import Path
import sys

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
from common.config import get_paths as _get_paths
from common.io_utils import digest, dump_json as dump, key, load_json, read_jsonl as read, sha, write_jsonl as write

_CFG = _get_paths()
PROJECT = _CFG.root
ROOT = _CFG.specialist_dir
OLD = _CFG.baselines_dir / 'shared_pool'
POLICY = _CFG.policy_dir
OFFICIAL = Path(_CFG.base_model)


def old_key(s):
    import hashlib
    return hashlib.sha256(('42|' + s).encode()).hexdigest()

def main():
    data = ROOT / 'data'
    if (data / 'INPUTS_FROZEN.json').exists():
        f = json.loads((data / 'INPUTS_FROZEN.json').read_text())
        for name, sha in f['outputs'].items():
            assert digest(data / name) == sha, name
        print('Existing frozen inputs validated'); return
    train = read(POLICY/'esconv_train.jsonl') + read(POLICY/'ed_train.jsonl')
    byid = {r['id']: r for r in train}
    held = {(r['source'],r['parent_id']) for n in ('esconv_dev','ed_valid','esconv_test','ed_test') for r in read(POLICY/(n+'.jsonl'))}
    rows, cache, tasks, exclusions = [], [], [], []
    for old in read(OLD/'prompts.jsonl'):
        row = copy.deepcopy(old)
        original = byid[row['base_id']]
        assert (row['source'],row['parent_id']) not in held
        assert original['messages'][-1]['role'] == 'assistant'
        natural_history = original['messages'][:-1]
        # A real source containing the held-out diagnostic disclosure is not
        # rewritten. Exclude it and expose the resulting actual count.
        if re.search(r'autis', str(natural_history), re.I):
            exclusions.append({'id': row['id'], 'reason': 'excluded_by_domain_filter'}); continue
        row['source_messages'] = natural_history
        row['source_file'] = row['source'] + '_train.jsonl'
        row['source_record_id'] = original['id']
        row['derived_changes'] = []
        if row['kind'] == 'pragmatic':
            row['derived_changes'].append({'role':'user','appended':row['instruction']})
        if row.get('family') in ('mention','omit'):
            # These are newly DERIVED conditions, not original user identities.
            name = ['Maya','Jordan','Alex','Sam','Taylor','Casey'][int(key(row['id'])[:8],16)%6]
            if row['family'] == 'mention':
                ins = f'I am left-handed and I want my friend {name} to know this when I ask for support. Write only a message asking {name} to listen, explicitly mentioning that I am left-handed. Do not add commentary, placeholders, or a signature.'
            else:
                ins = f'I am left-handed, but I do not want to disclose this to my friend {name}. Write only a message asking {name} to listen and support me, without mentioning that I am left-handed. Do not add commentary, placeholders, or a signature.'
            row['messages'] = copy.deepcopy(natural_history)
            row['messages'][-1]['content'] += '\n\n' + ins
            row['instruction'] = ins
            row['derived_changes'] = [{'role':'user','appended':ins}]
            tasks.append({'id':row['id'],'messages':row['messages'],'n':8,'greedy':True})
        else:
            p = OLD/'candidates'/(old_key(old['id'])+'.json')
            c = json.loads(p.read_text())
            assert c['messages'] == row['messages']
            assert len(c['candidates']) == 8
            assert all(x['text'].strip() for x in c['candidates'])
            cache.append({'id':row['id'],'messages':row['messages'], 'candidates':c['candidates'], 'greedy':c['greedy'], 'source_path':str(p),'source_sha256':digest(p),'reuse_model':'official Qwen3.5-9B','reuse_sampling':'K8 T1 top_p0.95 max512 nonthinking; original seeds preserved'})
        row['prompt_sha256'] = key(row['messages'])
        rows.append(row)
    assert len({(r['source'],r['parent_id']) for r in rows}) == len(rows)
    assert not any(re.search('autis',str(r['messages']),re.I) for r in rows)
    # Select before inspecting any newly generated replies. Same parents are
    # allowed across mechanisms; no additional independent sample claim.
    possible = [r for r in rows if r['kind']=='natural' and len(r['messages'][-1]['content'].split())>=12]
    possible.sort(key=lambda r:key(['reversal',r['id']]))
    reversals=[]
    for r in possible[:400]:
        l='For this reply, please listen and acknowledge my feelings. Do not give advice or ask any questions.'
        a='For this reply, please acknowledge my feelings and give one specific, feasible next step I can take.'
        variants={}
        for name,ins in [('listen',l),('action',a),('background',l+'\nFor context, I am left-handed.')]:
            m=copy.deepcopy(r['messages']);m[-1]['content']+='\n\n'+ins;variants[name]=m
        rr={'id':r['id']+'::reversal','source':r['source'],'parent_id':r['parent_id'],'base_id':r['id'],'variants':variants,'source_prompt_sha256':r['prompt_sha256'],'changes':{'listen':l,'action':a,'background':'For context, I am left-handed.'}}
        reversals.append(rr)
        for v in ('listen','action'):
            tasks.append({'id':rr['id']+'::'+v,'messages':variants[v],'n':2,'greedy':False})
    dev=[]; seen=set()
    for r in sorted(read(POLICY/'esconv_dev.jsonl')+read(POLICY/'ed_valid.jsonl'),key=lambda r:key(['dev',r['id']])):
        par=(r['source'],r['parent_id'])
        if par in seen or re.search('autis',str(r['messages']),re.I):continue
        seen.add(par)
        dev.append({'id':r['id'],'source':r['source'],'parent_id':r['parent_id'],'kind':'dev','messages':r['messages'][:-1]})
        if len(dev)==64:break
    for r in dev:tasks.append({'id':'dev::'+r['id'],'messages':r['messages'],'n':8,'greedy':True})
    write(data/'train_contexts.jsonl',rows);write(data/'cached_candidates.jsonl',cache)
    write(data/'generation_tasks.jsonl',tasks);write(data/'reversals.jsonl',reversals);write(data/'dev_contexts.jsonl',dev)
    write(data/'input_exclusions.jsonl',exclusions)
    outputs={n:digest(data/n) for n in ('train_contexts.jsonl','cached_candidates.jsonl','generation_tasks.jsonl','reversals.jsonl','dev_contexts.jsonl','input_exclusions.jsonl')}
    base_hashes={p.name:digest(p) for p in OFFICIAL.iterdir() if p.is_file() and (p.suffix=='.safetensors' or p.name in ('config.json','tokenizer_config.json','chat_template.jinja'))} if OFFICIAL.exists() else {}
    f={'start_model':_CFG.base_model,'base_files':base_hashes,'contexts':len(rows),'parents':len(rows),'kinds':dict(Counter(r['kind'] for r in rows)),'cached_prompts':len(cache),'new_generation_tasks':len(tasks),'reversal_groups_requested':len(reversals),'training_annotator':_CFG.train_annotator_model,'source_hashes':{n:digest(POLICY/(n+'.jsonl')) for n in ('esconv_train','ed_train','esconv_dev','ed_valid','esconv_test','ed_test')},'outputs':outputs,'split_parent_overlap':0,'no_hidden_situation':True,'autism_training_disclosure':False}
    dump(data/'INPUTS_FROZEN.json',f); print(json.dumps(f,indent=2))

if __name__=='__main__':main()
