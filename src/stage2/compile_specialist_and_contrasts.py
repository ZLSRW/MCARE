"""Compile audited training annotations into structured Stage-1 sets (P_rev, P_need,
P_warm, P_req, P_acc) and the Empathy Specialist (pi_E) distillation set.

Implements Eqs. (12)-(16) (Section III-B3) and Section III-C.
"""
import difflib
import itertools
import json
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
PROTOCOL_HASH = None  # populated from EXECUTION_PROTOCOL.json when present


def segments(a,b,which):
    s=[]
    for op,i,j,k,l in difflib.SequenceMatcher(None,a,b,autojunk=False).get_opcodes():
        if op!='equal':s.append([i,j] if which=='old' else [k,l])
    return [x for x in s if x[1]>x[0]]

def pair(r,a,b,**extra):
    return {'id':r['id']+':'+str(a)+':'+str(b),'source':r['source'],'parent_id':r['parent_id'],
        'views':[{'messages':r['messages'],'responses':[r['candidates'][a]['text'],r['candidates'][b]['text']]}],**extra}

def length_match(a,b):
    x,y=len(a.split()),len(b.split())
    return abs(x-y)/max(1,x,y)<=.15

def evidence_view(messages,a):
    import copy
    m=copy.deepcopy(messages)
    # Exact original evidence, not a newly inferred user instruction.
    m[0]['content']+='\nTraining-only highlight of already visible user evidence (the original conversation is authoritative):\n'+'\n'.join('QUOTE: '+e['quote'] for e in a['evidence'])
    return m

def main():
    assert (ROOT/'data/TRAIN_ANNOTATIONS_COMPLETE.json').exists(), 'Annotations incomplete'
    proto_path = ROOT/'EXECUTION_PROTOCOL.json'
    protocol=json.loads(proto_path.read_text()) if proto_path.exists() else {}
    max_excl_frac=protocol.get('data',{}).get('annotation_exclusion_max_fraction',0.05)
    expected=read(ROOT/'data/train_contexts.jsonl')+[{**r,'id':'dev::'+r['id']} for r in read(ROOT/'data/dev_contexts.jsonl')]
    expected={r['id']:r for r in expected}
    actual={json.loads(p.read_text())['id']:p for p in (ROOT/'data/annotations').glob('*.json')}
    exclusions=[json.loads(p.read_text()) for p in (ROOT/'data/annotation_exclusions').glob('*.json')]
    excluded_contexts={x['id']:x for x in exclusions if x['kind']=='context'}
    assert not set(actual)&set(excluded_contexts),'A training context cannot be both validated and excluded'
    assert set(actual)|set(excluded_contexts)==set(expected),'Annotation processing coverage differs from frozen contexts'
    for i,x in excluded_contexts.items():
        assert x['input_hash']==key(expected[i]) and x['protocol_hash']==PROTOCOL_HASH
        assert x['used_for_training'] is False and x['failure']['validated_attempts']==4
    candidate_sources={r['id']:r for r in read(ROOT/'data/cached_candidates.jsonl')}
    candidate_sources.update({r['id']:r for p in (ROOT/'data/generated').glob('*.json') for r in [json.loads(p.read_text())]})
    reversal_sources={r['id']:r for r in read(ROOT/'data/reversals.jsonl')}
    reversal_files=list((ROOT/'data/reversal_annotations').glob('*.json'))
    excluded_reversals={x['id']:x for x in exclusions if x['kind']=='reversal'}
    actual_reversals={json.loads(p.read_text())['id'] for p in reversal_files}
    assert not actual_reversals&set(excluded_reversals)
    assert actual_reversals|set(excluded_reversals)==set(reversal_sources)
    for i,x in excluded_reversals.items():
        assert x['input_hash']==key(reversal_sources[i]) and x['protocol_hash']==PROTOCOL_HASH
        assert x['used_for_training'] is False and x['failure']['validated_attempts']==4
    assert len(exclusions)<=max_excl_frac*(len(expected)+len(reversal_sources))
    objectives={k:[] for k in ('P_rev','P_need','P_warm','P_req','P_acc','specialist_piE')}
    skips=[];dev=[]
    for p in sorted((ROOT/'data/annotations').glob('*.json')):
        r=json.loads(p.read_text());labs=r['labels'];cs=r['candidates']
        assert r['protocol_hash']==PROTOCOL_HASH
        if not r['audit']['request_applicable']:
            assert all(not label['goal_spans'] for label in labs),'Inapplicable completion spans cannot enter training'
        assert r['input_hash']==key(expected[r['id']])
        source=candidate_sources[r['id']]
        assert r['candidate_hash']==key(source) and cs==source['candidates']+[source['greedy']]
        # Exclude greedy response to keep candidate set uniform (K=8).
        indices=list(range(len(cs)-1))
        good=[i for i in indices if labs[i]['correct'] and labs[i]['quality']>=3 and labs[i]['warmth']>=3]
        bad=[i for i in indices if not labs[i]['correct']]
        not_ok=[i for i in indices if i not in good]
        if r['id'].startswith('dev::'):
            if good and bad and r['audit']['evidence']:
                v=pair(r,max(good,key=lambda i:labs[i]['quality']),max(bad,key=lambda i:labs[i]['quality']))
                v['teacher_messages']=evidence_view(r['messages'],r['audit']);v['responses_all']=[c['text'] for c in cs[:-1]]
                v['acceptable']=good;dev.append(v)
            continue
        if good:
            plus=max(good,key=lambda i:(labs[i]['quality'],labs[i]['warmth'],-i))
            neg=max(bad,key=lambda i:(labs[i]['quality'],labs[i]['warmth'],-i)) if bad else min(indices,key=lambda i:(labs[i]['quality'],labs[i]['warmth'],i))
            if plus!=neg:
                base=pair(r,plus,neg)
                if r['audit']['evidence']:
                    objectives['specialist_piE'].append({**base,'teacher_messages':evidence_view(r['messages'],r['audit']),
                       'responses_all':[c['text'] for c in cs[:-1]],'acceptable':good})
                goal=labs[plus]['goal_spans']
                if r['audit']['request_applicable'] and goal:
                    objectives['P_req'].append({**base,'goal_spans':[[cs[plus]['text'].index(s),cs[plus]['text'].index(s)+len(s)] for s in goal]})
        if len(good)>=2 and not_ok:
            top2=sorted(good,key=lambda i:(labs[i]['quality'],labs[i]['warmth'],-i),reverse=True)[:2]
            for a in top2:
                matched_b=[b for b in not_ok if length_match(cs[a]['text'],cs[b]['text'])]
                if matched_b:
                    b=max(matched_b,key=lambda i:(labs[i]['quality'],-i))
                    objectives['P_acc'].append(pair(r,a,b,pair_axis='acceptable'))
        # Same need/length/quality/warmth matched pairs (Eqs. 13-15).
        matched=[]
        for a,b in itertools.product(good,bad):
            item=pair(r,a,b,pair_axis='need')
            rank=(labs[b]['quality'],labs[a]['quality'],-a,-b)
            if length_match(cs[a]['text'],cs[b]['text']) and abs(labs[a]['warmth']-labs[b]['warmth'])<=.5 and abs(labs[a]['quality']-labs[b]['quality'])<=.5:
                matched.append((rank,item))
        if matched:
            objectives['P_need'].append(max(matched,key=lambda x:x[0])[1])
        else:
            skips.append({'id':r['id'],'objective':'P_need','reason':'No semantic need pair matching length, warmth and quality'})
        wp=[]
        for a,b in itertools.permutations(good,2):
            if labs[a]['warmth']-labs[b]['warmth']>=1 and abs(labs[a]['quality']-labs[b]['quality'])<=.5 and length_match(cs[a]['text'],cs[b]['text']):
                wp.append(((labs[a]['warmth']-labs[b]['warmth'],-a,-b),pair(r,a,b,pair_axis='warmth')))
        if wp:
            objectives['P_warm'].append(max(wp,key=lambda x:x[0])[1])
    # Same candidate replies across listen vs. action views (Eq. 12).
    for p in sorted(reversal_files):
        r=json.loads(p.read_text());c=r['candidates'];g=r['labels']
        assert r['protocol_hash']==PROTOCOL_HASH
        assert r['variants']==reversal_sources[r['id']]['variants']
        assert c==candidate_sources[r['id']+'::listen']['candidates']+candidate_sources[r['id']+'::action']['candidates']
        if not (r['review'].get('accepted') is True and r['review'].get('background_irrelevant') is True):
            skips.append({'id':r['id'],'objective':'P_rev','reason':'Derived context review failed','review':r['review']});continue
        pairs=[]
        for i,j in itertools.permutations(range(len(c)),2):
            if not (g['listen'][i]['correct'] and not g['listen'][j]['correct'] and g['action'][j]['correct'] and not g['action'][i]['correct'] and g['background'][i]['correct'] and not g['background'][j]['correct']):continue
            # Check quality floor in the view preferring the reply (the alternate view reflects the need mismatch in quality); check warmth in both.
            if g['listen'][i]['quality']<3 or g['action'][j]['quality']<3:continue
            if any(g[v][z]['warmth']<3 for v in ('listen','action') for z in (i,j)):continue
            if abs(g['listen'][i]['warmth']-g['action'][j]['warmth'])>.5:continue
            pairs.append((g['listen'][i]['quality']+g['action'][j]['quality'],i,j))
        if not pairs:
            skips.append({'id':r['id'],'objective':'P_rev','reason':'No same replies pass preference reversal and matching'});continue
        _,i,j=max(pairs)
        objectives['P_rev'].append({'id':r['id'],'source':r['source'],'parent_id':r['parent_id'],'views':[
          {'messages':r['variants']['listen'],'responses':[c[i]['text'],c[j]['text']]},
          {'messages':r['variants']['action'],'responses':[c[j]['text'],c[i]['text']]}]})
    for name,rows in objectives.items():
        assert rows, f'Objective {name} has no qualified data'
        write(ROOT/'data/objectives'/f'{name}.jsonl',rows)
    write(ROOT/'data/distillation_dev.jsonl',dev);write(ROOT/'data/objective_exclusions.jsonl',skips)
    lengths={}
    for p in (ROOT/'data/objectives').glob('*.jsonl'):
        rows=read(p);stats=[]
        for view in range(max(len(r['views']) for r in rows)):
            pairs=[r['views'][view]['responses'][:2] for r in rows if len(r['views'])>view and len(r['views'][view]['responses'])==2]
            if pairs:
                a=[len(x[0].split()) for x in pairs];b=[len(x[1].split()) for x in pairs]
                stats.append({'view':view,'pairs':len(pairs),'chosen_mean_words':sum(a)/len(a),
                    'rejected_mean_words':sum(b)/len(b),'chosen_longer_fraction':sum(x>y for x,y in zip(a,b))/len(a)})
        lengths[p.stem]=stats
    dump(ROOT/'data/PAIR_LENGTH_STATISTICS.json',lengths)
    report={'objective_counts':{k:len(v) for k,v in objectives.items()},
      'distillation_dev_count':len(dev),'exclusion_counts':dict(Counter(str(r['objective']) for r in skips)),
      'objective_hashes':{p.name:digest(p) for p in (ROOT/'data/objectives').glob('*.jsonl')},
      'annotation_protocol_hash':PROTOCOL_HASH,'inputs_frozen_sha256':digest(ROOT/'data/INPUTS_FROZEN.json'),
      'compiler_sha256':digest(Path(__file__).resolve()),
      'unresolved_annotation_exclusions':len(exclusions),
      'unresolved_context_exclusions':len(excluded_contexts),
      'unresolved_reversal_exclusions':len(excluded_reversals),
      'training_annotation_coverage_by_kind':{
          kind:{'frozen':sum(r.get('kind')==kind and not i.startswith('dev::') for i,r in expected.items()),
                'validated':sum(expected[i].get('kind')==kind and not i.startswith('dev::') for i in actual),
                'excluded':sum(expected[i].get('kind')==kind and not i.startswith('dev::') for i in excluded_contexts)}
          for kind in sorted({r['kind'] for i,r in expected.items() if not i.startswith('dev::')})},
      'unresolved_annotation_exclusion_ids':[{'id':x['id'],'kind':x['kind'],'stage':x['failure']['stage']} for x in exclusions],
      'completion_span_policy':'Only independently audited explicit requests have completion targets; inapplicable proxy proposals are retained as unused metadata.'}
    dump(ROOT/'data/OBJECTIVES_FROZEN.json',report);print(json.dumps(report,indent=2))

if __name__=='__main__':main()
