"""History-only feedback audit, deterministic train-only selection and cache reuse."""
import argparse
import collections
import copy
import json
import re
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
from baselines.matched_pool_common import *

PRESCREEN = re.compile(
    r"(?:already|previously|before).{0,45}(?:tried|done)|"
    r"(?:tried|trying).{0,80}(?:not|didn't|doesn't|wasn't|no |help|work)|"
    r"(?:that|this|those|your advice|you).{0,45}(?:won't|can't|doesn't|didn't|don't|not |never|useless|unhelpful)|"
    r"(?:stop|don't|do not|rather not|no more).{0,35}(?:asking|ask|talk|giv|suggest|advice|discuss)|"
    r"(?:not |never ).{0,25}(?:help|work|want|need|agree|understand)|"
    r"(?:can't|cannot|unable|impossible).{0,45}(?:do|try|afford|find|access|go|talk)|"
    r"(?:thanks|thank you|sorry|no),? (?:but|though)", re.I)


def train():
    return read(POLICY / 'esconv_train.jsonl') + read(POLICY / 'ed_train.jsonl')


def prepare():
    from transformers import AutoTokenizer
    rows = train()
    held = {(r['source'], r['parent_id']) for name in
            ['esconv_dev', 'esconv_test', 'ed_valid', 'ed_test']
            for r in read(POLICY / f'{name}.jsonl')}
    assert not {(r['source'], r['parent_id']) for r in rows} & held
    tok = AutoTokenizer.from_pretrained(_CFG.base_model, local_files_only=True)
    eligible, candidates = [], []
    for r in rows:
        assert r['messages'][-1]['role'] == 'assistant' and r['messages'][-2]['role'] == 'user'
        messages = r['messages'][:-1]
        if re.search(r'autis', str(messages), re.I):
            continue
        ids = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, enable_thinking=False)
        if hasattr(ids, 'keys'):
            ids = ids['input_ids']
        # Reserve 512 reply tokens AND room for the added pragmatic instruction.
        if len(ids) > 3300:
            continue
        record = {k: r[k] for k in ['id', 'source', 'parent_id', 'target']}
        record.update(messages=messages, prompt_tokens=len(ids))
        eligible.append(record)
        if len(messages) >= 4 and PRESCREEN.search(messages[-1]['content']):
            candidates.append(record)
    candidates.sort(key=lambda r: rank('feedback-audit', r['id']))
    write(OUT / 'eligible_train.jsonl', eligible)
    write(OUT / 'feedback_candidates.jsonl', candidates)
    protocol = {'seed': 42, 'requested_counts': {'natural': 1200, 'feedback_max': 500, 'pragmatic': 300},
        'shortfall_policy': 'Backfill unfilled feedback slots with natural train histories, disclose actual counts.',
        'one_prefix_per_parent': True, 'hidden_situation_inputs': False,
        'feedback_prescreen_regex': PRESCREEN.pattern, 'feedback_prescreen_is_not_gold': True,
        'feedback_candidates': len(candidates), 'eligible_train_prefixes': len(eligible),
        'audit_prompt': AUDIT_PROMPT, 'review_prompt': REVIEW_PROMPT,
        'audit_model': _CFG.train_annotator_model, 'review_is_same_model_not_human': True,
        'core_families': {f: 60 for f in ['listen', 'draft', 'update', 'mention', 'omit']},
        'no_word_caps_or_twosent': True, 'split_parent_overlap': 0,
        'source_hashes': {name: sha(POLICY / f'{name}.jsonl') for name in
                         ['esconv_train', 'ed_train', 'esconv_dev', 'esconv_test', 'ed_valid', 'ed_test']},
        'Joint-DPO': 'chosen=max(zQ+zE), rejected=min(zQ+zE); no rule filtering or pair-length restrictions',
        'SafeDPO': 'Transform the SAME Joint-DPO initial pair using the shared pragmatic checks: keep/swap/drop; prior margin 10 only on compliant/violating pairs.',
        'training': {'seed': 42, 'epochs': 2, 'beta': .1, 'learning_rate': 5e-5,
                     'effective_batch': 16, 'lora_rank': 16, 'lora_alpha': 32}}
    path = OUT / 'PROTOCOL.json'
    if path.exists():
        assert json.loads(path.read_text()) == protocol, 'Do not change a frozen protocol on restart'
    else:
        dump(path, protocol)
    print('prepared', len(eligible), 'train prefixes;', len(candidates), 'feedback candidates', flush=True)


def audit():
    path = OUT / 'feedback_audit.jsonl'
    done = {r['id'] for r in read(path)}
    todo = [r for r in read(OUT / 'feedback_candidates.jsonl') if r['id'] not in done]
    for start in range(0, len(todo), 32):
        batch = todo[start:start+32]
        convs = [[{'role': 'user', 'content': AUDIT_PROMPT.format(dialogue=dialogue(r['messages']))}] for r in batch]
        raw = chat(convs, AUDIT_SCHEMA)
        preliminary = []
        for r, t in zip(batch, raw):
            a = json.loads(t)
            latest = r['messages'][-1]['content']
            assistants = [m['content'] for m in r['messages'] if m['role'] == 'assistant']
            valid = (a['category'] != 'none' and bool(a['user_evidence'].strip()) and
                     a['user_evidence'] in latest and bool(a['target'].strip()) and
                     bool(a['narrow_response_requirement'].strip()) and
                     (not a['prior_assistant_evidence'] or any(a['prior_assistant_evidence'] in t for t in assistants)))
            preliminary.append((r, t, a, valid))
        reviews = chat([[{'role': 'user', 'content': REVIEW_PROMPT.format(
                       dialogue=dialogue(r['messages']), annotation=json.dumps(a))}]
                        for r, t, a, valid in preliminary if valid], REVIEW_SCHEMA, 400) if any(x[3] for x in preliminary) else []
        it = iter(reviews)
        with path.open('a') as handle:
            for r, t, a, valid in preliminary:
                review = json.loads(next(it)) if valid else {'accepted': False, 'reason': 'none or failed verbatim/specific-target checks'}
                result = {'id': r['id'], 'source': r['source'], 'parent_id': r['parent_id'],
                          'input_hash': message_hash(r['messages']), 'annotation': a,
                          'verbatim_valid': valid, 'review': review,
                          'accepted': bool(valid and review['accepted']), 'raw': t}
                handle.write(json.dumps(result, ensure_ascii=False) + '\n')
            handle.flush()
        print('audited feedback', min(start+32, len(todo)), '/', len(todo), flush=True)


def core_instruction(family, i):
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    old, new = days[i % 7], days[(i+2) % 7]
    name = ['Maya', 'Jordan', 'Alex', 'Sam', 'Taylor', 'Casey'][i % 6]
    if family == 'listen':
        text = 'For this reply, please just listen and acknowledge what I am feeling. Do not give advice or suggestions, and do not ask me a question.'
    elif family == 'draft':
        text = f'Please help me contact my friend {name} for support. I cannot meet on {old} evening, but I can talk on {new} morning. Write only the message I can send, proposing a call on {new} morning; do not include an introduction, commentary, alternatives, placeholders, or a signature.'
    elif family == 'update':
        text = f'Here is a correction for the message to my friend {name}: I originally said {old} evening, but that time is no longer possible. My updated availability is {new} afternoon. Write only a message proposing a call at the updated time and making clear the old time will not work. Do not add commentary, placeholders, or a signature.'
    elif family == 'mention':
        text = f'I am left-handed and I want my friend {name} to know this when I ask for support. Write only a message asking {name} to listen, explicitly mentioning that I am left-handed. Do not add commentary, placeholders, or a signature.'
    else:
        text = f'I am left-handed, but I do not want to disclose this to my friend {name}. Write only a message asking {name} to listen and support me, without mentioning that I am left-handed. Do not add commentary, placeholders, or a signature.'
    return text


def freeze():
    assert not (OUT / 'FROZEN.json').exists(), 'Already frozen; validate/reuse instead of rebuilding'
    rows = read(OUT / 'eligible_train.jsonl'); byid = {r['id']: r for r in rows}
    audited = read(OUT / 'feedback_audit.jsonl')
    assert {r['id'] for r in audited} == {r['id'] for r in read(OUT / 'feedback_candidates.jsonl')}
    used, selected, seen_prompts = set(), [], set()
    def add(r, kind, extra=None):
        parent = (r['source'], r['parent_id'])
        assert parent not in used
        item = {k: copy.deepcopy(r[k]) for k in ['source', 'parent_id', 'messages']}
        item.update(id=f"mixed:{kind}:{r['id']}", base_id=r['id'], kind=kind,
                    reference=r['target'] if kind != 'pragmatic' else None)
        if extra:
            item.update(extra)
        if kind == 'pragmatic':
            item['messages'][-1]['content'] = item['messages'][-1]['content'].rstrip() + '\n\n' + item['instruction']
        item['prompt_sha256'] = message_hash(item['messages'])
        if item['prompt_sha256'] in seen_prompts:
            return False
        used.add(parent)
        seen_prompts.add(item['prompt_sha256'])
        selected.append(item)
        return True
    accepted = sorted([a for a in audited if a['accepted']], key=lambda r: rank('feedback-select', r['id']))
    for a in accepted:
        r = byid[a['id']]
        if (r['source'], r['parent_id']) not in used:
            add(r, 'feedback', {'feedback': a['annotation']})
        if len(selected) == 500:
            break
    nfeedback = len(selected)
    byparent = collections.defaultdict(list)
    for r in rows:
        byparent[(r['source'], r['parent_id'])].append(r)
    representatives = [min(rs, key=lambda r: rank('prefix', r['id'])) for rs in byparent.values()]
    core = []
    for source in ('esconv', 'ed'):
        candidates = sorted([r for r in representatives if r['source'] == source and (r['source'], r['parent_id']) not in used], key=lambda r: rank('core', r['id']))
        core.extend(candidates[:150])
    assert len(core) == 300
    families = ['listen', 'draft', 'update', 'mention', 'omit']
    for i, r in enumerate(core):
        family = families[i % 5]; instruction = core_instruction(family, i)
        assert add(r, 'pragmatic', {'family': family, 'instruction': instruction})
    natural_target = 1700 - nfeedback
    available = {source: sorted([r for r in representatives if r['source'] == source and (r['source'], r['parent_id']) not in used], key=lambda r: rank('natural', r['id'])) for source in ('esconv', 'ed')}
    # Distinct parents may share a greeting-only visible history. In the same
    # fixed parent order, use that parent's next deterministic prefix when
    # needed, without changing selection based on generated model replies.
    natural_added = 0
    for source, limit in [('esconv', natural_target // 2), ('ed', natural_target)]:
        added_source = 0
        for representative in available[source]:
            if added_source >= limit or natural_added >= natural_target:
                break
            parent = (representative['source'], representative['parent_id'])
            for r in sorted(byparent[parent], key=lambda r: rank('prefix', r['id'])):
                if add(r, 'natural'):
                    natural_added += 1; added_source += 1
                    break
    assert natural_added == natural_target
    selected.sort(key=lambda r: rank('pool-order', r['id']))
    assert len(selected) == len(used) == 2000
    write(OUT / 'prompts.jsonl', selected)
    report = {'rows': 2000, 'parents': 2000, 'counts': dict(collections.Counter(r['kind'] for r in selected)),
              'source_counts': dict(collections.Counter(r['source'] for r in selected)),
              'feedback_categories': dict(collections.Counter(r['feedback']['category'] for r in selected if r['kind'] == 'feedback')),
              'feedback_backfill_natural': 500-nfeedback, 'feedback_prescreen_rows': len(audited),
              'feedback_accepted_prefixes': sum(a['accepted'] for a in audited),
              'core_families': dict(collections.Counter(r['family'] for r in selected if r['kind'] == 'pragmatic')),
              'prompts_sha256': sha(OUT / 'prompts.jsonl'), 'protocol_sha256': sha(OUT / 'PROTOCOL.json'),
              'audit_sha256': sha(OUT / 'feedback_audit.jsonl'), 'selection_uses_generated_replies': False,
              'source_hashes': json.loads((OUT / 'PROTOCOL.json').read_text())['source_hashes']}
    dump(OUT / 'FROZEN.json', report)
    validate()
    print(json.dumps(report, ensure_ascii=False), flush=True)


def validate():
    from transformers import AutoTokenizer
    frozen = json.loads((OUT / 'FROZEN.json').read_text())
    assert sha(OUT / 'prompts.jsonl') == frozen['prompts_sha256']
    assert sha(OUT / 'PROTOCOL.json') == frozen['protocol_sha256']
    assert sha(OUT / 'feedback_audit.jsonl') == frozen['audit_sha256']
    for name, digest in frozen['source_hashes'].items():
        assert sha(POLICY / f'{name}.jsonl') == digest
    byid = {r['id']: r for r in train()}
    held = {(r['source'], r['parent_id']) for name in ['esconv_dev', 'ed_valid', 'esconv_test', 'ed_test'] for r in read(POLICY / f'{name}.jsonl')}
    rows = read(OUT / 'prompts.jsonl')
    tok = AutoTokenizer.from_pretrained(_CFG.base_model, local_files_only=True)
    seen = set(); maximum = 0
    for r in rows:
        original = byid[r['base_id']]
        assert (r['source'], r['parent_id']) == (original['source'], original['parent_id'])
        assert (r['source'], r['parent_id']) not in held
        expected = copy.deepcopy(original['messages'][:-1])
        if r['kind'] == 'pragmatic':
            expected[-1]['content'] = expected[-1]['content'].rstrip() + '\n\n' + r['instruction']
        assert r['messages'] == expected
        assert r['prompt_sha256'] == message_hash(r['messages']) not in seen
        seen.add(r['prompt_sha256'])
        if r['kind'] == 'feedback':
            assert r['feedback']['user_evidence'] in r['messages'][-1]['content']
        ids = tok.apply_chat_template(r['messages'], tokenize=True, add_generation_prompt=True, enable_thinking=False)
        if hasattr(ids, 'keys'):
            ids = ids['input_ids']
        maximum = max(maximum, len(ids)); assert len(ids)+512 <= 4096
    assert len(rows) == len({(r['source'], r['parent_id']) for r in rows}) == 2000
    dump(OUT / 'VALIDATED.json', {'rows': 2000, 'parents': 2000, 'heldout_overlap': 0,
         'maximum_prompt_tokens': maximum, 'candidate_token_reserve': 512,
         'raw_visible_histories_verified': True, 'prompt_sha256': frozen['prompts_sha256']})


def reuse(cache_dir=None):
    validate()
    rows = read(OUT / 'prompts.jsonl')
    wanted = {r['base_id']: r for r in rows if r['kind'] != 'pragmatic'}
    reused = scored = 0
    legacy = Path(cache_dir) if cache_dir else (OUT / 'candidate_cache')
    for path in sorted((legacy / 'candidates').glob('shard_*.jsonl')):
        cached_scores = {r['id']: r for r in read(legacy / 'candidate_scores' / path.name)}
        digest = sha(path)
        for old in read(path):
            new = wanted.get(old['id'])
            if new is None:
                continue
            assert message_hash(old['messages']) == new['prompt_sha256']
            candidates = [dict(c, origin_candidate_id=c['candidate_id'], candidate_id=f"{new['id']}::c{i}") for i, c in enumerate(old['candidates'])]
            record = {**new, 'candidates': candidates, 'greedy': old['greedy'],
                      'origin': {'file': str(path), 'sha256': digest, 'id': old['id'], 'seed': old['seed']}}
            key = rank(new['id'])
            dump(OUT / 'candidates' / f'{key}.json', record); reused += 1
            if old['id'] in cached_scores:
                score = cached_scores[old['id']]
                assert score['candidate_shard_sha256'] == digest
                scores = [dict(s, candidate_id=f"{new['id']}::c{i}") for i, s in enumerate(score['scores'])]
                assert [s['candidate_id'] for s in score['scores']] == [c['origin_candidate_id'] for c in candidates]
                dump(OUT / 'scores' / f'{key}.json', {'id': new['id'], 'prompt_sha256': new['prompt_sha256'],
                     'candidate_sha256': sha(OUT / 'candidates' / f'{key}.json'), 'scores': scores,
                     'origin_score_file': str(legacy / 'candidate_scores' / path.name)})
                scored += 1
    dump(OUT / 'CACHE_REUSE.json', {'candidate_prompts_reused': reused, 'scores_reused': scored,
         'new_candidates_needed': len(rows)-reused, 'new_score_prompts_needed': len(rows)-scored})
    print('cache reused', reused, 'prompts,', scored, 'scores', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['prepare', 'audit', 'freeze', 'validate', 'reuse'])
    parser.add_argument('--cache-dir', type=Path, default=None, help='Optional candidate cache directory for reuse mode')
    args = parser.parse_args()
    if args.mode == 'reuse':
        reuse(args.cache_dir)
    else:
        globals()[args.mode]()
