"""Sample missing candidates, reuse/scoring Q+Support, annotate pragmatic costs, form pairs."""
import argparse
import collections
import json
import math
import os
import re
import statistics
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
from baselines.matched_pool_common import *

COST_ITEM = schema({'index': {'type': 'integer', 'minimum': 0, 'maximum': 7},
    'pragmatic_violation': BOOL, 'fabricated_assistant_experience': BOOL,
    'unsupported_identity_projection': BOOL, 'reason': TEXT})
COST_SCHEMA = schema({'checks': {'type': 'array', 'items': COST_ITEM, 'minItems': 8, 'maxItems': 8}})
COST_PROMPT = '''Check each of eight candidate replies INDEPENDENTLY against the same
training history and its applicable requirement. Do not compare quality, warmth, or
length, and do not prefer any candidate. No quality scores are provided.
For ordinary natural dialogue there is no hidden user requirement and no blanket ban
on advice/questions. For feedback, enforce ONLY the supplied narrow feedback target:
failed attempts/rejection do not prohibit all advice; only proposing the same failed
strategy unchanged while disregarding the expressed feedback violates this check.
For an appended pragmatic instruction, check its requested behavior and stated facts
semantically, accepting paraphrases. "Message only" excludes commentary, alternative
messages, placeholder names/signatures, and appended explanations. First-person words
inside a requested message draft refer to the user and do NOT fabricate the assistant's
personal experience. Fabricated_assistant_experience means claiming the assistant
personally lived an event, not merely expressing empathy. Unsupported identity means
asserting an identity/diagnosis about the user unsupported by visible user text.
Return eight records, indices 0 through 7 exactly once; each reason must cite the
concrete issue or say no applicable violation. Count NO length limits or sentence caps.
History:\n{dialogue}\nApplicable requirement:\n{requirement}\nCandidates:\n{candidates}'''


def sample():
    import torch
    from vllm import LLM, SamplingParams
    from baselines.matched_pool_data import validate
    validate()
    todo = [r for r in read(OUT / 'prompts.jsonl') if not (OUT / 'candidates' / f"{rank(r['id'])}.json").exists()]
    if not todo:
        print('all candidates already present', flush=True); return
    llm = LLM(model=_CFG.base_model, dtype='bfloat16',
               max_model_len=4096, gpu_memory_utilization=.80, seed=20261011,
               max_num_seqs=64, limit_mm_per_prompt={'image': 0, 'video': 0})
    for start in range(0, len(todo), 32):
        batch = todo[start:start+32]; seed = 20261011 + start // 32
        replies = llm.chat([r['messages'] for r in batch], SamplingParams(n=8, temperature=1.,
             top_p=.95, max_tokens=512, seed=seed), chat_template_kwargs={'enable_thinking': False})
        greedy = llm.chat([r['messages'] for r in batch], SamplingParams(temperature=0,
             max_tokens=512, seed=0), chat_template_kwargs={'enable_thinking': False})
        for r, reply, g in zip(batch, replies, greedy):
            assert len(reply.outputs) == 8
            cs = [{'candidate_id': f"{r['id']}::c{i}", 'raw': c.text, 'text': c.text.strip(),
                   'tokens': len(c.token_ids), 'finish_reason': c.finish_reason,
                   'hit_cap': c.finish_reason == 'length'} for i, c in enumerate(reply.outputs)]
            dump(OUT / 'candidates' / f"{rank(r['id'])}.json", {**r, 'candidates': cs,
                 'seed': seed, 'greedy': {'text': g.outputs[0].text.strip(),
                 'tokens': len(g.outputs[0].token_ids), 'finish_reason': g.outputs[0].finish_reason}})
        print('sampled new prompts', min(start+32, len(todo)), '/', len(todo), flush=True)


def _score_train_helpfulness(flat):
    import concurrent.futures
    import time
    import urllib.request
    from common import prompts_and_schemas as P
    from stage2.score_candidates import HELP_SYSTEM, HELP_USER

    url = _CFG.train_annotator_url + '/chat/completions'
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {_CFG.train_annotator_api_key}',
    }

    def _one(pair):
        r, c = pair
        body = {
            'model': _CFG.train_annotator_served_name,
            'temperature': 0,
            'max_tokens': 1,
            'logprobs': True,
            'top_logprobs': 20,
            'chat_template_kwargs': {'enable_thinking': False},
            'messages': [
                {'role': 'system', 'content': HELP_SYSTEM},
                {'role': 'user', 'content': HELP_USER.format(
                    dialogue=P.render_dialogue(r['messages'], mark_latest=False),
                    reply=c['text'].strip() or '(empty reply)',
                )},
            ],
        }
        data = json.dumps(body).encode('utf-8')
        for attempt in range(4):
            try:
                req = urllib.request.Request(url, data=data, headers=headers)
                with opener.open(req, timeout=300) as resp:
                    res = json.load(resp)
                top = {x['token']: x['logprob'] for x in res['choices'][0]['logprobs']['content'][0]['top_logprobs']}
                exp, _ = P.digit_expectation(top)
                return float(exp if exp is not None else 3.0)
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(2 ** attempt)

    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
        return list(pool.map(_one, flat))


def score():
    """Score matched-pool candidates on helpfulness and support utility."""
    from baselines.external.rlff_rm import RLFFRM, context_text
    prompts = read(OUT / 'prompts.jsonl')
    todo = [r for r in prompts if not (OUT / 'scores' / f"{rank(r['id'])}.json").exists()]
    if not todo:
        print('all Q/Support scores already cached', flush=True); return
    support = RLFFRM(_CFG.train_support_rm, 'cuda')
    for start in range(0, len(todo), 16):
        batch = todo[start:start+16]
        records = [json.loads((OUT / 'candidates' / f"{rank(r['id'])}.json").read_text()) for r in batch]
        flat = [(r, c) for r in records for c in r['candidates']]
        help_scores = _score_train_helpfulness(flat)
        probs = support.score([context_text([m for m in r['messages'] if m['role'] != 'system']) for r, c in flat], [c['text'] for r, c in flat])
        cursor = 0
        for r in records:
            scores = []
            for c in r['candidates']:
                q = float(help_scores[cursor]); p = float(probs[cursor])
                scores.append({'candidate_id': c['candidate_id'],
                               'q': q, 'p_E': p, 'zQ': (q - 3.0) / 0.8, 'zE': (p - 0.60) / 0.20})
                cursor += 1
            dump(OUT / 'scores' / f"{rank(r['id'])}.json", {'id': r['id'],
                 'prompt_sha256': r['prompt_sha256'],
                 'candidate_sha256': sha(OUT / 'candidates' / f"{rank(r['id'])}.json"), 'scores': scores})
        print('scored new prompts', min(start+16, len(todo)), '/', len(todo), flush=True)


def costs():
    fab_pat = re.compile(
        r"(i('ve| have) (been (through|there|in a similar)|gone through|experienced)|when i was|in my (own )?(life|experience)|"
        r"my own (experience|life)|i (also )?went through|as someone who has|i know exactly how (it|that) feels)",
        re.I,
    )
    identity_pat = re.compile(r"\b(diagnos(is|ed)|disorder|condition|spectrum|neurodiver)\b", re.I)
    protocol = {'prompt': COST_PROMPT, 'schema': COST_SCHEMA, 'model': _CFG.train_annotator_model,
                'candidate_scores_hidden': True, 'length_not_a_constraint': True,
                'natural_prescreen': 'Only candidates matching surface fabrication/identity patterns need global flag verification; unmatched natural responses receive zero proxy cost.'}
    path = OUT / 'COST_PROTOCOL.json'
    if path.exists():
        assert json.loads(path.read_text()) == protocol
    else:
        dump(path, protocol)
    formatting = 'Return compact JSON on ONE line, without indentation or formatting whitespace. Keep each reason to one brief sentence.'
    repair = {'frozen_cost_protocol_sha256': sha(path), 'decision_criteria_unchanged': True,
              'formatting_instruction': formatting, 'max_tokens': 2400,
              'retries_on_invalid_json': 3, 'invalid_outputs_never_become_labels': True}
    dump(OUT / 'COST_SERIALIZATION_REPAIR.json', repair)

    def parse_checks(text):
        result = json.loads(text)
        checks = sorted(result['checks'], key=lambda c: c['index'])
        assert len(checks) == 8 and [c['index'] for c in checks] == list(range(8))
        for c in checks:
            assert all(type(c[k]) is bool for k in ['pragmatic_violation', 'fabricated_assistant_experience', 'unsupported_identity_projection'])
            assert isinstance(c['reason'], str)
        return checks
    todo = [r for r in read(OUT / 'prompts.jsonl') if not (OUT / 'costs' / f"{rank(r['id'])}.json").exists()]
    for start in range(0, len(todo), 8):
        batch = todo[start:start+8]; tasks = []; records = []
        for r in batch:
            candidate_path = OUT / 'candidates' / f"{rank(r['id'])}.json"
            record = json.loads(candidate_path.read_text()); records.append(record)
            require_semantics = r['kind'] != 'natural' or any(fab_pat.search(c['text']) or identity_pat.search(c['text']) for c in record['candidates'])
            if not require_semantics:
                checks = [{'index': i, 'pragmatic_violation': False, 'fabricated_assistant_experience': False,
                           'unsupported_identity_projection': False, 'reason': 'No explicit check; global phrase prescreen negative.'} for i in range(8)]
                dump(OUT / 'costs' / f"{rank(r['id'])}.json", {'id': r['id'], 'checks': checks,
                     'candidate_sha256': sha(candidate_path), 'source': 'natural-global-prescreen'})
                continue
            requirement = r.get('instruction') or json.dumps(r.get('feedback') or {'no_explicit_extra_requirement': True})
            candidates = '\n\n'.join(f"[{i}] {c['text']}" for i, c in enumerate(record['candidates']))
            task = [{'role': 'user', 'content': COST_PROMPT.format(dialogue=dialogue(r['messages']), requirement=requirement, candidates=candidates) + '\n\n' + formatting}]
            tasks.append((record, task))
        raw = chat([task for r, task in tasks], COST_SCHEMA, 2400) if tasks else []
        for (r, task), text in zip(tasks, raw):
            for attempt in range(4):
                try:
                    checks = parse_checks(text)
                    break
                except (ValueError, KeyError, TypeError, AssertionError) as error:
                    dump(OUT / 'cost_errors' / f"{rank(r['id'])}_{attempt}.json",
                         {'id': r['id'], 'raw': text, 'error': repr(error), 'attempt': attempt,
                          'candidate_sha256': sha(OUT / 'candidates' / f"{rank(r['id'])}.json")})
                    if attempt == 3:
                        raise
                    print('Retry invalid cost JSON', r['id'], attempt+1, flush=True)
                    text = chat([task], COST_SCHEMA, 2400)[0]
            dump(OUT / 'costs' / f"{rank(r['id'])}.json", {'id': r['id'], 'checks': checks,
                 'candidate_sha256': sha(OUT / 'candidates' / f"{rank(r['id'])}.json"),
                 'source': f'{_CFG.train_annotator_model}-semantic-proxy', 'raw': text,
                 'serialization_repair_sha256': sha(OUT / 'COST_SERIALIZATION_REPAIR.json'), 'retry_count': attempt})
        print('cost annotations', min(start+8, len(todo)), '/', len(todo), flush=True)


def pairs():
    from baselines.matched_pool_data import validate
    validate()
    labels = ['Joint-DPO', 'SafeDPO']
    pairs = {k: [] for k in labels}; audits = {k: [] for k in labels}; skips = []
    for prompt in read(OUT / 'prompts.jsonl'):
        key = rank(prompt['id']); path = OUT / 'candidates' / f'{key}.json'
        r = json.loads(path.read_text()); s = json.loads((OUT / 'scores' / f'{key}.json').read_text())
        h = json.loads((OUT / 'costs' / f'{key}.json').read_text())
        assert len(r['candidates']) == len(s['scores']) == len(h['checks']) == 8
        assert r['prompt_sha256'] == s['prompt_sha256'] == prompt['prompt_sha256']
        assert s['candidate_sha256'] == h['candidate_sha256'] == sha(path)
        seen, candidates = set(), []
        for c, score, check in zip(r['candidates'], s['scores'], h['checks']):
            assert c['candidate_id'] == score['candidate_id']
            text = c['text'].strip()
            if not text or text in seen or not all(math.isfinite(score[k]) for k in ['q', 'p_E', 'zQ', 'zE']):
                continue
            seen.add(text)
            cost = sum(check[k] for k in ['pragmatic_violation', 'fabricated_assistant_experience', 'unsupported_identity_projection'])
            candidates.append({**c, **score, 'score': score['zQ']+score['zE'], 'cost': cost, 'check': check})
        if len(candidates) < 2:
            skips.append({'id': r['id'], 'reason': 'fewer than two valid unique replies'}); continue
        ordered = sorted(candidates, key=lambda c: (c['score'], c['candidate_id']))
        bad, good = ordered[0], ordered[-1]
        if not good['score'] > bad['score']:
            skips.append({'id': r['id'], 'reason': 'tied score extrema'}); continue
        def add(label, good, bad, margin=None):
            pair = {'id': r['id'], 'source': r['source'], 'parent_id': r['parent_id'],
                    'kind': r['kind'], 'prompt_sha256': r['prompt_sha256'],
                    'messages': r['messages']+[{'role': 'assistant', 'content': good['text']}],
                    'rejected_response': bad['text']}
            if margin is not None:
                pair['margin'] = margin
            pairs[label].append(pair)
            audits[label].append({'id': r['id'], 'kind': r['kind'], 'chosen_id': good['candidate_id'],
                'rejected_id': bad['candidate_id'], 'chosen_cost': good['cost'], 'rejected_cost': bad['cost'],
                'chosen_score': good['score'], 'rejected_score': bad['score'],
                'chosen_words': len(good['text'].split()), 'rejected_words': len(bad['text'].split()),
                'chosen_hit_cap': good['hit_cap'], 'rejected_hit_cap': bad['hit_cap'],
                'margin': margin, 'chosen_check': good['check'], 'rejected_check': bad['check']})
        add(labels[0], good, bad)
        if good['cost'] and bad['cost']:
            skips.append({'id': r['id'], 'method': 'SafeDPO', 'reason': 'both initial replies violate'}); continue
        if good['cost']:
            good, bad = bad, good
        add(labels[1], good, bad, 10. if bad['cost'] else 0.)
    report = {'pool_sha256': sha(OUT / 'prompts.jsonl'), 'protocol_sha256': sha(OUT / 'PROTOCOL.json'),
              'cost_protocol_sha256': sha(OUT / 'COST_PROTOCOL.json'), 'seed': 42, 'runs': {}}
    # Verify SafeDPO changes only the input pair's orientation/admission.
    initial = {r['id']: r for r in pairs[labels[0]]}
    initial_audit = {r['id']: r for r in audits[labels[0]]}
    for row, a in zip(pairs[labels[1]], audits[labels[1]]):
        original = initial[row['id']]; oa = initial_audit[row['id']]
        assert row['messages'][:-1] == original['messages'][:-1]
        current_texts = {row['messages'][-1]['content'], row['rejected_response']}
        original_texts = {original['messages'][-1]['content'], original['rejected_response']}
        assert current_texts == original_texts and len(current_texts) == 2
        assert a['chosen_cost'] == 0
        assert row['margin'] == (10. if a['rejected_cost'] else 0.)
        if oa['chosen_cost'] == 0:
            assert a['chosen_id'] == oa['chosen_id'] and a['rejected_id'] == oa['rejected_id']
        else:
            assert oa['rejected_cost'] == 0
            assert a['chosen_id'] == oa['rejected_id'] and a['rejected_id'] == oa['chosen_id']
    for label, rows in pairs.items():
        assert rows, f'{label} produced no pairs'
        path = RUN / 'data' / f'{label}.jsonl'; write(path, rows)
        write(RUN / 'data' / f'{label}_audit.jsonl', audits[label])
        a = audits[label]
        report['runs'][label] = {'data': str(path), 'sha256': sha(path), 'pairs': len(rows),
            'mean_chosen_words': statistics.mean(x['chosen_words'] for x in a),
            'mean_rejected_words': statistics.mean(x['rejected_words'] for x in a),
            'chosen_longer_fraction': statistics.mean(x['chosen_words'] > x['rejected_words'] for x in a),
            'compliant_violating_pairs': sum(x['chosen_cost'] == 0 and x['rejected_cost'] > 0 for x in a),
            'chosen_hit_cap': sum(x['chosen_hit_cap'] for x in a), 'rejected_hit_cap': sum(x['rejected_hit_cap'] for x in a)}
    report['safe_transform_verified'] = True
    write(OUT / 'pair_exclusions.jsonl', skips)
    dump(RUN / 'data/SHARED_PAIRS_READY.json', report)
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['sample', 'score', 'costs', 'pairs'])
    mode = parser.parse_args().mode
    globals()[mode]()
