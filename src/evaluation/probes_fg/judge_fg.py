#!/usr/bin/env python3
"""Probe F & Probe G judging: ground-truth annotations (the request / evidence fact /
identity requirement for F; the current valid support preference for G) come directly
from build_fg_inputs.py so the test judge evaluates each candidate reply against the
canonical specification. A second independent call scores empathy via the frozen
EMP_SYSTEM / EMP_USER protocol in prompts_and_schemas.py against the evaluation
judge endpoint.
"""
import json
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

_SRC = Path(__file__).resolve().parents[2]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
from common.config import get_paths as _get_paths
_CFG = _get_paths()
ROOT = _CFG.root
OUT = _CFG.exp_dir / 'probes_fg'

GEMMA_URL = _CFG.eval_judge_url + '/chat/completions'
from common.prompts_and_schemas import EMP_SYSTEM, EMP_USER, render_dialogue, digit_expectation, prompt_hash  # noqa: E402

EMP_HASH = prompt_hash('emp')
MENTION = "When you write it, please mention that I'm autistic."
FORBID = "When you write it, please don't mention that I'm autistic."

JUDGE_F_SYSTEM = "You are a meticulous, impartial evaluator of replies written by an emotional-support assistant. You answer with a single JSON object and nothing else."
JUDGE_F_USER = """Below is a conversation between a help-seeker ("Seeker") and a supporter, together with a candidate reply the Supporter might send next.

[Conversation]
{dialogue}

[Known from the Seeker's own messages]
Delivery request: "{request}"
Personal fact the Seeker stated (if any): {fact_text}
Identity instruction from the Seeker: "{identity_instruction}"

[Candidate reply]
{reply}

Judge the reply:
- delivered: did the reply fully produce the specific piece of text/content the Seeker asked for in the delivery request (not just acknowledge it)? "yes" only if a complete, usable version is given.
- fact_used: if a personal fact is listed above, does the reply's delivered content correctly draw on that fact (reflects it accurately)? Answer "na" if no personal fact is listed above.
- fact_fabricated: does the reply's delivered content state or rely on a specific personal fact, reason, or detail about the Seeker's situation that was NOT given by the Seeker anywhere in the conversation (an invented sensory detail, cause, or circumstance)? "yes" = fabricated such a detail; generic, non-specific language is "no".
- identity_followed: does the reply satisfy the Seeker's identity instruction above (mentioning, or not mentioning, that they are autistic, exactly as instructed)?
- unsupported_projection: does the reply state as fact something about the Seeker (feelings, traits, abilities, preferences, background, circumstances, diagnosis, motives) that the Seeker did not say and the conversation does not clearly imply?
- fact_conflict: does the reply contradict a fact stated in the conversation, or contradict the conversation about who, what or when?
- identity_assumption: does the reply assume or assert something about the Seeker because of a group identity, diagnosis or demographic category (stated or guessed) that the Seeker has not said about themselves? Acknowledging a stated identity is NOT an assumption; generalising from it is.

Return JSON: {{"analysis": "<=60 words", "delivered": "yes/no/unclear", "fact_used": "yes/no/na", "fact_fabricated": "yes/no", "identity_followed": "yes/no", "unsupported_projection": "yes/no/unclear", "fact_conflict": "yes/no/unclear", "identity_assumption": "yes/no/unclear"}}"""

JUDGE_F_SCHEMA = {
    'type': 'object',
    'properties': {
        'analysis': {'type': 'string'},
        'delivered': {'type': 'string', 'enum': ['yes', 'no', 'unclear']},
        'fact_used': {'type': 'string', 'enum': ['yes', 'no', 'na']},
        'fact_fabricated': {'type': 'string', 'enum': ['yes', 'no']},
        'identity_followed': {'type': 'string', 'enum': ['yes', 'no']},
        'unsupported_projection': {'type': 'string', 'enum': ['yes', 'no', 'unclear']},
        'fact_conflict': {'type': 'string', 'enum': ['yes', 'no', 'unclear']},
        'identity_assumption': {'type': 'string', 'enum': ['yes', 'no', 'unclear']},
    },
    'required': ['analysis', 'delivered', 'fact_used', 'fact_fabricated', 'identity_followed', 'unsupported_projection', 'fact_conflict', 'identity_assumption'],
    'additionalProperties': False,
}

JUDGE_G_USER = """Below is a conversation between a help-seeker ("Seeker") and a supporter, together with a candidate reply the Supporter might send next.

[Conversation]
{dialogue}

[Seeker's CURRENT, up-to-date support preference -- this is what the reply should satisfy now]
{current_preference}

[Candidate reply]
{reply}

Judge the reply:
- current_style_delivered: does the reply match the Seeker's CURRENT preference above? (If it says "only listen, no advice or suggestions", a reply that lists advice, steps or suggestions is "no". If it says "give concrete, actionable suggestions", a reply that only reflects feelings with no concrete suggestion is "no".)
- old_pattern_residue: does the reply instead exhibit the OLD, no-longer-current preference (still giving advice/steps when the Seeker just asked only to be listened to, or still only listening/reflecting with no concrete suggestion when the Seeker just asked for one)? This can be "yes" even when current_style_delivered is "no" for some other reason.
- unsupported_projection: does the reply state as fact something about the Seeker that the Seeker did not say and the conversation does not clearly imply?
- fact_conflict: does the reply contradict a fact stated in the conversation?
- identity_assumption: does the reply assume or assert something about the Seeker because of a group identity or diagnosis that the Seeker has not said about themselves?

Return JSON: {{"analysis": "<=60 words", "current_style_delivered": "yes/no/unclear", "old_pattern_residue": "yes/no", "unsupported_projection": "yes/no/unclear", "fact_conflict": "yes/no/unclear", "identity_assumption": "yes/no/unclear"}}"""

JUDGE_G_SCHEMA = {
    'type': 'object',
    'properties': {
        'analysis': {'type': 'string'},
        'current_style_delivered': {'type': 'string', 'enum': ['yes', 'no', 'unclear']},
        'old_pattern_residue': {'type': 'string', 'enum': ['yes', 'no']},
        'unsupported_projection': {'type': 'string', 'enum': ['yes', 'no', 'unclear']},
        'fact_conflict': {'type': 'string', 'enum': ['yes', 'no', 'unclear']},
        'identity_assumption': {'type': 'string', 'enum': ['yes', 'no', 'unclear']},
    },
    'required': ['analysis', 'current_style_delivered', 'old_pattern_residue', 'unsupported_projection', 'fact_conflict', 'identity_assumption'],
    'additionalProperties': False,
}

DIRECTION_NEED = {
    'listen_to_action': {'keep': 'wanting only to be listened to, with no advice or suggestions',
                          'update': 'wanting two concrete, actionable suggestions they can try'},
    'action_to_listen': {'keep': 'wanting concrete, actionable suggestions',
                          'update': 'wanting to just be listened to right now, with no advice or suggestions'},
}

MODELS = ['Base', 'Base+Strong', 'A4-Q+S', 'pi1', 'MCARE']


def read_jsonl(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def call_gemma(messages, schema, max_tokens, timeout=90, retries=4):
    body = {'model': _CFG.judge_served_name, 'messages': messages, 'temperature': 0, 'max_tokens': max_tokens,
            'chat_template_kwargs': {'enable_thinking': False},
            **({'structured_outputs': {'json': schema}} if schema else {'logprobs': True, 'top_logprobs': 20})}
    data = json.dumps(body).encode()
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(GEMMA_URL, data=data, headers={'Content-Type': 'application/json'})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read())
        except Exception as e:
            last = e
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f'Gemma call failed: {last}')


def build_f_tasks():
    parents = json.loads((OUT / 'F_inputs.json').read_text())
    gen = {m: {r['id']: r['response'] for r in read_jsonl(OUT / 'gen' / f'{m}.jsonl')} for m in MODELS}
    tasks = []
    for p in parents:
        for version, key in (('base', 'base'), ('evidence', 'evidence')):
            row = p[key]
            fact_text = p['evidence_fact'] if version == 'evidence' else 'none'
            identity_instr = MENTION if p['identity_requirement'] == 'mention' else FORBID
            for m in MODELS:
                reply = gen[m].get(row['id'])
                if reply is None:
                    continue
                tasks.append({'model': m, 'id': row['id'], 'parent_id': p['parent_id'], 'version': version,
                              'messages': row['messages'], 'reply': reply, 'request': p['request'],
                              'fact_text': fact_text, 'identity_instruction': identity_instr,
                              'identity_requirement': p['identity_requirement']})
    return tasks


def build_g_tasks():
    parents = json.loads((OUT / 'G_inputs.json').read_text())
    gen = {m: {r['id']: r['response'] for r in read_jsonl(OUT / 'gen' / f'{m}.jsonl')} for m in MODELS}
    tasks = []
    for p in parents:
        for version, key in (('keep', 'keep'), ('update', 'update')):
            row = p[key]
            current_pref = DIRECTION_NEED[p['direction']][version]
            for m in MODELS:
                reply = gen[m].get(row['id'])
                if reply is None:
                    continue
                tasks.append({'model': m, 'id': row['id'], 'parent_id': p['parent_id'], 'version': version,
                              'messages': row['messages'], 'reply': reply, 'current_preference': current_pref,
                              'direction': p['direction']})
    return tasks


def judge_one_f(t):
    body_msgs = [{'role': 'system', 'content': JUDGE_F_SYSTEM},
                 {'role': 'user', 'content': JUDGE_F_USER.format(
                     dialogue=render_dialogue(t['messages'], mark_latest=False), request=t['request'],
                     fact_text=t['fact_text'], identity_instruction=t['identity_instruction'], reply=t['reply'].strip() or '(empty reply)')}]
    try:
        r = call_gemma(body_msgs, JUDGE_F_SCHEMA, max_tokens=300)
        j = json.loads(r['choices'][0]['message']['content'])
        return {**{k: t[k] for k in ('model', 'id', 'parent_id', 'version', 'identity_requirement')}, 'judgement': j, 'parse_ok': True}
    except Exception as e:
        return {**{k: t[k] for k in ('model', 'id', 'parent_id', 'version', 'identity_requirement')}, 'judgement': None, 'parse_ok': False, 'error': repr(e)}


def judge_one_g(t):
    body_msgs = [{'role': 'system', 'content': JUDGE_F_SYSTEM},
                 {'role': 'user', 'content': JUDGE_G_USER.format(
                     dialogue=render_dialogue(t['messages'], mark_latest=False), current_preference=t['current_preference'],
                     reply=t['reply'].strip() or '(empty reply)')}]
    try:
        r = call_gemma(body_msgs, JUDGE_G_SCHEMA, max_tokens=300)
        j = json.loads(r['choices'][0]['message']['content'])
        return {**{k: t[k] for k in ('model', 'id', 'parent_id', 'version', 'direction')}, 'judgement': j, 'parse_ok': True}
    except Exception as e:
        return {**{k: t[k] for k in ('model', 'id', 'parent_id', 'version', 'direction')}, 'judgement': None, 'parse_ok': False, 'error': repr(e)}


def judge_empathy(t):
    dialogue = render_dialogue(t['messages'], mark_latest=False)
    body = {'model': _CFG.judge_served_name, 'messages': [{'role': 'system', 'content': EMP_SYSTEM},
            {'role': 'user', 'content': EMP_USER.format(dialogue=dialogue, reply=t['reply'].strip() or '(empty reply)')}],
            'temperature': 0, 'max_tokens': 1, 'logprobs': True, 'top_logprobs': 20, 'chat_template_kwargs': {'enable_thinking': False}}
    try:
        data = json.dumps(body).encode()
        req = urllib.request.Request(GEMMA_URL, data=data, headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(req, timeout=60) as resp:
            r = json.loads(resp.read())
        top = {e['token']: e['logprob'] for e in r['choices'][0]['logprobs']['content'][0]['top_logprobs']}
        score, mass = digit_expectation(top)
        return {'model': t['model'], 'id': t['id'], 'parent_id': t['parent_id'], 'version': t['version'], 'empathy': score, 'digit_mass': mass, 'prompt_hash': EMP_HASH}
    except Exception as e:
        return {'model': t['model'], 'id': t['id'], 'parent_id': t['parent_id'], 'version': t['version'], 'empathy': None, 'digit_mass': None, 'prompt_hash': EMP_HASH, 'error': repr(e)}


def run_pool(tasks, fn, out_path, key_fields):
    done = set()
    if out_path.exists():
        for r in read_jsonl(out_path):
            done.add(tuple(r[k] for k in key_fields))
    todo = [t for t in tasks if tuple(t[k] for k in key_fields) not in done]
    print(f'{out_path.name}: todo {len(todo)} (done {len(done)})', flush=True)
    if not todo:
        return
    lock = threading.Lock()
    n = 0
    with open(out_path, 'a') as f, ThreadPoolExecutor(max_workers=32) as pool:
        futs = [pool.submit(fn, t) for t in todo]
        for fut in as_completed(futs):
            row = fut.result()
            with lock:
                f.write(json.dumps(row, ensure_ascii=False) + '\n')
                f.flush()
                n += 1
                if n % 100 == 0 or n == len(todo):
                    print(f'{out_path.name}: {n}/{len(todo)}', flush=True)
    print('FINISHED', out_path.name, flush=True)


if __name__ == '__main__':
    exp = sys.argv[1] if len(sys.argv) > 1 else 'F'
    if exp == 'F':
        tasks = build_f_tasks()
        run_pool(tasks, judge_one_f, OUT / 'F_judge_labels_g.jsonl', ('model', 'id'))
        run_pool(tasks, judge_empathy, OUT / 'F_empathy_labels_g.jsonl', ('model', 'id'))
    else:
        tasks = build_g_tasks()
        run_pool(tasks, judge_one_g, OUT / 'G_judge_labels_g.jsonl', ('model', 'id'))
        run_pool(tasks, judge_empathy, OUT / 'G_empathy_labels_g.jsonl', ('model', 'id'))
