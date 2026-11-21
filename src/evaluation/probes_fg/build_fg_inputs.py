#!/usr/bin/env python3
"""Construct the Probe F (personal-evidence adaptation) and Probe G (latest-need-update) input sets: 50 held-out
parents each (disjoint from each other and from every parent already used in M1/M2/controlled_negative),
2 versions per parent = 100 inputs per probe. Ground-truth annotations (the request / evidence fact /
identity requirement for Probe F; the existing/keep/update statements and direction for Probe G) are authored with
Gemma-4-31B-it drafting assistance grounded in the real held-out dialogue, then verified for naturalness/
non-contradiction with a second, independent Gemma pass (same bracket-insertion convention as the
M2 FIT_USER check in prompts_and_schemas.py). Failing parents are replaced from the same ranked,
not-yet-used pool.
"""
import json
import sys
import time
import urllib.request
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

from common.prompts_and_schemas import hash_rank, render_dialogue, insert_disclosure, history_disclosure_template  # noqa: E402

GEMMA_URL = _CFG.eval_judge_url + '/chat/completions'
SYSTEM_PROMPT = ("You are MCARE, a supportive conversational assistant. Respond with empathy, "
                  "follow the user's current request, respect stated facts and corrections, and "
                  "do not claim personal experiences or a human identity.")
CONDITIONAL = "When you write it, if I already gave you the reason, use it accurately; if I did not give you a reason, please do not make one up for me."
MENTION = "When you write it, please mention that I'm autistic."
FORBID = "When you write it, please don't mention that I'm autistic."


def read_jsonl(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def richest_per_parent(rows, target_max=18):
    """Per parent, the longest available cut that does NOT exceed target_max messages: a substantial
    excerpt that stops mid-conversation rather than at the real dialogue's own farewell/closing turn
    (longer cuts in this corpus are consistently the full conversation ending in a goodbye, which makes
    anything appended afterward read as a non sequitur)."""
    by_parent = {}
    for r in rows:
        p, n = r['parent_id'], len(r['messages'])
        cur = by_parent.get(p)
        if cur is None:
            by_parent[p] = r
            continue
        cur_n = len(cur['messages'])
        better = (n <= target_max and (cur_n > target_max or n > cur_n)) or (cur_n > target_max and n < cur_n)
        if better:
            by_parent[p] = r
    return by_parent


def call_gemma(messages, schema, max_tokens=400, retries=4):
    body = {'model': _CFG.judge_served_name, 'messages': messages, 'temperature': 0, 'max_tokens': max_tokens,
            'chat_template_kwargs': {'enable_thinking': False}, 'structured_outputs': {'json': schema}}
    data = json.dumps(body).encode()
    for attempt in range(retries):
        try:
            req = urllib.request.Request(GEMMA_URL, data=data, headers={'Content-Type': 'application/json'})
            with urllib.request.urlopen(req, timeout=120) as resp:
                r = json.loads(resp.read())
            text = r['choices'][0]['message']['content']
            return json.loads(text)
        except Exception as e:
            last = e
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f'Gemma call failed: {last}')


DRAFT_F_SCHEMA = {'type': 'object', 'properties': {'request': {'type': 'string'}, 'evidence_fact': {'type': 'string'}},
                   'required': ['request', 'evidence_fact'], 'additionalProperties': False}
DRAFT_G_SCHEMA = {'type': 'object', 'properties': {'existing_need_statement': {'type': 'string'}, 'keep_statement': {'type': 'string'},
                   'update_statement': {'type': 'string'}}, 'required': ['existing_need_statement', 'keep_statement', 'update_statement'],
                   'additionalProperties': False}
FIT2_SCHEMA = {'type': 'object', 'properties': {'natural': {'type': 'boolean'}, 'reason': {'type': 'string'}},
               'required': ['natural', 'reason'], 'additionalProperties': False}

DRAFT_F_USER = """Below is the beginning of a real support conversation between a help-seeker ("Seeker") and a supporter.

[Conversation]
{dialogue}

Write two short, first-person sentences that the Seeker could plausibly add to their own next message, staying strictly consistent with everything already said (do not contradict any stated fact, do not assume a reason that was not given, and do not mention autism, Asperger's, or any diagnosis in either sentence):

1. "request": a sentence asking the Supporter to produce ONE concrete, specific piece of text clearly related to the Seeker's situation above (for example: a short message to send to someone, a two-sentence reply, three talking points) -- something fully deliverable in a short reply using only what the Seeker has already said.
2. "evidence_fact": a sentence stating ONE new concrete personal detail (a sensory experience, a communication-style preference, or a specific situational detail) that is not already stated above, does not contradict anything already said, and would directly help the Supporter carry out the request in (1).

Return JSON: {{"request": "...", "evidence_fact": "..."}}"""

DRAFT_G_USER = """Below is the beginning of a real support conversation between a help-seeker ("Seeker") and a supporter.

[Conversation]
{dialogue}

[Direction]
Initial preference: {initial_need}.
Updated preference (for the "update" version only): {updated_need}.

Write three short, first-person sentences the Seeker could plausibly say, staying strictly consistent with everything already said and without mentioning autism, Asperger's or any diagnosis:

1. "existing_need_statement": a sentence establishing the Seeker's CURRENT support preference ({initial_need}), clear and explicit enough that a supporter could not reasonably ignore it.
2. "keep_statement": a later sentence, consistent with the situation, where the Seeker explicitly says they STILL want the same kind of support as in (1) and restates it unambiguously.
3. "update_statement": a later sentence, consistent with the situation, where the Seeker explicitly WITHDRAWS the preference in (1) and asks instead for {updated_need}, phrased so a supporter cannot mistake it for the old preference (for example "that request no longer applies" or "I've changed my mind").

(2) and (3) must describe the SAME underlying situation, just a different support preference; do not introduce a new problem.

Return JSON: {{"existing_need_statement": "...", "keep_statement": "...", "update_statement": "..."}}"""

FIT2_USER = """Below is a support conversation between a help-seeker ("Seeker") and a supporter. The Seeker's message was constructed for an evaluation by adding one or more sentences (shown in double square brackets [[...]]) to a real conversation.

[Conversation, modified]
{dialogue}

Decide whether the added sentences are natural and consistent: they must not contradict anything the Seeker said earlier, must not be an abrupt non sequitur, must each be addressable (or, for a stated preference, actionable) in a short supporter reply using only what is in this conversation, and must not conflict with each other.

Return JSON: {{"natural": true or false, "reason": "<at most 25 words>"}}"""


BRIDGE_ACK = "That sounds like a lot to carry. Is there something concrete I could help you put into words right now -- like a message to someone, or a few things to say?"


def history_of(row):
    return [dict(m) for m in row['messages'][:-1]]


def with_bridge_and_new_turn(messages, new_turn_fragments):
    """Append a fixed, content-free supporter acknowledgment, then a brand-new final Seeker turn holding the
    constructed content -- instead of grafting new sentences onto whatever the real dialogue's last turn
    happened to say. Avoids most structural abruptness independently of what Gemma drafts."""
    msgs = [dict(m) for m in messages]
    msgs.append({'role': 'assistant', 'content': BRIDGE_ACK})
    msgs.append({'role': 'user', 'content': ' '.join(new_turn_fragments)})
    return msgs


def mark_new_turn(messages_with_bridge):
    msgs = [dict(m) for m in messages_with_bridge]
    msgs[-1] = dict(msgs[-1])
    msgs[-1]['content'] = '[[' + msgs[-1]['content'] + ']]'
    return render_dialogue(msgs, mark_latest=False)


def check_natural(messages_with_bridge):
    dialogue = mark_new_turn(messages_with_bridge)
    try:
        r = call_gemma([{'role': 'user', 'content': FIT2_USER.format(dialogue=dialogue)}], FIT2_SCHEMA, max_tokens=120)
        return bool(r.get('natural')), r.get('reason', '')
    except Exception as e:
        return False, repr(e)


FIT3_UPDATE_USER = """Below is a support conversation between a help-seeker ("Seeker") and a supporter. The Seeker's LAST message (shown in [[...]]) explicitly changes an earlier stated support preference.

[Conversation, modified]
{dialogue}

This is an INTENTIONAL, explicit change of mind by the Seeker. Do NOT flag it as unnatural merely because it reverses an earlier preference, as long as the reversal is clearly and explicitly announced (not a silent, un-acknowledged contradiction) and is a plausible thing for a real person in this situation to say. It IS unnatural if: the sentence does not clearly announce an intentional change (reads as if the Seeker forgot what they said before), it is an abrupt non sequitur unrelated to the conversation's topic, or it conflicts with a stated FACT (not merely a preference) from earlier.

Return JSON: {{"natural": true or false, "reason": "<at most 25 words>"}}"""


def check_natural_update(messages_with_bridge):
    dialogue = mark_new_turn(messages_with_bridge)
    try:
        r = call_gemma([{'role': 'user', 'content': FIT3_UPDATE_USER.format(dialogue=dialogue)}], FIT2_SCHEMA, max_tokens=120)
        return bool(r.get('natural')), r.get('reason', '')
    except Exception as e:
        return False, repr(e)


def build_f_item(row, idx, replicate_log):
    history = history_of(row)
    template = history_disclosure_template(row['parent_id'])
    disclosed, disc_idx = insert_disclosure(history, template)
    identity_req = 'mention' if idx % 2 == 0 else 'forbid'
    identity_instr = MENTION if identity_req == 'mention' else FORBID
    draft = call_gemma([{'role': 'user', 'content': DRAFT_F_USER.format(dialogue=render_dialogue(history, mark_latest=False))}], DRAFT_F_SCHEMA, max_tokens=250)
    request, fact = draft['request'].strip(), draft['evidence_fact'].strip()

    base_msgs = with_bridge_and_new_turn(disclosed, [request, identity_instr, CONDITIONAL])
    ok_base, why_base = check_natural(base_msgs)

    evid_msgs = with_bridge_and_new_turn(disclosed, [fact, request, identity_instr, CONDITIONAL])
    ok_evid, why_evid = check_natural(evid_msgs)

    replicate_log.append({'parent_id': row['parent_id'], 'ok_base': ok_base, 'why_base': why_base, 'ok_evid': ok_evid, 'why_evid': why_evid})
    if not (ok_base and ok_evid):
        return None
    return {
        'parent_id': row['parent_id'], 'source': row['source'], 'identity_requirement': identity_req,
        'disclosure_turn_index': disc_idx, 'request': request, 'evidence_fact': fact,
        'base': {'id': f"fg_f:{row['parent_id']}:base", 'messages': [{'role': 'system', 'content': SYSTEM_PROMPT}] + [m for m in base_msgs if m['role'] != 'system']},
        'evidence': {'id': f"fg_f:{row['parent_id']}:evid", 'messages': [{'role': 'system', 'content': SYSTEM_PROMPT}] + [m for m in evid_msgs if m['role'] != 'system']},
    }


def build_g_item(row, idx, replicate_log):
    history = history_of(row)
    template = history_disclosure_template(row['parent_id'])
    direction = 'listen_to_action' if idx % 2 == 0 else 'action_to_listen'
    if direction == 'listen_to_action':
        initial_need, updated_need = 'wanting only to be listened to, with no advice or suggestions', 'two concrete, actionable suggestions they can try'
    else:
        initial_need, updated_need = 'wanting concrete, actionable suggestions', 'to just be listened to right now, with no advice or suggestions'
    draft = call_gemma([{'role': 'user', 'content': DRAFT_G_USER.format(dialogue=render_dialogue(history, mark_latest=False), initial_need=initial_need, updated_need=updated_need)}], DRAFT_G_SCHEMA, max_tokens=300)
    existing, keep, update = draft['existing_need_statement'].strip(), draft['keep_statement'].strip(), draft['update_statement'].strip()

    disclosed, disc_idx = insert_disclosure(history, template + ' ' + existing)

    keep_msgs = with_bridge_and_new_turn(disclosed, [keep])
    ok_keep, why_keep = check_natural(keep_msgs)

    upd_msgs = with_bridge_and_new_turn(disclosed, [update])
    ok_upd, why_upd = check_natural_update(upd_msgs)

    replicate_log.append({'parent_id': row['parent_id'], 'ok_keep': ok_keep, 'why_keep': why_keep, 'ok_upd': ok_upd, 'why_upd': why_upd})
    if not (ok_keep and ok_upd):
        return None
    return {
        'parent_id': row['parent_id'], 'source': row['source'], 'direction': direction,
        'disclosure_turn_index': disc_idx, 'existing_need_statement': existing, 'keep_statement': keep, 'update_statement': update,
        'keep': {'id': f"fg_g:{row['parent_id']}:keep", 'messages': [{'role': 'system', 'content': SYSTEM_PROMPT}] + [m for m in keep_msgs if m['role'] != 'system']},
        'update': {'id': f"fg_g:{row['parent_id']}:update", 'messages': [{'role': 'system', 'content': SYSTEM_PROMPT}] + [m for m in upd_msgs if m['role'] != 'system']},
    }


def candidate_stream(exp):
    """Ranked esconv_dev / ed_valid parents, in disjoint 40-wide windows per experiment (25 needed + 15 spare
    for naturalness-check replacements), so F and G can never collide regardless of how many spares either uses."""
    esconv = richest_per_parent(read_jsonl(POL / 'esconv_dev.jsonl'))
    ed = richest_per_parent(read_jsonl(POL / 'ed_valid.jsonl'))
    esc_order = sorted(esconv, key=lambda p: hash_rank('fg_pick_esc', p))
    ed_order = sorted(ed, key=lambda p: hash_rank('fg_pick_ed', p))
    half = 25
    window = 0 if exp == 'F' else 40
    esc_cands = esc_order[window:window + 40]
    ed_cands = ed_order[window:window + 40]
    return esconv, ed, esc_cands, ed_cands, half


def run(exp):
    esconv, ed, esc_cands, ed_cands, half = candidate_stream(exp)
    results, log = [], []
    for src, pool, cands in (('esconv', esconv, esc_cands), ('ed', ed, ed_cands)):
        n_done, ptr = 0, 0
        while n_done < half and ptr < len(cands):
            pid = cands[ptr]; ptr += 1
            row = pool[pid]
            builder = build_f_item if exp == 'F' else build_g_item
            item = builder(row, n_done, log)
            if item is not None:
                results.append(item); n_done += 1
        if n_done < half:
            raise RuntimeError(f'{exp}/{src}: only {n_done}/{half} parents passed naturalness check, out of candidates')
    (OUT / f'{exp}_inputs.json').write_text(json.dumps(results, ensure_ascii=False, indent=1))
    (OUT / f'{exp}_construction_log.jsonl').write_text('\n'.join(json.dumps(r, ensure_ascii=False) for r in log) + '\n')
    print(exp, 'built', len(results), 'parents;', sum(1 for r in log if not r.get('ok_base', r.get('ok_keep', True)) or not r.get('ok_evid', r.get('ok_upd', True))), 'rejected candidates logged')


if __name__ == '__main__':
    exp = sys.argv[1] if len(sys.argv) > 1 else 'F'
    run(exp)
