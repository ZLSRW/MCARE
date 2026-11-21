"""Paths and frozen protocol for the new 2k shared candidate pool."""
import hashlib
import json
from pathlib import Path

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
from common.config import get_paths as _get_paths
_CFG = _get_paths()
ROOT = _CFG.root
RUN = _CFG.baselines_dir
OUT = RUN / 'shared_pool'
POLICY = _CFG.policy_dir
SEED = 42


def read(path):
    return [json.loads(l) for l in path.open() if l.strip()] if path.exists() else []


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rank(*parts):
    return hashlib.sha256('|'.join(map(str, (SEED,) + parts)).encode()).hexdigest()


def message_hash(messages):
    return hashlib.sha256(json.dumps(messages, sort_keys=True, ensure_ascii=False,
                                     separators=(',', ':')).encode()).hexdigest()


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)


def write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
    tmp.replace(path)


def schema(properties):
    return {'type': 'object', 'properties': properties, 'required': list(properties),
            'additionalProperties': False}


TEXT = {'type': 'string'}
BOOL = {'type': 'boolean'}
AUDIT_SCHEMA = schema({'category': {'type': 'string', 'enum':
    ['none', 'explicit_prohibition', 'rejection', 'failed_attempt']},
    'user_evidence': TEXT, 'prior_assistant_evidence': TEXT, 'target': TEXT,
    'narrow_response_requirement': TEXT, 'reason': TEXT})
AUDIT_PROMPT = '''Audit only this public TRAINING conversation, before any new reply exists.
Determine whether the LATEST user turn explicitly prohibits a support behavior, rejects
a particular earlier suggestion, or reports trying a SPECIFIC strategy unsuccessfully.
Return none for ordinary distress, disagreement about events, inability without an
identified strategy, vague "I don't know", or a request for more help without pushback.
Quote exact substrings of the latest user message and, when referring to an earlier
suggestion, the relevant earlier assistant message. Identify the concrete target.
Explicit prohibition requires an expressed boundary. Rejection and failed attempts
are separate categories: do NOT invent a general prohibition on advice. Their narrow
requirement is to acknowledge the feedback and avoid presenting that SAME strategy
unchanged as a fresh solution; adapted or different help remains allowed.
No inference from unseen situation/prompt fields or the original next reply.
Return JSON only, with category, user_evidence, prior_assistant_evidence (empty if not
applicable), target, narrow_response_requirement, reason.
Conversation:\n{dialogue}'''
REVIEW_SCHEMA = schema({'accepted': BOOL, 'reason': TEXT})
REVIEW_PROMPT = '''Verify this proposed feedback annotation using only the training history.
Accept only if the latest user actually supplies the quoted evidence, the target is
specific and supported, and the response requirement does not broaden rejection or
an unsuccessful attempt into a ban on all advice or all questions. An ordinary
negative emotion or "I don't know" is insufficient. The annotation must be useful
for checking whether a new reply ignores this particular feedback. Return JSON
accepted and reason.\nConversation:\n{dialogue}\nAnnotation:\n{annotation}'''


def dialogue(messages):
    return '\n'.join(f"{m['role']}: {m['content']}" for m in messages if m['role'] != 'system')


def chat(conversations, json_schema, max_tokens=650, max_workers=16):
    import concurrent.futures
    import time
    import urllib.request

    url = _CFG.train_annotator_url + '/chat/completions'
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {_CFG.train_annotator_api_key}',
    }

    def _one(msgs):
        body = {
            'model': _CFG.train_annotator_served_name,
            'messages': msgs,
            'temperature': 0,
            'max_tokens': max_tokens,
            'seed': 0,
            'chat_template_kwargs': {'enable_thinking': False},
            'structured_outputs': {'json': json_schema},
        }
        data = json.dumps(body).encode('utf-8')
        for attempt in range(4):
            try:
                req = urllib.request.Request(url, data=data, headers=headers)
                with opener.open(req, timeout=300) as resp:
                    res = json.load(resp)
                return res['choices'][0]['message']['content']
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(2 ** attempt)

    if len(conversations) <= 1 or max_workers <= 1:
        return [_one(c) for c in conversations]
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        return list(pool.map(_one, conversations))
