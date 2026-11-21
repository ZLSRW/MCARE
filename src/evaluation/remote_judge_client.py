"""Run the existing frozen evaluation prompts through the shared local Gemma server.
Only transport changes: offline LLM.chat is replaced by a local HTTP client.
"""
import concurrent.futures
import json
import os
import sys
import time
import urllib.request
from types import SimpleNamespace as NS

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
from common.config import get_paths as _get_paths


class RemoteLLM:
    def chat(self, conversations, sampling_params, chat_template_kwargs=None):
        params = sampling_params if isinstance(sampling_params, list) else [sampling_params] * len(conversations)
        def one(pair):
            messages, p = pair
            body = dict(model='gemma', messages=messages, temperature=p.temperature,
                        top_p=p.top_p, top_k=p.top_k, max_tokens=p.max_tokens,
                        n=p.n, seed=p.seed, skip_special_tokens=p.skip_special_tokens,
                        chat_template_kwargs=chat_template_kwargs or {'enable_thinking': False})
            if p.logprobs is not None:
                body.update(logprobs=True, top_logprobs=p.logprobs)
            structured = getattr(p, 'structured_outputs', None)
            if structured and structured.json:
                body['structured_outputs'] = {'json': structured.json}
            if p.max_tokens >= 3000:
                endpoint=_get_paths().judge_url.removesuffix('/v1')+'/tokenize'
                token_request=urllib.request.Request(endpoint,data=json.dumps({'model':'gemma','messages':messages,'add_generation_prompt':True,'chat_template_kwargs':body['chat_template_kwargs']}).encode(),headers={'Content-Type':'application/json'})
                with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(token_request,timeout=60) as response:
                    budget=json.load(response)
                # The evaluation retains its original 8192-token context;
                # the shared server may be larger for RLVER's multi-turn simulator.
                remaining=min(8192,budget['max_model_len'])-budget['count']
                if remaining<=0:raise ValueError('Frozen judge input exceeds Gemma context; history must not be silently truncated')
                body['max_tokens']=min(body['max_tokens'],remaining)
                if body['max_tokens']!=p.max_tokens:print('Gemma context cap',p.max_tokens,'->',body['max_tokens'],flush=True)
            data = json.dumps(body).encode()
            for attempt in range(4):
                try:
                    request = urllib.request.Request(_get_paths().judge_url + '/chat/completions', data=data, headers={'Content-Type': 'application/json'})
                    with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(request, timeout=900) as response:
                        result = json.load(response)
                    outputs = []
                    for c in result['choices']:
                        message = c['message']; text = message.get('content') or ''
                        # If the server parser separates reasoning, explicitly mark
                        # the completed trace for the existing strict final_answer.
                        reasoning = message.get('reasoning') or message.get('reasoning_content')
                        if reasoning and text and (chat_template_kwargs or {}).get('enable_thinking'):
                            text = '</think>' + text
                        lp = []
                        for token in (c.get('logprobs') or {}).get('content', []):
                            lp.append({i: NS(decoded_token=x['token'], logprob=x['logprob']) for i, x in enumerate(token['top_logprobs'])})
                        outputs.append(NS(text=text, logprobs=lp, finish_reason=c['finish_reason']))
                    return NS(outputs=outputs)
                except Exception:
                    if attempt == 3: raise
                    time.sleep(2 ** attempt)
        workers=int(os.environ.get('GEMMA_CLIENT_WORKERS','8'))
        if not 1<=workers<=32:raise ValueError('GEMMA_CLIENT_WORKERS must be between 1 and 32')
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            results = []
            for i, result in enumerate(pool.map(one, zip(conversations, params)), 1):
                results.append(result)
                if i % 100 == 0: print(f'Gemma RPC {i}/{len(conversations)}', flush=True)
            return results


def main():
    from evaluation import run_need_judge as J
    J.load_llm = lambda *a, **kw: RemoteLLM()
    mode = sys.argv.pop(1)
    if mode == 'judge': J.main()
    elif mode == 'verify':
        import need_verify_flags as V
        V.main()
    else: raise ValueError(mode)


if __name__ == '__main__': main()
