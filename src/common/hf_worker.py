"""HF sampling and local-only training-annotation API on the authorized server.

All generated rows are persisted immediately and resumed by content hash.
No truncation: over-context examples fail visibly rather than dropping a need.
"""
import argparse
import json
import os
import random
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from concurrent.futures import Future, TimeoutError as FutureTimeout
import queue
import threading
import select
import socket
from pathlib import Path
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
from common.config import get_paths as _get_paths
from common.io_utils import digest, dump_json as dump, key, load_json, read_jsonl as read, sha, write_jsonl as write

_CFG = _get_paths()
ROOT = _CFG.exp_dir
PROJECT = _CFG.root


def load(path):
    import torch
    from transformers import AutoModelForImageTextToText, AutoTokenizer
    tok=AutoTokenizer.from_pretrained(path,local_files_only=True,padding_side='left')
    kwargs={'dtype':torch.bfloat16,'attn_implementation':'sdpa','local_files_only':True}
    count=torch.cuda.device_count()
    if count>1:
        kwargs.update(device_map='auto',max_memory={i:'43GiB' for i in range(count)})
    else:kwargs['device_map']='cuda:0'
    model=AutoModelForImageTextToText.from_pretrained(path,**kwargs).eval()
    tok.pad_token=tok.pad_token or tok.eos_token
    return model,tok

def generate(model,tok,messages,n=1,greedy=True,max_tokens=512,seed=0):
    import torch
    torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);random.seed(seed)
    inputs=tok.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,
        enable_thinking=False,return_dict=True,return_tensors='pt').to(model.device)
    prompt_tokens=inputs['input_ids'].shape[-1]
    if prompt_tokens+max_tokens>32768:raise ValueError('Context limit; refusing silent truncation')
    kw={'max_new_tokens':max_tokens,'do_sample':not greedy,'num_return_sequences':n,
        'pad_token_id':tok.pad_token_id,'use_cache':True}
    if not greedy:kw.update(temperature=1.,top_p=.95)
    start=time.monotonic()
    with torch.inference_mode():out=model.generate(**inputs,**kw)
    rec=[]
    eos=model.generation_config.eos_token_id
    eos={eos} if isinstance(eos,int) else set(eos or [])
    for i in range(n):
        ids=out[i,prompt_tokens:].tolist()
        stop=next((j+1 for j,t in enumerate(ids) if t in eos),len(ids))
        ids=ids[:stop]
        rec.append({'text':tok.decode(ids,skip_special_tokens=True).strip(),
            'raw':tok.decode(ids,skip_special_tokens=False),'tokens':len(ids),
            'hit_cap':len(ids)>=max_tokens and (not ids or ids[-1] not in eos),
            'finish_reason':'stop' if ids and ids[-1] in eos else 'length'})
    return rec,{'prompt_tokens':prompt_tokens,'seconds':time.monotonic()-start}

def generate_batch(model,tok,histories,max_tokens=768,gate=None,mask_fn=None):
    """Fixed left-padded greedy batches, shared by base and every adapter.

    Gate features use only each unpadded USER history. The cached decode keeps
    one scalar for each batch item, never the average across different users.
    """
    import torch
    old_padding=tok.padding_side;tok.padding_side='left'
    try:
        inputs=tok.apply_chat_template(histories,tokenize=True,add_generation_prompt=True,
            enable_thinking=False,return_dict=True,return_tensors='pt',padding=True).to(model.device)
    finally:tok.padding_side=old_padding
    width=inputs['input_ids'].shape[-1]
    if width+max_tokens>32768:raise ValueError('Evaluation batch over context limit')
    if gate:
        assert mask_fn is not None
        masks=[]
        for messages,attention in zip(histories,inputs['attention_mask']):
            m=mask_fn(tok,messages,model.device)[0]
            assert len(m)==int(attention.sum()),'Gate/tokenizer history alignment failed'
            masks.append(torch.nn.functional.pad(m,(width-len(m),0)))
        gate.reset(torch.stack(masks))
    start=time.monotonic()
    with torch.inference_mode():
        outputs=model.generate(**inputs,max_new_tokens=max_tokens,do_sample=False,
            pad_token_id=tok.pad_token_id,use_cache=True)
    seconds=time.monotonic()-start
    eos=model.generation_config.eos_token_id
    eos={eos} if isinstance(eos,int) else set(eos or [])
    result=[]
    for j in range(len(histories)):
        ids=outputs[j,width:].tolist()
        stop=next((k+1 for k,t in enumerate(ids) if t in eos),len(ids));ids=ids[:stop]
        result.append({'text':tok.decode(ids,skip_special_tokens=True).strip(),
          'raw':tok.decode(ids,skip_special_tokens=False),'tokens':len(ids),
          'hit_cap':len(ids)>=max_tokens and (not ids or ids[-1] not in eos),
          'finish_reason':'stop' if ids and ids[-1] in eos else 'length',
          'gate':float(gate.current[j].mean()) if gate else None,
          'prompt_tokens':int(inputs['attention_mask'][j].sum()),
          'seconds':seconds,'batch_size':len(histories)})
    return result

def sample(args):
    rows=read(args.tasks)[args.shard::args.shards]
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    todo=[r for r in rows if not (out/(key(r['id'])+'.json')).exists()]
    if not todo:print('sampling complete');return
    model,tok=load(args.model)
    for pos,r in enumerate(todo):
        seed=int(key(['candidate',r['id']])[:8],16)%2147483647
        cs,stat=generate(model,tok,r['messages'],n=r.get('n',8),greedy=False,seed=seed)
        if any(not c['text'] for c in cs):raise ValueError('Empty candidate '+r['id'])
        for i,c in enumerate(cs):c['candidate_id']=r['id']+'::c'+str(i)
        g,_=generate(model,tok,r['messages'],greedy=True) if r.get('greedy') else ([None],None)
        dump(out/(key(r['id'])+'.json'),{**r,'candidates':cs,'greedy':g[0],
            'seed':seed,'model':args.model,'model_kind':'official Qwen3.5-9B','statistics':stat})
        print(json.dumps({'id':r['id'],'done':pos+1,'todo':len(todo),**stat}),flush=True)

def serve(args):
    backend='torch'
    if args.fla:
        # Opt-in is confined to the independent training-label server. The
        # policy training and frozen evaluation engine retain their defaults.
        import sys,inspect
        import transformers.models.qwen3_5.modeling_qwen3_5 as native
        if os.environ.get('MCARE_FLA_PATHS'): sys.path[:0]=os.environ['MCARE_FLA_PATHS'].split(':')
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule,fused_recurrent_gated_delta_rule
        def compatible(fn):
            allowed=tuple(inspect.signature(fn).parameters)
            def wrapped(*a,**kw):return fn(*a,**{k:v for k,v in kw.items() if k in allowed})
            return wrapped
        native.torch_chunk_gated_delta_rule=compatible(chunk_gated_delta_rule)
        native.torch_recurrent_gated_delta_rule=compatible(fused_recurrent_gated_delta_rule)
        backend='isolated fla-core 0.3.2; Triton 3.2; bf16'
    model,tok=load(args.model)
    pending=queue.Queue()
    state={'current_batch':None}
    def inference_loop():
        import torch
        while True:
            req,fut=pending.get()
            if fut.cancelled():continue
            batch=[(req,fut)]
            until=time.monotonic()+.12
            while len(batch)<8 and time.monotonic()<until:
                try:
                    item=pending.get(timeout=max(.001,until-time.monotonic()))
                    if not item[1].cancelled():batch.append(item)
                except queue.Empty:break
            batch=[(r,f) for r,f in batch if f.set_running_or_notify_cancel()]
            if not batch:continue
            try:
                inp=tok.apply_chat_template([r['messages'] for r,f in batch],tokenize=True,
                    add_generation_prompt=True,enable_thinking=False,return_dict=True,
                    return_tensors='pt',padding=True).to(model.device)
                width=inp['input_ids'].shape[-1];budget=max(int(r.get('max_tokens',1800)) for r,f in batch)
                if width+budget>32768:raise ValueError('Annotation batch over context limit')
                start=time.monotonic()
                state['current_batch']={'batch_size':len(batch),'padded_prompt_tokens':width,
                    'max_new_tokens':budget,'started_monotonic':start}
                with torch.inference_mode():ids=model.generate(**inp,max_new_tokens=budget,
                    do_sample=False,pad_token_id=tok.pad_token_id,use_cache=True)
                eos=model.generation_config.eos_token_id;eos={eos} if isinstance(eos,int) else set(eos or [])
                for j,(r,f) in enumerate(batch):
                    tokens=ids[j,width:].tolist();stop=next((i+1 for i,t in enumerate(tokens) if t in eos),len(tokens));tokens=tokens[:stop]
                    f.set_result({'text':tok.decode(tokens,skip_special_tokens=True).strip(),
                        'hit_cap':len(tokens)>=budget and (not tokens or tokens[-1] not in eos),
                        'batch_size':len(batch),'prompt_tokens':int(inp['attention_mask'][j].sum()),
                        'seconds':time.monotonic()-start,'linear_attention_backend':backend})
            except Exception as e:
                import traceback
                traceback.print_exc()
                for r,f in batch:f.set_exception(e)
            finally:state['current_batch']=None
    threading.Thread(target=inference_loop,daemon=True).start()
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            current=state['current_batch']
            if current:current={**current,'elapsed_seconds':time.monotonic()-current['started_monotonic']}
            b=json.dumps({'model':args.model,'ready':True,'linear_attention_backend':backend,'pending_requests':pending.qsize(),'current_batch':current}).encode()
            self.send_response(200);self.end_headers();self.wfile.write(b)
        def do_POST(self):
            try:
                req=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                fut=Future();pending.put((req,fut));deadline=time.monotonic()+1800
                while True:
                    try:result=fut.result(timeout=.5);break
                    except FutureTimeout:
                        if time.monotonic()>deadline:
                            fut.cancel();raise TimeoutError('Training audit exceeded 1800 seconds')
                        # A restarted audit client closes its sockets. Cancel
                        # queued orphan requests before spending GPU inference
                        # on outputs no live client can persist.
                        readable,_,_=select.select([self.connection],[],[],0)
                        if readable and self.connection.recv(1,socket.MSG_PEEK)==b'':
                            fut.cancel();return
                b=json.dumps(result).encode()
                self.send_response(200)
            except Exception as e:
                import traceback
                traceback.print_exc();b=json.dumps({'error':repr(e)}).encode();self.send_response(500)
            self.end_headers();self.wfile.write(b)
    print('TRAIN_JUDGE_READY',args.port,flush=True)
    ThreadingHTTPServer((args.host,args.port),Handler).serve_forever()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['sample','serve'])
    ap.add_argument('--model',required=True);ap.add_argument('--tasks');ap.add_argument('--out')
    ap.add_argument('--shard',type=int,default=0);ap.add_argument('--shards',type=int,default=1)
    ap.add_argument('--port',type=int,default=8089)
    ap.add_argument('--fla',action='store_true',help='isolated optional kernels for training-audit server only')
    a=ap.parse_args();(sample if a.mode=='sample' else serve)(a)

if __name__=='__main__':main()
