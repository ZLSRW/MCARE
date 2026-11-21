"""Empathy Specialist (pi_E) and Structured Preference Trainer.

Trains the LoRA empathy specialist pi_E via evidence-conditioned self-distillation
and empathy-preference DPO (Section III-C) from offline training annotations.
"""
import argparse
import contextlib
import json
import math
import os
import random
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForImageTextToText, AutoTokenizer
import sys

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
from common.config import get_paths as _get_paths
from common.io_utils import digest, dump_json as dump, key, load_json, read_jsonl as read, sha, write_jsonl as write

_CFG = _get_paths()
PROJECT = _CFG.root
ROOT = _CFG.specialist_dir
from stage2.losses import distill, dpo, reversal


def load_train(args):
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    tok = AutoTokenizer.from_pretrained(args.base, local_files_only=True)
    base = AutoModelForImageTextToText.from_pretrained(
        args.base, dtype=torch.bfloat16, device_map='cuda:0', attn_implementation='sdpa', local_files_only=True
    )
    pattern = r'^(model\.language_model(?=\.).*\.(in_proj_b|up_proj|down_proj|v_proj|k_proj|gate_proj|o_proj|in_proj_a|in_proj_z|in_proj_qkv|q_proj|out_proj))$'
    config = LoraConfig(r=16, lora_alpha=32, lora_dropout=.05, target_modules=pattern, task_type='CAUSAL_LM')
    model = get_peft_model(base, config)
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    model.config.use_cache = False
    model.train()
    return model, tok, None

def tokenize(tok,messages,text,max_length=4096):
    prefix=tok.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)
    full=tok.apply_chat_template(messages+[{'role':'assistant','content':text}],tokenize=False,add_generation_prompt=False,enable_thinking=False)
    if not full.startswith(prefix):raise ValueError('Chat template completion prefix mismatch')
    loc=len(prefix)
    if full[loc:loc+len(text)]!=text:raise ValueError('Response text does not match template boundary')
    enc=tok(full,add_special_tokens=False,return_offsets_mapping=True)
    if len(enc['input_ids'])>max_length:raise ValueError('Overlength example: no silent truncation')
    offsets=enc['offset_mapping'];positions=[];relative=[]
    for i,(a,b) in enumerate(offsets):
        if i and b>loc and a<loc+len(text):positions.append(i-1);relative.append((max(0,a-loc),min(len(text),b-loc)))
    # Also teach actual turn termination; never mask only deleted characters.
    eos=tok.convert_tokens_to_ids('<|im_end|>')
    term=next((i for i,t in enumerate(enc['input_ids']) if t==eos and offsets[i][0]>=loc+len(text)),None)
    if term is None:raise ValueError('No final assistant turn EOS')
    positions.append(term-1);relative.append((len(text),len(text)))
    um=[0.]*len(enc['input_ids'])
    # User token locations only, preceding the generated response. A semantic
    # history representation may encode earlier assistant turns contextually.
    cursor=0
    for m in messages:
        start=prefix.find(m['content'],cursor)
        if start<0:raise ValueError('Cannot locate history message')
        end=start+len(m['content']);cursor=end
        if m['role']=='user':
            for i,(a,b) in enumerate(offsets):
                if a<end and b>start and b<=loc:um[i]=1.
    if not any(um):raise ValueError('No user tokens for history gate')
    return {'ids':enc['input_ids'],'positions':positions,'offsets':relative,'user_mask':um,'text':text,'messages':messages}

def span_mask(seq,spans,termination=False):
    vals=[float(any(a<d and b>c for c,d in spans)) for a,b in seq['offsets']]
    if termination:vals[-1]=1.
    return torch.tensor(vals,dtype=torch.float32)

class Scorer:
    def __init__(self,model,tok,gate):
        self.model=model;self.tok=tok;self.gate=gate;self.tokens={};self.reference={}
        self.cache=ROOT/'data/reference_cache';self.cache.mkdir(parents=True,exist_ok=True)

    def sequence(self,messages,text):
        k=key([messages,text])
        if k not in self.tokens:self.tokens[k]=tokenize(self.tok,messages,text)
        return self.tokens[k]

    def lp(self,seq,reference=False):
        k=key(['official-base',seq['ids'],seq['positions']])
        if reference:
            if k in self.reference:return self.reference[k].to(self.model.device)
            p=self.cache/(k+'.json')
            if p.exists():
                out=torch.tensor(json.loads(p.read_text())['token_logps'],dtype=torch.float32)
                self.reference[k]=out;return out.to(self.model.device)
        ids=torch.tensor([seq['ids']],device=self.model.device)
        pos=torch.tensor(seq['positions'],device=ids.device)
        target=ids[0,pos+1]
        if self.gate:self.gate.reset(torch.tensor([seq['user_mask']],device=ids.device))
        training=self.model.training
        cm=self.model.disable_adapter() if reference else contextlib.nullcontext()
        with cm, torch.no_grad() if reference else contextlib.nullcontext():
            if reference:self.model.eval()
            logits=self.model(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False,logits_to_keep=pos).logits[0]
            if logits.shape[0]!=len(pos):raise ValueError('Model ignored selected logit positions')
            out=-F.cross_entropy(logits.float(),target,reduction='none')
        if reference:
            self.model.train(training)
            dump(p,{'token_logps':out.cpu().tolist(),'sequence_hash':k})
            self.reference[k]=out.cpu()
        return out

    def view(self,v):
        seqs=[self.sequence(v['messages'],s) for s in v['responses']]
        refs=[self.lp(s,True) for s in seqs]
        policies=[];gates=[]
        for s in seqs:
            policies.append(self.lp(s))
            if self.gate:gates.append(self.gate.current)
        return seqs,policies,refs,gates

    def sft(self,messages,text):
        return -self.lp(self.sequence(messages,text)).mean()

def objective(row, objective_name, scorer):
    views = [scorer.view(v) for v in row['views']]
    sums = [torch.stack([p.sum() for p in v[1]]) for v in views]
    refs = [torch.stack([p.sum() for p in v[2]]) for v in views]
    extra = {}
    if objective_name == 'P_rev':
        pref = torch.stack([dpo(p, r) for p, r in zip(sums, refs)]).mean()
        sl = sums[0][0] - sums[0][1]
        # Action view responses are intentionally stored in reversed order.
        sa = sums[1][1] - sums[1][0]
        hinge = reversal(sl, sa)
        extra['need_interaction'] = float((sl - sa).detach())
        return pref + 0.1 * hinge, extra
    pref = dpo(sums[0], refs[0])
    loss = pref
    if objective_name == 'P_req':
        seq = views[0][0][0]
        p = views[0][1][0]
        mask = span_mask(seq, row['goal_spans']).to(p.device)
        if not mask.any():
            raise ValueError('Goal span has no actual tokens')
        loss = loss - 0.1 * (p * mask).sum() / mask.sum()
        extra['goal_tokens'] = float(mask.sum())
    elif objective_name == 'specialist_piE':
        original = row['views'][0]['messages']
        teacher = row['teacher_messages']
        student, teach = [], []
        for s in row['responses_all']:
            teach.append(scorer.lp(scorer.sequence(teacher, s), True).mean())
            match = next((i for i, t in enumerate(row['views'][0]['responses']) if t == s), None)
            student.append(views[0][1][match].mean() if match is not None else scorer.lp(scorer.sequence(original, s)).mean())
        kl = distill(torch.stack(student), torch.stack(teach))
        loss = loss + 0.1 * kl
        extra['teacher_kl'] = float(kl.detach())
    return loss, extra


def checkpoint(out, model, gate, optimizer, scheduler, state):
    tmp = out / 'checkpoint-writing'
    tmp.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(tmp)
    torch.save({
        'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(), 'state': state,
        'python_rng': random.getstate(), 'numpy_rng': np.random.get_state(),
        'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all(),
    }, tmp / 'training_state.pt')
    target = out / 'checkpoint-last'
    import shutil
    prior = out / 'checkpoint-prior'
    if prior.exists():
        shutil.rmtree(prior)
    if target.exists():
        target.rename(prior)
    tmp.rename(target)
    if prior.exists():
        shutil.rmtree(prior)


def main():
    ap = argparse.ArgumentParser(description='Train Empathy Specialist (pi_E) or individual structured Stage-1 objectives.')
    ap.add_argument('--objective', default='specialist_piE',
                    choices=['specialist_piE', 'P_rev', 'P_need', 'P_warm', 'P_req', 'P_acc'])
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--base', default=_CFG.base_model)
    ap.add_argument('--data', type=Path, default=None)
    ap.add_argument('--out-dir', type=Path, default=None)
    ap.add_argument('--max-steps', type=int, default=0)
    args = ap.parse_args()

    data_path = args.data or (ROOT / 'data/objectives' / f'{args.objective}.jsonl')
    rows = read(data_path)
    assert rows, f'No rows loaded from {data_path}'
    run_name = f'piE_s{args.seed}' if args.objective == 'specialist_piE' else f'{args.objective}_s{args.seed}'
    out = args.out_dir or (ROOT / 'runs' / ('smoke' if args.max_steps else 'main') / run_name)
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'COMPLETE.json').exists():
        complete = json.loads((out / 'COMPLETE.json').read_text())
        assert complete['observed_updates'] == complete['expected_updates']
        assert digest(out / 'final/adapter_model.safetensors') == complete['adapter_sha256']
        print('Already complete; artifact and update count validated')
        return
    model, tok, gate = load_train(args)
    score = Scorer(model, tok, gate)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=5e-5, weight_decay=.1)
    updates = 2 * math.ceil(len(rows) / 16)
    smoke = bool(args.max_steps)
    if smoke:
        updates = min(updates, args.max_steps)
    warmup = max(1, math.ceil(updates * .05))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: (s + 1) / warmup if s < warmup else max(0., .5 * (1 + math.cos(math.pi * (s - warmup) / max(1, updates - warmup))))
    )
    state = {'update': 0, 'epoch': 0, 'offset': 0}
    cp = out / 'checkpoint-last'
    if cp.exists():
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file
        set_peft_model_state_dict(model, load_file(cp / 'adapter_model.safetensors'))
        saved = torch.load(cp / 'training_state.pt', weights_only=False, map_location='cpu')
        optimizer.load_state_dict(saved['optimizer'])
        scheduler.load_state_dict(saved['scheduler'])
        state = saved['state']
        random.setstate(saved['python_rng']); np.random.set_state(saved['numpy_rng'])
        torch.set_rng_state(saved['torch_rng']); torch.cuda.set_rng_state_all(saved['cuda_rng'])
    config = {
        'objective': args.objective, 'seed': args.seed, 'label': run_name, 'rows': len(rows),
        'parents': len({(r['source'], r['parent_id']) for r in rows}), 'expected_updates': updates,
        'smoke_only': smoke, 'data_sha256': digest(data_path),
        'trainable_parameters': sum(p.numel() for p in params), 'base': args.base,
        'code_sha256': {p.name: digest(p) for p in (Path(__file__).resolve(), Path(__file__).with_name('losses.py'))},
        'versions': {n: __import__('importlib.metadata', fromlist=['version']).version(n) for n in ('torch', 'transformers', 'peft')},
    }
    if (out / 'CONFIG.json').exists():
        prior = json.loads((out / 'CONFIG.json').read_text())
        for k in ('data_sha256', 'expected_updates', 'base', 'smoke_only', 'code_sha256'):
            assert prior[k] == config[k], f'Resume configuration changed: {k}'
    dump(out / 'CONFIG.json', config)
    for r in rows:
        for v in r['views']:
            for t in v['responses']:
                score.sequence(v['messages'], t)
        if args.objective == 'specialist_piE':
            for t in r['responses_all']:
                score.sequence(r['teacher_messages'], t)
    start = time.monotonic()
    for epoch in range(state['epoch'], 2):
        order = list(range(len(rows)))
        random.Random(args.seed + epoch).shuffle(order)
        begin = state['offset'] if epoch == state['epoch'] else 0
        for offset in range(begin, len(order), 16):
            ids = order[offset:offset + 16]
            optimizer.zero_grad(set_to_none=True)
            metrics = []
            for i in ids:
                r = rows[i]
                loss, extra = objective(r, args.objective, score)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f'Nonfinite loss {r["id"]}')
                (loss / len(ids)).backward()
                metrics.append({'loss': float(loss.detach()), **extra})
            norm = float(torch.nn.utils.clip_grad_norm_(params, 1.))
            if not math.isfinite(norm):
                raise FloatingPointError('Nonfinite gradient')
            optimizer.step(); scheduler.step()
            state.update(update=state['update'] + 1, epoch=epoch, offset=offset + len(ids))
            record = {'update': state['update'], 'total': updates, 'epoch': epoch, 'offset': state['offset'],
                      'loss': sum(m['loss'] for m in metrics) / len(metrics),
                      'gradient_norm': norm, 'lr': scheduler.get_last_lr()[0], 'seconds': time.monotonic() - start,
                      'details': metrics, 'gpu_peak_bytes': torch.cuda.max_memory_allocated()}
            with (out / 'train.jsonl').open('a') as f:
                f.write(json.dumps(record) + '\n')
            print(json.dumps({k: v for k, v in record.items() if k != 'details'}), flush=True)
            if state['update'] % 10 == 0 or state['update'] == updates:
                checkpoint(out, model, gate, optimizer, scheduler, state)
            if state['update'] >= updates:
                break
        if state['update'] >= updates:
            break
        state.update(epoch=epoch + 1, offset=0)
    assert state['update'] == updates
    if smoke:
        dump(out / 'SMOKE_COMPLETE.json', {**config, 'observed_updates': state['update']})
    else:
        model.save_pretrained(out / 'final'); tok.save_pretrained(out / 'final')
        dump(out / 'COMPLETE.json', {**config, 'observed_updates': state['update'], 'state': state,
             'adapter_sha256': digest(out / 'final/adapter_model.safetensors')})


if __name__ == '__main__':
    main()
