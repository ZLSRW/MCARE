"""Blind test-time evaluation judge (Phase B semantic alignment + independent empathy expectation).

Phase B (greedy, temperature 0, structured JSON): receives frozen dialogue history,
verified Phase-A requirements, and one candidate reply with model names and gold
references hidden -> need_judge_labels_g.jsonl.
Independent empathy (1-5): first-token expectation over digits 1-5 -> empathy_labels_g.jsonl.
Defaults to the held-out cross-family judge (Gemma-4-31B-it).
usage: PYTHONPATH=src python src/evaluation/run_need_judge.py --labels Base MCARE ...
"""
import argparse
import json
import os
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
from common import prompts_and_schemas as C  # noqa: E402


def load_llm(gpu_util, model=None):
    from vllm import LLM
    kw = dict(model=model or C.GEMMA_MODEL, dtype="bfloat16", max_model_len=8192, gpu_memory_utilization=gpu_util, seed=0, enable_prefix_caching=True, max_num_seqs=96)
    if int(os.environ.get("NEED_TP", "1")) > 1:                  # same weights / precision on 2 cards: more KV cache (Gemma-4-31B has only ~13k tokens of KV on one 80 GB card)
        kw["tensor_parallel_size"] = int(os.environ["NEED_TP"])
        kw["disable_custom_all_reduce"] = True                  # custom all-reduce failed with "invalid argument" (CUDA IPC); NCCL over NVLink is used instead
    try:
        return LLM(limit_mm_per_prompt={"image": 0, "video": 0}, **kw)
    except Exception as e:                              # another model family may not know these multimodal limit keys
        print("load with multimodal limits failed (%r); retrying without" % (e,), flush=True)
        return LLM(**kw)


def audit_of(gid, gen_row, audit, pairs_by_d1):
    """Phase-A evidence for a generation row: M2 D1 rows use the evidence of the underlying history."""
    base_id = pairs_by_d1.get(gid, gid)
    a = audit.get(base_id)
    return a["verified"] if a and a["verified"] and a["flags"] and a["flags"]["eligible_strict"] else None       # only histories of D_need are judged


def build_items(labels, sets, audit_suffix=""):
    gen_inputs = {r["id"]: r for r in C.read_jsonl(C.EVAL / "generation_inputs.jsonl")}
    audit = {r["id"]: r for r in C.read_jsonl(C.EVAL / f"need_audit_raw{audit_suffix}.jsonl")}
    pairs = C.read_jsonl(C.EVAL / "m2_pairs.jsonl")
    d1_to_base = {p["d1_id"]: p["history_id"] for p in pairs}
    d1_to_base.update({p["d1l_id"]: p["history_id"] for p in pairs if "d1l_id" in p})
    items = []
    for label in labels:
        for r in C.read_jsonl(C.EVAL / "gen" / f"{label}.jsonl"):
            g = gen_inputs[r["id"]]
            if not set(g["sets"]) & set(sets):
                continue
            if label.startswith("PROBE_") and not set(g["sets"]) & {"esconv_main", "m2_d1"}:       # known-answer probes need only the main histories and the disclosure histories
                continue
            ver = audit_of(r["id"], g, audit, d1_to_base)
            items.append({"label": label, "id": r["id"], "parent_id": g["parent_id"], "sets": g["sets"], "messages": g["messages"], "reply": r["response"], "audit": ver})
    return items


def judge_prompt(it):
    ann = C.render_annotations(it["audit"])
    return [{"role": "system", "content": C.JUDGE_SYSTEM},
            {"role": "user", "content": C.JUDGE_USER.format(dialogue=C.render_dialogue(it["messages"], mark_latest=False), annotations=ann, reply=it["reply"].strip() or "(empty reply)")}]


def second_sample(items, frac):
    keep = set()
    for it in items:
        if C.hash_rank("second", it["label"], it["id"]) % 1000 < int(frac * 1000):
            keep.add((it["label"], it["id"]))
    return keep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", nargs="+", required=True)
    ap.add_argument("--sets", nargs="+", default=["esconv_main", "esconv_sens", "ed_main", "m2_d0", "m2_d1", "m2_d1l"])
    ap.add_argument("--second", action="store_true")
    ap.add_argument("--second-frac", type=float, default=0.2)
    ap.add_argument("--skip-empathy", action="store_true")
    ap.add_argument("--skip-primary", action="store_true")
    ap.add_argument("--gpu-util", type=float, default=0.90)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--audit-suffix", default="_g", help="suffix of need_audit_raw*.jsonl (default: _g)")
    ap.add_argument("--judge-model", default=None, help="evaluation judge model (default: MCARE_EVAL_JUDGE_MODEL / Gemma-4-31B-it)")
    ap.add_argument("--suffix", default="_g", help="output filename suffix (default: _g)")
    ap.add_argument("--second-validation", action="store_true", help="also run the thinking-mode second pass on every hand-labelled validation item")
    ap.add_argument("--subset-second", action="store_true", help="judge only the pre-fixed second-pass sample and the hand-labelled validation items (cross-family check)")
    args = ap.parse_args()
    items = build_items(args.labels, args.sets, args.audit_suffix)
    if args.limit:
        items = items[:args.limit]
    ph = C.prompt_hash("judge")
    sfx = args.suffix
    prim_path, emp_path, sec_path = C.EVAL / f"need_judge_labels{sfx}.jsonl", C.EVAL / f"empathy_labels{sfx}.jsonl", C.EVAL / f"need_judge_labels_second{sfx}.jsonl"
    if args.subset_second:
        keep = second_sample([it for it in items if it["audit"] is not None], args.second_frac)
        for fn in ("judge_random_sample.json", "judge_sample.json"):
            vs = C.EVAL / "validation" / fn
            if vs.exists():
                keep |= {tuple(x["key"].split("|", 1)) for x in json.load(open(vs))}
        items = [it for it in items if (it["label"], it["id"]) in keep]
    done = lambda p: {(r["label"], r["id"]) for r in C.read_jsonl(p)} if p.exists() else set()   # noqa: E731
    llm = None
    from vllm import SamplingParams
    from vllm.sampling_params import StructuredOutputsParams

    # ------------------------------------------------ primary Phase-B judgement (needs an audit record)
    if not args.skip_primary:
        have = done(prim_path)
        todo = [it for it in items if (it["label"], it["id"]) not in have and it["audit"] is not None]
        print("primary todo", len(todo), "missing audit", sum(it["audit"] is None for it in items), flush=True)
        if todo:
            llm = llm or load_llm(args.gpu_util, args.judge_model)
            convs = [judge_prompt(it) for it in todo]
            params = [SamplingParams(temperature=0.0, max_tokens=700, structured_outputs=StructuredOutputsParams(json=C.judge_schema(len(it["audit"]["positive_need"]), len(it["audit"]["negative_need"])))) for it in todo]
            outs = llm.chat(convs, params, chat_template_kwargs={"enable_thinking": False})
            with C.locked_append(prim_path) as handle:
                for it, o in zip(todo, outs):
                    text = o.outputs[0].text
                    p = C.parse_json(text)
                    handle.write(json.dumps({"label": it["label"], "id": it["id"], "parent_id": it["parent_id"], "raw": text, "parse_ok": p is not None, "judgement": p, "prompt_hash": ph}, ensure_ascii=False) + "\n")
    # ------------------------------------------------ independent empathy
    if not args.skip_empathy:
        have = done(emp_path)
        todo = [it for it in items if (it["label"], it["id"]) not in have and set(it["sets"]) & {"esconv_main", "ed_main"} and not it["label"].startswith("PROBE_")]
        print("empathy todo", len(todo), flush=True)
        if todo:
            llm = llm or load_llm(args.gpu_util, args.judge_model)
            convs = [[{"role": "system", "content": C.EMP_SYSTEM}, {"role": "user", "content": C.EMP_USER.format(dialogue=C.render_dialogue(it["messages"], mark_latest=False), reply=it["reply"].strip() or "(empty reply)")}] for it in todo]
            outs = llm.chat(convs, SamplingParams(temperature=0.0, max_tokens=1, logprobs=20), chat_template_kwargs={"enable_thinking": False})
            with C.locked_append(emp_path) as handle:
                for it, o in zip(todo, outs):
                    lp = o.outputs[0].logprobs[0]
                    top = {v.decoded_token: v.logprob for v in lp.values()}
                    score, mass = C.digit_expectation(top)
                    handle.write(json.dumps({"label": it["label"], "id": it["id"], "parent_id": it["parent_id"], "empathy": score, "digit_mass": mass, "prompt_hash": ph}) + "\n")
    # ------------------------------------------------ second pass (thinking mode)
    if args.second:
        prim = {(r["label"], r["id"]): r for r in C.read_jsonl(prim_path)} if prim_path.exists() else {}
        keep = second_sample([it for it in items if it["audit"] is not None], args.second_frac)
        keep |= {k for k, r in prim.items() if (r["judgement"] or {}).get("uncertain") or not r["parse_ok"]}
        if args.second_validation:
            for fn in ("judge_random_sample.json", "judge_sample.json"):
                vs = C.EVAL / "validation" / fn
                if vs.exists():
                    keep |= {tuple(x["key"].split("|", 1)) for x in json.load(open(vs))}
        have = done(sec_path)
        todo = [it for it in items if (it["label"], it["id"]) in keep and (it["label"], it["id"]) not in have and it["audit"] is not None]
        print("second todo", len(todo), flush=True)
        if todo:
            llm = llm or load_llm(args.gpu_util, args.judge_model)
            convs = [judge_prompt(it) for it in todo]
            outs = llm.chat(convs, SamplingParams(temperature=0.6, top_p=0.95, top_k=20, max_tokens=3500, seed=0, skip_special_tokens=not ("gemma" in (args.judge_model or "").lower())), chat_template_kwargs={"enable_thinking": True})
            with C.locked_append(sec_path) as handle:
                for it, o in zip(todo, outs):
                    text = o.outputs[0].text
                    final = C.final_answer(text)          # an unfinished reasoning trace has no verdict
                    p = C.parse_json(final)
                    handle.write(json.dumps({"label": it["label"], "id": it["id"], "parent_id": it["parent_id"], "raw_final": final[-1500:], "parse_ok": p is not None, "judgement": p, "finish": o.outputs[0].finish_reason, "prompt_hash": ph}, ensure_ascii=False) + "\n")
    print("done", flush=True)


if __name__ == "__main__":
    main()
