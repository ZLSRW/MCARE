"""Greedy non-thinking evaluation response generator for MCARE and baseline policies.

Decodes one reply per frozen history in generation_inputs.jsonl under a shared
chat template and system prompt (temperature=0, max_new_tokens=768).
usage: PYTHONPATH=src python src/evaluation/generate_responses.py --model DIR --out gen/<label>.jsonl [--engine vllm|hf] [--strong-prompt] [--max-tokens 768]
"""
import argparse
import json
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
from common import prompts_and_schemas as C  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--engine", choices=["vllm", "hf"], default="vllm")
    ap.add_argument("--strong-prompt", action="store_true")
    ap.add_argument("--max-tokens", type=int, default=768)
    ap.add_argument("--gpu-util", type=float, default=0.35)
    ap.add_argument("--kv-gib", type=float, default=5.0)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--inputs", default=str(C.EVAL / "generation_inputs.jsonl"))
    ap.add_argument("--panel", default=str(C.EVAL / "panel_all.jsonl"))
    ap.add_argument("--check-missing", action="store_true", help="print the number of histories without a reply in --out and exit")
    args = ap.parse_args()
    all_rows = C.read_jsonl(args.inputs)
    panel = {r["id"]: r for r in C.read_jsonl(args.panel)}
    have = {r["id"]: r for r in C.read_jsonl(args.out)} if Path(args.out).exists() else {}      # greedy decoding is deterministic per history: replies that exist are kept (incremental top-up after a change of the input set)
    rows = [r for r in all_rows if r["id"] not in have]
    if args.check_missing:
        print(len(rows))
        return
    if not rows:
        with open(args.out, "w", encoding="utf-8") as handle:
            for r in all_rows:
                handle.write(json.dumps(have[r["id"]], ensure_ascii=False) + "\n")
        print("nothing to generate", flush=True)
        return
    convs = [C.with_strong_prompt(r["messages"]) if args.strong_prompt else r["messages"] for r in rows]
    recs = []
    if args.engine == "vllm":
        from vllm import LLM, SamplingParams
        llm = LLM(model=args.model, dtype="bfloat16", max_model_len=6144, gpu_memory_utilization=args.gpu_util, seed=0, max_num_seqs=64, limit_mm_per_prompt={"image": 0, "video": 0},
                  **({"kv_cache_memory_bytes": int(args.kv_gib * 2**30)} if args.kv_gib > 0 else {}))
        outs = llm.chat(convs, SamplingParams(temperature=0.0, max_tokens=args.max_tokens, seed=0), chat_template_kwargs={"enable_thinking": False})
        for r, o in zip(rows, outs):
            c = o.outputs[0]
            recs.append({"response": c.text.strip(), "hit_cap": c.finish_reason == "length", "generated_tokens": len(c.token_ids), "finish_reason": c.finish_reason})
    else:
        from common.hf_worker import generate_batch as hf_generate_batch, load as hf_load
        model, tokenizer = hf_load(args.model)
        for s in range(0, len(convs), args.batch_size):
            for res in hf_generate_batch(model, tokenizer, convs[s:s + args.batch_size], max_tokens=args.max_tokens):
                recs.append({"response": res["text"], "hit_cap": bool(res["hit_cap"]), "generated_tokens": res["tokens"], "finish_reason": res["finish_reason"]})
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    new = {r["id"]: {"id": r["id"], "panel": panel[r["id"]]["panel"], "family": panel[r["id"]]["family"], **rec} for r, rec in zip(rows, recs)}
    with open(args.out, "w", encoding="utf-8") as handle:
        for r in all_rows:
            handle.write(json.dumps(new.get(r["id"]) or have[r["id"]], ensure_ascii=False) + "\n")
    n_empty = sum(not x["response"] for x in recs)
    print(f"wrote {args.out}: {len(recs)} replies, hit_cap {sum(x['hit_cap'] for x in recs)}, empty {n_empty}", flush=True)


if __name__ == "__main__":
    main()
