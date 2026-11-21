"""ESCRAG-R1 retrieval-augmented GRPO, stage 1: initial responses + retrieval for every RL prompt.

The released trainer (verl fork) rolls out twice per step: the current policy writes an initial <think>/<response> output for the dialogue
history, the full output is embedded with bge-m3 and the top-3 chunks of the retrieval corpus are fetched, then the policy writes the final
response conditioned on (history, initial response, chunks); only that second generation is rewarded and updated. ms-swift cannot chain two
generations inside one GRPO step, so the first pass is done once here with the frozen cold-start policy (temperature 0.9, top-p 0.95, the
paper's sampling) and the second pass is trained with GRPO on the resulting fixed second-stage prompts. Deviation: the initial responses come from
the cold-start policy, not from the live policy.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[2]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
from baselines.external.esc_repro_common import format_reward, normalize_output, read_jsonl, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--family", choices=("qwen25", "qwen35"), required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20260930)
    parser.add_argument("--thinking", action="store_true", help="Qwen3.5 native thinking mode (the cold-start SFT was trained through the chat template, which puts the <think> block in the generation prompt)")
    args = parser.parse_args()

    from vllm import LLM, SamplingParams

    rows = read_jsonl(args.prompts)
    llm = LLM(model=str(args.policy), dtype="bfloat16", max_model_len=6144, gpu_memory_utilization=0.5, seed=args.seed,
              limit_mm_per_prompt={"image": 0, "video": 0} if args.family == "qwen35" else None)
    params = SamplingParams(temperature=args.temperature, top_p=args.top_p, max_tokens=args.max_tokens, seed=args.seed)
    outputs = llm.chat([r["messages"] for r in rows], params, chat_template_kwargs={"enable_thinking": args.thinking})
    initial = [normalize_output(o.outputs[0].text.strip()) for o in outputs]
    del llm

    from escrag_retrieval import BgeM3, Retriever, rag_messages

    retriever = Retriever(BgeM3("cuda"))
    hits = retriever.top(initial, n=3)
    result = []
    for row, text, chunks in zip(rows, initial, hits):
        result.append({"id": row["id"], "parent_id": row["parent_id"], "history": row["history"], "initial_response": text,
                       "messages": rag_messages(row["history"], text, chunks)})
    write_jsonl(args.out, result)
    stats = {"rows": len(result), "initial_format_ok": sum(format_reward(t) for t in initial) / len(initial),
             "mean_initial_chars": sum(map(len, initial)) / len(initial),
             "mean_prompt_chars": sum(len(r["messages"][1]["content"]) for r in result) / len(result)}
    print(json.dumps(stats), flush=True)


if __name__ == "__main__":
    main()
