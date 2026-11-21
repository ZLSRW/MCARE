"""Offline candidate scorers for MCARE Stage 1 and Stage 2 pair construction:
  - empathy:     Qwen3.8-27B first-token expectation e_i in [1, 5]
  - helpfulness: Qwen3.8-27B first-token expectation h_i in [1, 5] (or HelpSteer2 critic)
  - support:     Support-utility reward model probability o_i in [0, 1]
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from common import prompts_and_schemas as P
from common.config import get_paths
from common.io_utils import dump_json, key, read_jsonl, sha256_file, write_jsonl

TRAIN_EMP_SYSTEM = (
    "You are an offline training annotator rating emotional attunement and warmth in a "
    "supportive dialogue reply. Output only a single integer digit from 1 to 5."
)
TRAIN_EMP_USER = (
    "[Dialogue Context]\n{dialogue}\n\n[Candidate Supporter Reply]\n{reply}\n\n"
    "Rate how emotionally attuned, compassionate, and responsive to the user's feelings "
    "this training candidate is, from 1 (cold, generic, or dismissive) to 5 (deeply "
    "empathetic, specific, and emotionally validating).\nScore (1-5):"
)
HELP_SYSTEM = (
    "You evaluate how helpful, coherent, accurate, and directly useful a supportive reply is "
    "to the user's conversation. Respond with a single integer digit from 1 to 5 and nothing else."
)
HELP_USER = (
    "[Conversation]\n{dialogue}\n\n[Supporter Reply]\n{reply}\n\n"
    "Rate overall helpfulness and response quality from 1 (unhelpful, off-topic, or confused) "
    "to 5 (clear, coherent, accurate, and directly helpful).\nRating (1-5):"
)
SUPPORT_RM_PREFIX = (
    "Instruction: You are a helpful assistant tasked with evaluating whether a patient's emotional problem "
    "has been effectively addressed following a conversation with a therapist.\n"
)


def _score_logprob_expectation(
    candidates_path: Path,
    out_path: Path,
    system_prompt: str,
    user_template: str,
    score_key: str,
    workers: int,
) -> None:
    cfg = get_paths()
    url = cfg.train_annotator_url + "/chat/completions"
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {cfg.train_annotator_api_key}",
    }
    done = {json.loads(line)["candidate_id"] for line in out_path.open(encoding="utf-8") if line.strip()} if out_path.exists() else set()
    rows = read_jsonl(candidates_path)
    flat = [(r, c) for r in rows for c in r["candidates"] if c["candidate_id"] not in done]
    print(f"{score_key} todo={len(flat)} done={len(done)}", flush=True)

    def _one(messages: list[dict[str, Any]], reply: str) -> tuple[float | None, float]:
        body = {
            "model": cfg.train_annotator_served_name,
            "temperature": 0,
            "max_tokens": 1,
            "logprobs": True,
            "top_logprobs": 20,
            "chat_template_kwargs": {"enable_thinking": False},
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": user_template.format(
                        dialogue=P.render_dialogue(messages, mark_latest=False),
                        reply=reply.strip() or "(empty reply)",
                    ),
                },
            ],
        }
        data = json.dumps(body).encode("utf-8")
        for attempt in range(5):
            try:
                req = urllib.request.Request(url, data=data, headers=headers)
                with opener.open(req, timeout=300) as resp:
                    res = json.load(resp)
                top = {x["token"]: x["logprob"] for x in res["choices"][0]["logprobs"]["content"][0]["top_logprobs"]}
                return P.digit_expectation(top)
            except Exception:
                if attempt == 4:
                    raise
                time.sleep(2**attempt)
        return None, 0.0

    out_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    count = 0
    with out_path.open("a", encoding="utf-8") as handle, concurrent.futures.ThreadPoolExecutor(workers) as pool:
        for (r, c), (val, mass) in zip(flat, pool.map(lambda rc: _one(rc[0]["messages"], rc[1]["text"]), flat)):
            handle.write(
                json.dumps({"candidate_id": c["candidate_id"], "id": r["id"], score_key: val, "digit_mass": mass}) + "\n"
            )
            count += 1
            if count % 500 == 0:
                handle.flush()
                print(json.dumps({"metric": score_key, "done": len(done) + count, "elapsed": round(time.monotonic() - t0)}), flush=True)

    dump_json(
        out_path.with_name(out_path.stem + "_COMPLETE.json"),
        {
            "metric": score_key,
            "annotator": f"{cfg.train_annotator_model} (training-only)",
            "rows": len(done) + count,
            "sha256": sha256_file(out_path),
        },
    )


def score_support_rm(candidates_path: Path, out_path: Path, model_path: Path, batch_size: int = 16) -> None:
    """Score problem-relief probability p_E using the training-split support reward model."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    tokenizer.padding_side = "right"
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, local_files_only=True
    ).to("cuda").eval()
    model.config.pad_token_id = tokenizer.pad_token_id

    rows = read_jsonl(candidates_path)
    encoded, owners = [], []
    for row in rows:
        turns = [m for m in row["messages"] if m["role"] != "system"]
        assert turns[-1]["role"] == "user"
        for candidate in row["candidates"]:
            retained = turns[:]
            while True:
                context = "\n".join(
                    f"{'Patient' if t['role'] == 'user' else 'Therapist'}: {t['content'].strip()}"
                    for t in retained
                )
                tokens = tokenizer(
                    f"{SUPPORT_RM_PREFIX}Conversation: {context.strip()}\nTherapist's response: {candidate['text'].strip()}"
                )["input_ids"]
                if len(tokens) <= 2048:
                    break
                assert len(retained) > 1
                retained = retained[1:]
            encoded.append(tokens)
            owners.append((row["id"], candidate["candidate_id"], key(candidate["text"])))

    order = sorted(range(len(encoded)), key=lambda i: len(encoded[i]))
    probs: list[float] = [0.0] * len(encoded)
    for start in range(0, len(order), batch_size):
        idx = order[start : start + batch_size]
        batch = tokenizer.pad({"input_ids": [encoded[i] for i in idx]}, padding=True, return_tensors="pt").to("cuda")
        with torch.inference_mode():
            p = torch.softmax(model(**batch).logits.float(), dim=-1)[:, 1].cpu().tolist()
        for i, val in zip(idx, p):
            assert math.isfinite(val) and 0.0 <= val <= 1.0
            probs[i] = float(val)

    out, cursor = [], 0
    for row in rows:
        scores = []
        for candidate in row["candidates"]:
            scores.append(
                {
                    "candidate_id": candidate["candidate_id"],
                    "candidate_text_sha256": owners[cursor][2],
                    "p_E": probs[cursor],
                }
            )
            cursor += 1
        out.append({"id": row["id"], "source_row_sha256": key(row), "scores": scores})
    write_jsonl(out_path, out)
    dump_json(
        out_path.with_name(out_path.stem + "_COMPLETE.json"),
        {"metric": "support", "rows": len(out), "sha256": sha256_file(out_path)},
    )


def main() -> None:
    cfg = get_paths()
    parser = argparse.ArgumentParser(description="Score Stage-1/2 candidate responses.")
    parser.add_argument("--metric", choices=["empathy", "helpfulness", "support"], required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=cfg.train_workers)
    parser.add_argument("--support-rm", type=Path, default=Path(cfg.train_support_rm))
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    if args.metric == "empathy":
        _score_logprob_expectation(args.candidates, args.out, TRAIN_EMP_SYSTEM, TRAIN_EMP_USER, "empathy", args.workers)
    elif args.metric == "helpfulness":
        _score_logprob_expectation(args.candidates, args.out, HELP_SYSTEM, HELP_USER, "helpfulness", args.workers)
    else:
        score_support_rm(args.candidates, args.out, args.support_rm, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
