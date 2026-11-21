"""Offline candidate compliance & warmth annotator (Qwen3.8-27B).

Evaluates K candidates independently against the visible dialogue history and
explicit training requirements at temperature 0, producing:
  - Five binary violation indicators: request_violation, boundary_violation,
    fact_violation, identity_violation, fabrication
  - Basic empathy warmth rating w_hat in {1, ..., 5}
  - Exact verbatim candidate/user spans for every positive violation flag

Uses the configured Qwen3.8-27B OpenAI-compatible endpoint.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from common.config import get_paths
from common.io_utils import dump_json, key, read_jsonl, sha256_file

PROMPT = """Evaluate every candidate independently as the next supportive reply to
the visible conversation. These are TRAINING examples; no test outputs or test
labels are supplied. Rate warmth 1 (cold/dismissive) to 5 (specific, sensitive,
emotionally attuned). Length, stock reassurance and unsolicited advice do not by
themselves improve warmth. Respect an explicit request for a draft or factual answer.
For request_violation, require an actual explicit request addressed to the assistant
that remains applicable; ordinary feelings, small talk and thanks are not requests.
For boundary_violation, require an expressed prohibition or supplied narrow feedback
target; a failed attempt does not ban all advice. fact_violation means contradicting
a fact actually stated in the visible history, not failing to mention it.
identity_violation means assuming personal traits or needs just from a group label;
acknowledging a disclosed identity without stereotype is allowed.
fabrication means the assistant claims real personal life experiences or a human
identity, not first-person words inside a requested message draft.
Return each index exactly once. For any violation, copy a short EXACT candidate
substring into candidate_quote and the corresponding EXACT user evidence into
user_quote (fabrication alone needs only candidate_quote). Do not invent evidence.
If no violation, both quotes must be empty. No reward scores are shown.
HISTORY:
{history}
APPLICABLE ADDITIONAL TRAINING REQUIREMENT:
{requirement}
CANDIDATES:
{candidates}
"""

BOOLS = ["request_violation", "boundary_violation", "fact_violation", "identity_violation", "fabrication"]
_PROPS = {
    "index": {"type": "integer", "minimum": 0, "maximum": 7},
    "warmth": {"type": "integer", "minimum": 1, "maximum": 5},
    **{k: {"type": "boolean"} for k in BOOLS},
    "candidate_quote": {"type": "string"},
    "user_quote": {"type": "string"},
}
SCHEMA = {
    "type": "object",
    "properties": {
        "checks": {
            "type": "array",
            "minItems": 8,
            "maxItems": 8,
            "items": {"type": "object", "properties": _PROPS, "required": list(_PROPS), "additionalProperties": False},
        }
    },
    "required": ["checks"],
    "additionalProperties": False,
}
ALIASES = {
    "i": "index",
    "w": "warmth",
    "r": "request_violation",
    "b": "boundary_violation",
    "f": "fact_violation",
    "d": "identity_violation",
    "a": "fabrication",
    "c": "candidate_quote",
    "h": "user_quote",
}
COMPACT_SCHEMA = {
    "type": "object",
    "properties": {
        "checks": {
            "type": "array",
            "minItems": 8,
            "maxItems": 8,
            "items": {
                "type": "object",
                "properties": {short: _PROPS[long] for short, long in ALIASES.items()},
                "required": list(ALIASES),
                "additionalProperties": False,
            },
        }
    },
    "required": ["checks"],
    "additionalProperties": False,
}
FORMAT = (
    "\nOutput compact JSON without indentation. Use these short keys with exactly the meanings above: "
    + json.dumps(ALIASES)
    + ". Boolean flags remain true/false, warmth remains 1–5, and quotes remain exact.\n"
)


def annotate_row(row: dict[str, Any], out_dir: Path) -> dict[str, Any] | None:
    cfg = get_paths()
    ann_path = out_dir / "annotations" / f"{key(row['id'])}.json"
    signature = key([PROMPT, SCHEMA, row["messages"], row["candidates"], row.get("instruction"), row.get("feedback")])
    if ann_path.exists():
        existing = json.loads(ann_path.read_text(encoding="utf-8"))
        assert existing["input_hash"] == signature
        return existing

    texts = [c["text"] for c in row["candidates"]]
    prompt = PROMPT.format(
        history=json.dumps(row["messages"], ensure_ascii=False),
        requirement=json.dumps(row.get("instruction") or row.get("feedback") or None, ensure_ascii=False),
        candidates=json.dumps([{"index": i, "text": t} for i, t in enumerate(texts)], ensure_ascii=False),
    )
    messages = [{"role": "user", "content": prompt + FORMAT}]
    endpoint = cfg.train_annotator_url + "/chat/completions"
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {cfg.train_annotator_api_key}",
    }

    last_error = None
    content = None
    for attempt in range(4):
        body = {
            "model": cfg.train_annotator_served_name,
            "messages": messages,
            "temperature": 0,
            "max_tokens": 1800,
            "seed": 42,
            "chat_template_kwargs": {"enable_thinking": False},
            "structured_outputs": {"json": COMPACT_SCHEMA},
        }
        try:
            req = urllib.request.Request(endpoint, data=json.dumps(body).encode("utf-8"), headers=headers)
            with opener.open(req, timeout=900) as resp:
                raw = json.load(resp)
            content = raw["choices"][0]["message"]["content"]
            canonical = [{ALIASES[k]: value for k, value in c.items()} for c in json.loads(content)["checks"]]
            checks = sorted(canonical, key=lambda a: a["index"])
            assert len(checks) == len(texts) and [c["index"] for c in checks] == list(range(len(texts)))
            user_turns = [m["content"] for m in row["messages"] if m["role"] == "user"]
            for i, c in enumerate(checks):
                assert type(c["warmth"]) is int and 1 <= c["warmth"] <= 5
                assert all(type(c[b]) is bool for b in BOOLS)
                if any(c[b] for b in BOOLS):
                    assert c["candidate_quote"].strip() and c["candidate_quote"] in texts[i]
                    if any(c[b] for b in BOOLS if b != "fabrication"):
                        assert c["user_quote"].strip() and any(c["user_quote"] in s for s in user_turns)
                else:
                    assert not c["candidate_quote"] and not c["user_quote"]
            result = {
                "id": row["id"],
                "input_hash": signature,
                "checks": checks,
                "annotator": f"{cfg.train_annotator_model} (training-only)",
                "protocol_hash": key([PROMPT, SCHEMA]),
                "purpose": "training semantic gate and warmth labeling",
                "attempts": attempt + 1,
                "request_sha256": key(body),
            }
            dump_json(ann_path, result)
            return result
        except Exception as err:
            last_error = repr(err)
            dump_json(
                out_dir / "annotation_errors" / f"{key([row['id'], attempt])}.json",
                {"id": row["id"], "attempt": attempt, "error": last_error, "raw": content, "input_hash": signature},
            )
            if content is not None:
                messages = [
                    {"role": "user", "content": prompt + FORMAT},
                    {"role": "assistant", "content": content},
                    {
                        "role": "user",
                        "content": (
                            f"Validation failed ({last_error}). Return all {len(texts)} records, exact quotes, and valid indices. "
                            "Re-evaluate using the original evidence without inventing or erasing violations."
                        ),
                    },
                ]
            time.sleep(min(2**attempt, 8))

    dump_json(
        out_dir / "annotation_unknown" / f"{key(row['id'])}.json",
        {"id": row["id"], "input_hash": signature, "error": last_error},
    )
    return None


def main() -> None:
    cfg = get_paths()
    parser = argparse.ArgumentParser(description="Annotate K=8 candidate responses with Qwen3.8-27B.")
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=cfg.stage2_dir)
    parser.add_argument("--workers", type=int, default=cfg.train_workers)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    rows = read_jsonl(args.candidates)
    if args.limit:
        rows = rows[: args.limit]
    done = 0
    unknown = 0
    proto_path = args.out_dir / "ANNOTATION_PROTOCOL.json"
    dump_json(
        proto_path,
        {
            "annotator": cfg.train_annotator_model,
            "prompt": PROMPT,
            "schema": SCHEMA,
            "input_sha256": sha256_file(args.candidates),
        },
    )
    with concurrent.futures.ThreadPoolExecutor(args.workers) as pool:
        futures = [pool.submit(annotate_row, row, args.out_dir) for row in rows]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            done += 1
            unknown += result is None
            if done % 20 == 0 or done == len(rows):
                print(json.dumps({"done": done, "total": len(rows), "unknown": unknown}), flush=True)


if __name__ == "__main__":
    main()
