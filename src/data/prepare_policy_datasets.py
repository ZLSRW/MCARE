from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
from common.config import get_paths


SYSTEM_PROMPT = (
    "You are MCARE, a supportive conversational assistant. Respond with empathy, "
    "follow the user's current request, respect stated facts and corrections, and "
    "do not claim personal experiences or a human identity."
)

AUTOBIOGRAPHICAL_PATTERNS = [
    re.compile(r"\bI(?:'ve| have) (?:also )?(?:been|had|experienced|gone through|felt)\b", re.I),
    re.compile(
        r"\bI (?:also )?(?:have|had|feel|felt|suffer(?:ed)? from|struggle(?:d)? with|"
        r"lost|went through|work(?:ed)?|live(?:d)?)\b",
        re.I,
    ),
    re.compile(r"\bwhen I (?:was|had|went|lost|worked|lived|experienced)\b", re.I),
    re.compile(
        r"\bmy (?:mother|father|mom|dad|wife|husband|partner|son|daughter|child|kids|"
        r"family|friend|boss|coworker|job|school|pet|dog|cat|experience)\b",
        re.I,
    ),
]


def clean_text(value: str) -> str:
    value = value.replace("_comma_", ",")
    value = value.replace("\r\n", "\n").replace("\r", "\n")
    value = re.sub(r"[ \t]+", " ", value)
    value = re.sub(r"\n{3,}", "\n\n", value)
    return value.strip()


def stable_fraction(value: str, salt: str) -> float:
    digest = hashlib.sha256(f"{salt}\0{value}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def content_hash(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def has_autobiographical_claim(text: str) -> bool:
    return any(pattern.search(text) for pattern in AUTOBIOGRAPHICAL_PATTERNS)


def write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            count += 1
    return count


def conversation_examples(
    messages: list[dict],
    source: str,
    parent_id: str,
    metadata: dict,
) -> list[dict]:
    examples = []
    for index, message in enumerate(messages):
        if message["role"] != "assistant" or index == 0:
            continue
        history = messages[: index + 1]
        if not any(item["role"] == "user" for item in history[:-1]):
            continue
        target = message["content"]
        examples.append(
            {
                "id": f"{source}:{parent_id}:{index}",
                "source": source,
                "parent_id": parent_id,
                "messages": [{"role": "system", "content": SYSTEM_PROMPT}, *history],
                "target": target,
                "target_autobiographical_claim": has_autobiographical_claim(target),
                "metadata": dict(metadata),
            }
        )
    return examples


def prepare_ed(raw_root: Path, output_root: Path) -> dict:
    input_root = raw_root / "empathetic_dialogues" / "extracted" / "empatheticdialogues"
    result = {"splits": {}, "excluded_autobiographical": 0, "deduplicated": 0}
    excluded = []
    for split in ("train", "valid", "test"):
        path = input_root / f"{split}.csv"
        csv.field_size_limit(100_000_000)
        grouped: dict[str, list[dict]] = defaultdict(list)
        with path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                grouped[row["conv_id"]].append(row)

        output = []
        seen = set()
        for conversation_id in sorted(grouped):
            rows = sorted(grouped[conversation_id], key=lambda row: int(row["utterance_idx"]))
            if not rows:
                continue
            situation_speaker = rows[0]["speaker_idx"]
            messages = []
            for row in rows:
                role = "user" if row["speaker_idx"] == situation_speaker else "assistant"
                content = clean_text(row["utterance"])
                if not content:
                    continue
                if messages and messages[-1]["role"] == role:
                    messages[-1]["content"] += "\n" + content
                else:
                    messages.append({"role": role, "content": content})
            examples = conversation_examples(
                messages,
                source="ed",
                parent_id=conversation_id,
                metadata={
                    "emotion": rows[0].get("context", ""),
                    "situation": clean_text(rows[0].get("prompt", "")),
                    "official_split": split,
                },
            )
            for example in examples:
                signature = content_hash(example["messages"])
                if signature in seen:
                    result["deduplicated"] += 1
                    continue
                seen.add(signature)
                if split == "train" and example["target_autobiographical_claim"]:
                    excluded.append(example)
                    result["excluded_autobiographical"] += 1
                    continue
                output.append(example)
        target_path = output_root / "policy" / f"ed_{split}.jsonl"
        write_jsonl(target_path, output)
        result["splits"][split] = {
            "parents": len(grouped),
            "examples": len(output),
            "path": str(target_path),
        }
    excluded_path = output_root / "audit" / "ed_excluded_autobiographical.jsonl"
    write_jsonl(excluded_path, excluded)
    result["excluded_path"] = str(excluded_path)
    return result


def split_esconv_parents(data: list[dict], seed: int) -> dict[int, str]:
    indices = list(range(len(data)))
    random.Random(seed).shuffle(indices)
    train_end = round(len(indices) * 0.70)
    dev_end = train_end + round(len(indices) * 0.15)
    mapping = {}
    for index in indices[:train_end]:
        mapping[index] = "train"
    for index in indices[train_end:dev_end]:
        mapping[index] = "dev"
    for index in indices[dev_end:]:
        mapping[index] = "test"
    return mapping


def prepare_esconv(raw_root: Path, output_root: Path, seed: int) -> dict:
    path = raw_root / "esconv" / "ESConv.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    parent_split = split_esconv_parents(data, seed)
    outputs = {split: [] for split in ("train", "dev", "test")}
    excluded = []
    parent_counts = Counter(parent_split.values())

    for parent_index, conversation in enumerate(data):
        messages = []
        strategies_by_message: list[list[str]] = []
        for turn in conversation.get("dialog", []):
            role = "user" if turn.get("speaker") == "seeker" else "assistant"
            content = clean_text(turn.get("content", ""))
            if not content:
                continue
            strategy = turn.get("annotation", {}).get("strategy")
            if messages and messages[-1]["role"] == role:
                messages[-1]["content"] += "\n" + content
                if strategy:
                    strategies_by_message[-1].append(strategy)
            else:
                messages.append({"role": role, "content": content})
                strategies_by_message.append([strategy] if strategy else [])

        parent_id = f"dialog-{parent_index:04d}"
        split = parent_split[parent_index]
        for example in conversation_examples(
            messages,
            source="esconv",
            parent_id=parent_id,
            metadata={
                "emotion_type": conversation.get("emotion_type"),
                "problem_type": conversation.get("problem_type"),
                "situation": conversation.get("situation"),
                "split": split,
            },
        ):
            message_index = int(example["id"].rsplit(":", 1)[1])
            strategies = strategies_by_message[message_index]
            example["metadata"]["strategies"] = strategies
            self_disclosure = "Self-disclosure" in strategies
            if split == "train" and (self_disclosure or example["target_autobiographical_claim"]):
                example["metadata"]["exclusion_reason"] = (
                    "self_disclosure_strategy" if self_disclosure else "autobiographical_pattern"
                )
                excluded.append(example)
            else:
                outputs[split].append(example)

    split_records = {}
    for split, rows in outputs.items():
        target_path = output_root / "policy" / f"esconv_{split}.jsonl"
        write_jsonl(target_path, rows)
        split_records[split] = {
            "parents": parent_counts[split],
            "examples": len(rows),
            "path": str(target_path),
        }
    excluded_path = output_root / "audit" / "esconv_excluded_self_disclosure.jsonl"
    write_jsonl(excluded_path, excluded)
    return {
        "splits": split_records,
        "excluded_self_disclosure_or_autobiographical": len(excluded),
        "excluded_path": str(excluded_path),
    }


def helpsteer_assignment(prompt: str) -> str:
    value = stable_fraction(prompt, "mcare-v2-helpsteer-prompt-split")
    if value < 0.10:
        return "dev"
    if value < 0.55:
        return "policy"
    return "critic"


def helpsteer_quality(row: dict) -> tuple:
    main = (row["helpfulness"], row["correctness"], row["coherence"])
    return (sum(main), min(main), *main, -row["verbosity"], -row["complexity"])


def prepare_helpsteer(raw_root: Path, output_root: Path) -> dict:
    path = raw_root / "helpsteer2" / "train.jsonl.gz"
    grouped: dict[str, list[dict]] = defaultdict(list)
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            # The policy sees clean_text(prompt). Split on that same value so
            # formatting variants cannot cross the policy/dev/critic boundary.
            grouped[clean_text(row["prompt"])].append(row)

    prompt_counts = Counter()
    response_counts = Counter()
    policy_examples = []
    policy_dev = []
    critic_rows = []
    excluded_autobiographical = []
    for prompt in sorted(grouped):
        assignment = helpsteer_assignment(prompt)
        prompt_counts[assignment] += 1
        rows = grouped[prompt]
        response_counts[assignment] += len(rows)
        if assignment == "critic":
            for row in rows:
                critic_rows.append({"prompt_group": content_hash(prompt), **row})
            continue

        eligible = [
            row
            for row in rows
            if min(row["helpfulness"], row["correctness"], row["coherence"]) >= 3
        ]
        if not eligible:
            continue
        best = max(eligible, key=helpsteer_quality)
        example = {
            "id": f"helpsteer2:{content_hash(prompt)[:20]}",
            "source": "helpsteer2",
            "parent_id": content_hash(prompt),
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": clean_text(prompt)},
                {"role": "assistant", "content": clean_text(best["response"])},
            ],
            "target": clean_text(best["response"]),
            "target_autobiographical_claim": has_autobiographical_claim(best["response"]),
            "metadata": {
                "assignment": assignment,
                "scores": {name: best[name] for name in (
                    "helpfulness", "correctness", "coherence", "complexity", "verbosity"
                )},
            },
        }
        if example["target_autobiographical_claim"]:
            excluded_autobiographical.append(example)
        elif assignment == "policy":
            policy_examples.append(example)
        else:
            policy_dev.append(example)

    paths = {
        "policy": output_root / "policy" / "helpsteer2_policy_train.jsonl",
        "dev": output_root / "policy" / "helpsteer2_policy_dev.jsonl",
        "critic": output_root / "critics" / "helpsteer2_critic_train.jsonl",
        "excluded": output_root / "audit" / "helpsteer2_excluded_autobiographical.jsonl",
    }
    write_jsonl(paths["policy"], policy_examples)
    write_jsonl(paths["dev"], policy_dev)
    write_jsonl(paths["critic"], critic_rows)
    write_jsonl(paths["excluded"], excluded_autobiographical)
    return {
        "prompt_groups": dict(prompt_counts),
        "responses": dict(response_counts),
        "high_quality_policy_examples": len(policy_examples),
        "high_quality_dev_examples": len(policy_dev),
        "critic_rows": len(critic_rows),
        "excluded_autobiographical": len(excluded_autobiographical),
        "paths": {name: str(path) for name, path in paths.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, default=get_paths().data_dir / "raw")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=get_paths().data_dir / "processed",
    )
    parser.add_argument("--seed", type=int, default=20260923)
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)

    manifest = {
        "version": "mcare-1.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "system_prompt": SYSTEM_PROMPT,
        "raw_root": str(args.raw_root),
        "output_root": str(args.output_root),
        "ed": prepare_ed(args.raw_root, args.output_root),
        "esconv": prepare_esconv(args.raw_root, args.output_root, args.seed),
        "helpsteer2": prepare_helpsteer(args.raw_root, args.output_root),
        "conditional_support": {
            "status": "pending_base_generation_and_filtering",
            "examples": 0,
        },
    }
    manifest_path = args.output_root / "data_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
