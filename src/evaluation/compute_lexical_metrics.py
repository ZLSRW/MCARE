"""Compute model-free lexical diversity, information richness, and reference coverage metrics.

Reproduces Table II (EAD-2, Ent-4, and ROUGE-L Recall on M1 natural support
across ESConv and EmpatheticDialogues), plus optional corpus diagnostics.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+|[^\w\s]", (text or "").lower())


def lcs_length(a: list[str], b: list[str]) -> int:
    if not a or not b:
        return 0
    dp = [0] * (len(b) + 1)
    for x in a:
        prev = 0
        for j, y in enumerate(b):
            cur = dp[j + 1]
            if x == y:
                dp[j + 1] = prev + 1
            elif dp[j] > dp[j + 1]:
                dp[j + 1] = dp[j]
            prev = cur
    return dp[-1]


def entropy_n(tokens_list: list[list[str]], n: int) -> float:
    counts: Counter[tuple[str, ...]] = Counter()
    total = 0
    for tokens in tokens_list:
        for i in range(len(tokens) - n + 1):
            counts[tuple(tokens[i : i + n])] += 1
            total += 1
    if total == 0:
        return 0.0
    return -sum((cnt / total) * math.log(cnt / total) for cnt in counts.values())


def ead_n(tokens_list: list[list[str]], n: int, vocab_size: int = 50000) -> float:
    ngrams: set[tuple[str, ...]] = set()
    total = 0
    for tokens in tokens_list:
        for i in range(len(tokens) - n + 1):
            ngrams.add(tuple(tokens[i : i + n]))
            total += 1
    if total == 0:
        return 0.0
    denom = vocab_size * (1.0 - ((vocab_size - 1.0) / vocab_size) ** total)
    return len(ngrams) / denom


def compute_subset_metrics(
    target_ids: list[str],
    refs: dict[str, str],
    responses: dict[str, str],
    vocab_size: int = 50000,
) -> dict[str, float | int]:
    hyp_tokens = [tokenize(responses[i]) for i in target_ids if i in responses]
    rl_recalls: list[float] = []
    uni_recalls: list[float] = []
    sent_d2: list[float] = []

    for gid in target_ids:
        if gid not in responses or gid not in refs:
            continue
        rt = tokenize(refs[gid])
        ht = tokenize(responses[gid])
        if rt:
            rl_recalls.append(lcs_length(rt, ht) / len(rt))
            rc = Counter(rt)
            hc = Counter(ht)
            uni_recalls.append(sum(min(rc[w], hc[w]) for w in rc) / len(rt))
        bigrams = [tuple(ht[j : j + 2]) for j in range(len(ht) - 1)]
        if bigrams:
            sent_d2.append(len(set(bigrams)) / len(bigrams))

    return {
        "n": len(hyp_tokens),
        "ead_2": ead_n(hyp_tokens, 2, vocab_size=vocab_size),
        "ent_2": entropy_n(hyp_tokens, 2),
        "ent_4": entropy_n(hyp_tokens, 4),
        "sent_distinct_2": (sum(sent_d2) / len(sent_d2)) if sent_d2 else 0.0,
        "rouge_l_recall": (sum(rl_recalls) / len(rl_recalls)) if rl_recalls else 0.0,
        "unigram_recall": (sum(uni_recalls) / len(uni_recalls)) if uni_recalls else 0.0,
    }


def load_responses(paths: list[Path], valid_ids: set[str]) -> dict[str, str]:
    resps: dict[str, str] = {}
    for p in paths:
        if not p.exists():
            continue
        with p.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                gid = row.get("id")
                if gid in valid_ids:
                    resps[gid] = row.get("response", "")
    return resps


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute EAD-2, Ent-4, and ROUGE-L Recall on M1 natural support.")
    parser.add_argument("--refs", type=Path, required=True, help="Path to references.json mapping id -> reference reply")
    parser.add_argument(
        "--models-json",
        type=Path,
        required=True,
        help="JSON mapping model display name -> generation JSONL path or list of JSONL paths",
    )
    parser.add_argument("--out-json", type=Path, default=None, help="Optional JSON output path")
    parser.add_argument("--out-csv", type=Path, default=None, help="Optional CSV output path")
    parser.add_argument("--vocab-size", type=int, default=50000, help="Effective bigram vocabulary size V for EAD-2")
    args = parser.parse_args()

    refs: dict[str, str] = json.loads(args.refs.read_text(encoding="utf-8"))
    raw_map: dict[str, Any] = json.loads(args.models_json.read_text(encoding="utf-8"))
    model_files = {
        name: [Path(p) for p in (paths if isinstance(paths, list) else [paths])]
        for name, paths in raw_map.items()
    }

    ref_ids = set(refs)
    subsets = [
        ("esconv", "ESConv", [k for k in refs if k.startswith("esconv:")]),
        ("ed", "ED", [k for k in refs if k.startswith("ed:")]),
    ]

    results: dict[str, dict[str, Any]] = {name: {} for name in model_files}
    for name, paths in model_files.items():
        responses = load_responses(paths, ref_ids)
        for key_prefix, _, target_ids in subsets:
            results[name][key_prefix] = compute_subset_metrics(target_ids, refs, responses, vocab_size=args.vocab_size)

    print(f"{'Model':15s} | {'ESC EAD-2':9s} | {'ESC Ent-4':9s} | {'ESC R-L Rec':11s} | {'ED EAD-2':8s} | {'ED Ent-4':8s} | {'ED R-L Rec':10s}")
    for name in model_files:
        es, ed = results[name]["esconv"], results[name]["ed"]
        print(
            f"{name:15s} | {es['ead_2']:9.3f} | {es['ent_4']:9.2f} | {es['rouge_l_recall']*100:10.2f}% | "
            f"{ed['ead_2']:8.3f} | {ed['ent_4']:8.2f} | {ed['rouge_l_recall']*100:9.2f}%"
        )

    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    if args.out_csv:
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.out_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["model", "esconv_n", "esconv_ead2", "esconv_ent4", "esconv_rouge_l_rec_pct", "ed_n", "ed_ead2", "ed_ent4", "ed_rouge_l_rec_pct"])
            for name in model_files:
                es, ed = results[name]["esconv"], results[name]["ed"]
                writer.writerow([
                    name,
                    es["n"], f"{es['ead_2']:.4f}", f"{es['ent_4']:.4f}", f"{es['rouge_l_recall']*100:.2f}",
                    ed["n"], f"{ed['ead_2']:.4f}", f"{ed['ent_4']:.4f}", f"{ed['rouge_l_recall']*100:.2f}",
                ])


if __name__ == "__main__":
    main()
