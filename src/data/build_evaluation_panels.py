"""Build parent-isolated development and sealed test evaluation panels (M1, M2, and M3).

Combines the controlled negative-boundary and M2 counterfactual probes (`listen`,
`draft_corrected`) with the six hard-constraint families (`listen`, `draft`,
`update`, `mention`, `omit`, `twosent`), asserting zero parent leakage across
splits.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from common.config import get_paths
from common.io_utils import read_jsonl, sha256_file, write_jsonl

POOLS = {
    "train": {
        "people": [
            ("my brother Daniel", "he", "him"),
            ("my friend Priya", "she", "her"),
            ("my manager Ms. Alvarez", "she", "her"),
            ("my roommate Theo", "he", "him"),
            ("my aunt Rosa", "she", "her"),
            ("my old teammate Jordan", "they", "them"),
            ("my sponsor Mark", "he", "him"),
            ("my cousin Lena", "she", "her"),
        ],
        "days": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"],
        "parts": ["morning", "afternoon", "evening"],
        "needs": [
            "plans made well in advance",
            "a quiet place to talk",
            "a little time to warm up to a conversation",
            "knowing the agenda before we meet",
        ],
        "topics": [
            "moving our catch-up to another day",
            "meeting somewhere quieter",
            "keeping our next conversation short and planned",
        ],
        "limits": [20, 26, 32],
        "listen_limits": [24, 30, 36],
        "disclose": [
            "I should mention that I'm autistic.",
            "For context, I'm autistic.",
            "I'm autistic, in case that matters here.",
        ],
        "listen": [
            "Right now I just need to be heard. Reflect back what you understand, with no advice and no questions, in under {n} words.",
            "I'm not looking for solutions today. Acknowledge how this sounds from my side without giving advice or asking me anything, and keep it under {n} words.",
            "Please just stay with me on this one: tell me what you hear, with no tips and no questions, in at most {n} words.",
        ],
        "draft": [
            "About meeting {person}: I assumed {sub} would be free {stale}. Update: I just heard back and {sub} can't do {stale}. Please write one short message (at most {n} words) asking {obj} to meet {new} instead. Only the message, nothing else.",
            "I need to reschedule with {person}. Earlier I thought {stale} was fine, but that has changed - {sub} is not available then. Write one message of at most {n} words proposing {new}, and nothing else.",
        ],
        "update": [
            "Earlier I told you phone calls are too much for me and that I prefer text. That has changed: a short call on {day} is actually fine now. Please write one short message (at most {n} words) to {person} proposing a call on {day}. Only the message.",
            "Quick correction to what I said before about calls: I used to avoid them, but a brief call on {day} works for me now. Draft one message of at most {n} words to {person} suggesting that call, and nothing else.",
        ],
        "mention": [
            "{disclose} Please write a short message to {person} (at most {n} words) that mentions I'm autistic, because I want {obj} to understand why I need {need}. Only the message.",
            "{disclose} Draft one message of at most {n} words to {person} that says I'm autistic and explains that I need {need}. Message only.",
        ],
        "omit": [
            "{disclose} Please write a short message to {person} about {topic}, at most {n} words, and do not mention that I'm autistic. Only the message.",
            "{disclose} Draft one message of at most {n} words to {person} about {topic}. Leave out any mention of me being autistic. Message only.",
        ],
        "twosent": ["Please answer in two sentences at most.", "Keep your reply to two sentences or fewer."],
    },
    "dev": {
        "people": [
            ("my sister-in-law Mira", "she", "her"),
            ("my coworker Samir", "he", "him"),
            ("my neighbor Mrs. Okafor", "she", "her"),
            ("my study partner Chen", "they", "them"),
            ("my uncle Pavel", "he", "him"),
            ("my friend Noor", "she", "her"),
        ],
        "days": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
        "parts": ["morning", "afternoon", "evening"],
        "needs": [
            "clear plans ahead of time",
            "somewhere with less noise",
            "some quiet time before we talk",
            "a heads-up about who else will be there",
        ],
        "topics": [
            "changing the day of our get-together",
            "picking a calmer place for our next chat",
            "planning a shorter visit",
        ],
        "limits": [22, 28, 34],
        "listen_limits": [27, 33],
        "disclose": ["Just so you know, I'm autistic.", "I'm autistic - mentioning it for context."],
        "listen": [
            "For this reply I only need to feel heard. Please say back what you notice in under {n} words, without advice or questions.",
            "No fixing, please, and no questions. In fewer than {n} words, let me know you understand how this feels.",
        ],
        "draft": [
            "Regarding {person}: I thought {sub} could meet {stale}, but I've now learned that {sub} cannot make {stale}. Please compose a single message, no longer than {n} words, that asks {obj} to meet {new} instead. Provide just the message.",
            "Change of plan with {person}: {stale} is off because {sub} isn't free then. Write one message under {n} words proposing {new}. The message only, please.",
        ],
        "update": [
            "Before, I said I can't handle phone calls and would rather text. I've changed my mind - a short call on {day} would actually be good. Write one message of at most {n} words to {person} proposing that call. Just the message.",
            "Update on what I said about calls: I'm open to one now, briefly, on {day}. Please draft a message to {person} (at most {n} words) suggesting it. Message only.",
        ],
        "mention": [
            "{disclose} Write a brief message to {person} of at most {n} words that tells {obj} I'm autistic and that I need {need}. Message only.",
            "{disclose} Please compose a message (at most {n} words) to {person} that says I'm autistic, so {sub} understands I need {need}. Just the message.",
        ],
        "omit": [
            "{disclose} Write a brief message to {person} about {topic}, at most {n} words, without saying anything about me being autistic. Message only.",
            "{disclose} Compose a message of at most {n} words to {person} about {topic}; please don't bring up that I'm autistic. Just the message.",
        ],
        "twosent": ["Reply in no more than two sentences, please.", "Two sentences maximum for your answer."],
    },
}

DEV_PLAN = [(f, s, n) for f in ("listen", "draft", "update", "mention", "omit", "twosent") for s, n in (("esconv", 12), ("ed", 38))]
TEST_PLAN = [(f, s, 11 if s == "esconv" else 39) for f, s, _ in DEV_PLAN]


def usable(row: dict) -> bool:
    m = row["messages"]
    return len(m) >= 3 and m[-1]["role"] == "assistant" and m[-2]["role"] == "user" and sum(len(x["content"].split()) for x in m[:-1]) <= 600


def make_controlled(row: dict, source: str, condition: str) -> dict:
    messages = [dict(message) for message in row["messages"][:-1]]
    if condition == "listen":
        suffix = (
            "For this reply, I only want you to acknowledge what I shared and leave space "
            "for me to continue. Please do not give advice or ask me questions."
        )
    elif condition == "draft_corrected":
        suffix = (
            "About talking to my sister Maya: I thought she could talk tomorrow evening. "
            "Correction: I just checked, and she cannot talk then. Please write one short "
            "message I could send Maya asking for another time to talk about this. Give only "
            "the actual message, without advice or an introduction."
        )
    else:
        raise ValueError(condition)
    messages[-1]["content"] = messages[-1]["content"].rstrip() + "\n\n" + suffix
    family = f"{source}:{row['parent_id']}"
    return {
        "id": f"test:{family}:{condition}",
        "panel": "M2_controlled",
        "family": family,
        "source": source,
        "condition": condition,
        "messages": messages,
        "construction_known": {
            "request": "listen_only" if condition == "listen" else "sendable_sms_only",
            "prohibited": ["advice", "questions"] if condition == "listen" else ["advice", "introduction"],
            "corrected_fact": None if condition == "listen" else "Maya cannot talk tomorrow evening",
        },
    }


def make_hard(family: str, pool: dict, rng: random.Random, messages: list) -> tuple[list, dict]:
    person, sub, obj = rng.choice(pool["people"])
    rules, extra = [], {}
    if family != "natural":
        if family == "listen":
            n = rng.choice(pool["listen_limits"])
            suffix = rng.choice(pool["listen"]).format(n=n)
            rules = [["no_question"], ["no_advice"], ["max_words", n], ["no_projection"]]
        elif family == "draft":
            n = rng.choice(pool["limits"])
            old_day, new_day = rng.sample(pool["days"], 2)
            old_part, new_part = rng.choice(pool["parts"]), rng.choice(pool["parts"])
            stale, new = f"on {old_day} {old_part}", f"{new_day} {new_part}"
            suffix = rng.choice(pool["draft"]).format(person=person, sub=sub, obj=obj, stale=stale, new=new, n=n)
            rules = [["direct"], ["max_words", n], ["has_time", new_day, new_part], ["stale_absent", old_day, old_part], ["no_projection"]]
        elif family == "update":
            n, day = rng.choice(pool["limits"]), rng.choice(pool["days"])
            suffix = rng.choice(pool["update"]).format(person=person, day=day, n=n)
            rules = [["direct"], ["has_words", "call", day], ["max_words", n], ["no_projection"]]
        elif family in ("mention", "omit"):
            n = rng.choice(pool["limits"])
            suffix = rng.choice(pool[family]).format(
                disclose=rng.choice(pool["disclose"]),
                person=person,
                obj=obj,
                sub=sub,
                need=rng.choice(pool["needs"]),
                topic=rng.choice(pool["topics"]),
                n=n,
            )
            rules = [["direct"], ["has_identity"] if family == "mention" else ["no_identity"], ["max_words", n]]
        else:
            suffix = rng.choice(pool["twosent"])
            rules = [["max_sentences", 2], ["no_projection"]]
        messages = [dict(m) for m in messages]
        messages[-1]["content"] = messages[-1]["content"].rstrip() + "\n\n" + suffix
    else:
        rules = [["no_projection"]]
    return messages, {"family": family, "rules": rules, "ref_words": None, **extra}


def parent_pools(policy_dir: Path, split_suffix: str, rng: random.Random) -> dict[str, list[tuple[str, list[dict]]]]:
    result = {}
    for source in ("esconv", "ed"):
        filename = f"{source}_{split_suffix}.jsonl"
        grouped = defaultdict(list)
        for row in read_jsonl(policy_dir / filename):
            m = row["messages"]
            if len(m) > 9 or m[-2]["role"] != "user":
                continue
            if 35 <= len(m[-2]["content"]) <= 450 and len(row["target"]) >= 12:
                grouped[row["parent_id"]].append(row)
        parents = sorted(grouped)
        rng.shuffle(parents)
        result[source] = [(p, grouped[p]) for p in parents]
    return result


def main() -> None:
    cfg = get_paths()
    parser = argparse.ArgumentParser(description="Build sealed test panels (M1 natural, M2 controlled, M3 hard constraints).")
    parser.add_argument("--policy-dir", type=Path, default=cfg.policy_dir)
    parser.add_argument("--out-dir", type=Path, default=cfg.exp_dir / "test_panel")
    parser.add_argument("--seed", type=int, default=20261010)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    pools_by_source = parent_pools(args.policy_dir, "test", rng)
    quota = {"esconv": 60 + 60 + 66, "ed": 140 + 140 + 234}
    for src in pools_by_source:
        assert len(pools_by_source[src]) >= quota[src], (src, len(pools_by_source[src]), quota[src])

    cur = {"esconv": 0, "ed": 0}

    def take(source: str, count: int) -> list[tuple[str, list[dict]]]:
        out = pools_by_source[source][cur[source] : cur[source] + count]
        cur[source] += count
        return out

    rows = []
    for source, n_nat, n_ctl in (("esconv", 60, 60), ("ed", 140, 140)):
        for parent, cands in take(source, n_nat):
            row = rng.choice(cands)
            rows.append({
                "id": f"test:natural:{source}:{parent}",
                "panel": "M1_natural",
                "family": f"{source}:{parent}",
                "source": source,
                "messages": row["messages"][:-1],
                "reference_source": f"{source}_test",
                "reference_parent_id": parent,
            })
        for parent, cands in take(source, n_ctl):
            row = rng.choice(cands)
            for condition in ("listen", "draft_corrected"):
                rows.append(make_controlled(row, source, condition))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out_dir / "panel.jsonl", rows)

    m3 = []
    rng3 = random.Random(args.seed + 1)
    for family, source, count in TEST_PLAN:
        for parent, cands in take(source, count):
            row = rng3.choice([c for c in cands if usable(c)] or cands)
            messages, facts = make_hard(family, POOLS["dev"], rng3, [dict(m) for m in row["messages"][:-1]])
            m3.append({
                "id": f"test:{family}:{source}:{parent}",
                "messages": messages,
                "ctx": json.dumps(messages, ensure_ascii=False),
                "cond": family,
                "facts": json.dumps(facts),
                "source": source,
                "parent_id": parent,
            })
    rng3.shuffle(m3)
    write_jsonl(args.out_dir / "m3_panel.jsonl", m3)

    manifest = {
        "seed": args.seed,
        "m1_m2_rows": len(rows),
        "m3_rows": len(m3),
        "esconv_parents_used": cur["esconv"],
        "ed_parents_used": cur["ed"],
        "parent_overlap_m1m2_m3": len({r["family"].split(":", 1)[1] for r in rows} & {r["parent_id"] for r in m3}),
        "panel_sha256": sha256_file(args.out_dir / "panel.jsonl"),
        "m3_panel_sha256": sha256_file(args.out_dir / "m3_panel.jsonl"),
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
