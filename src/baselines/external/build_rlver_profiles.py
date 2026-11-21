"""Supportive-dialogue scenarios for the RLVER re-implementation (paper Appendix B.2: 500 training + 100 test scenarios over 8 user goals).

The released code reads persona / scene / topic records from data/train_profile.jsonl, which is not published. Each record is rebuilt here:
  * seed: the emotion type, problem type and situation summary of an ESConv train-split dialogue (parents that appear in the evaluation panel are
    excluded), so no new personal data is invented and every scenario is tied to an existing, leak-free seed;
  * goal: one of the 8 hidden intentions of the paper's Table 5, with the paper's per-goal counts (66/66/66/64/63/60/58/57 train, 11/13/12/13/12/11/15/13 test);
  * a local LLM writes persona, background, topics, hidden theme and the per-emotion-level reactions that the released simulator prompts refer to.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import re
from pathlib import Path

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[2]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
from baselines.external.esc_repro_common import E, POLICY, panel_excluded_parents, read_jsonl, write_jsonl

GOALS = [
    ("You believe you bear no responsibility or fault in the situation, and you want the other person to agree that you are not at fault.", 66, 11),
    ("You hope the other person will guide you to engage in self-reflection regarding the incident and help you achieve personal growth.", 66, 13),
    ("You hope the other person will critically analyze the underlying problems in the incident.", 66, 12),
    ("You hope the other person will deeply empathize with your feelings, rather than simply offering comfort.", 64, 13),
    ("You want the other person to attentively listen to your emotional outpouring.", 63, 12),
    ("You want to analyze the reasons behind the actions of other individuals involved in the incident.", 60, 11),
    ("You hope to receive advice that can genuinely help you overcome your current difficulties.", 58, 15),
    ("You hope the other person will sincerely praise your specific actions in the situation.", 57, 13),
]

WRITER_SYSTEM = "You write character sheets for role-play actors who practise supportive conversations. Output one JSON object and nothing else."
WRITER_USER = """Create a fictional person and the background of one conversation in which this person confides in a friend (the listener).

Inspiration (rewrite it into a new fictional case; do not copy names or identifying details):
- emotion: {emotion}
- problem type: {problem}
- situation summary: {situation}

The person's hidden intention in the conversation (never stated outright, it shapes what they hope to get from the listener): {goal}
Personal profile to respect: {gender}, about {age} years old; {negativity}; in conversation they are {style}.

Return a JSON object with these keys:
"persona": string, 3-5 sentences: name, age, occupation, personality, and the habits and behavior in conversation (speaking style, how they react to comfort, advice or questions).
"background": string, 5-8 sentences: what happened, with concrete details (who, when, what was said or done), and how the person currently feels.
"topics": list of 2-3 short strings, things the person may want to confide about.
"hidden_theme": one sentence stating the deeper thing the person wants from the listener, consistent with the hidden intention above.
"reactions": object with the keys "S", "A", "B", "C", "F"; each value 1-2 sentences describing how this person talks at that emotion level (S: very satisfied and ready to say goodbye; A: warm and open; B: neutral, short answers; C: guarded or negative; F: wants to end the conversation)."""


def json_from(text: str) -> dict | None:
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    ok = isinstance(data.get("persona"), str) and isinstance(data.get("background"), str) and isinstance(data.get("topics"), list) \
        and isinstance(data.get("hidden_theme"), str) and isinstance(data.get("reactions"), dict) and all(k in data["reactions"] for k in "SABCF")
    return data if ok else None


def compose(data: dict) -> tuple[str, str]:
    reactions = "\n".join(f"Emotion-{k}: {data['reactions'][k]}" for k in "SABCF")
    scene = (f"{data['background'].strip()}\n"
             f"Topics the player may want to confide to the NPC: {'; '.join(str(t) for t in data['topics'])}\n"
             f"Hidden theme (never disclose it directly): {data['hidden_theme'].strip()}\n"
             f"Reactions of the player at each emotion level:\n{reactions}")
    return data["persona"].strip(), scene


async def write_all(seeds: list[dict], url: str, model: str, concurrency: int) -> list[dict]:
    from openai import AsyncOpenAI

    gate = asyncio.Semaphore(concurrency)
    client = AsyncOpenAI(base_url=url, api_key="EMPTY", timeout=600, max_retries=2)

    async def one(seed: dict) -> dict | None:
        prompt = WRITER_USER.format(**{k: seed[k] for k in ("emotion", "problem", "situation", "goal", "gender", "age", "negativity", "style")})
        for _ in range(4):
            async with gate:
                r = await client.chat.completions.create(model=model, temperature=0.9, top_p=0.95, max_tokens=1400,
                                                        messages=[{"role": "system", "content": WRITER_SYSTEM}, {"role": "user", "content": prompt}],
                                                        extra_body={"chat_template_kwargs": {"enable_thinking": False}})
            data = json_from(r.choices[0].message.content or "")
            if data:
                persona, scene = compose(data)
                return {**{k: seed[k] for k in ("id", "split", "goal", "source_parent")}, "player": persona, "scene": scene, "raw": data}
        return None

    results = await asyncio.gather(*(one(s) for s in seeds))
    return [r for r in results if r]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=E / "rlver_repro/data/profiles.jsonl")
    parser.add_argument("--url", default="http://127.0.0.1:8030/v1")
    parser.add_argument("--model", default="sim")
    parser.add_argument("--concurrency", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260932)
    parser.add_argument("--fake", action="store_true", help="template profiles for smoke tests only (no LLM)")
    args = parser.parse_args()
    rng = random.Random(args.seed)
    excluded = panel_excluded_parents()
    parents = {}
    for row in read_jsonl(POLICY / "esconv_train.jsonl"):
        if row["parent_id"] not in excluded and row["parent_id"] not in parents:
            meta = row["metadata"]
            if meta.get("situation", "").strip():
                parents[row["parent_id"]] = meta
    order = sorted(parents)
    rng.shuffle(order)
    seeds, cursor = [], 0
    for split, index in (("train", 1), ("test", 2)):
        for goal, *counts in GOALS:
            for _ in range(counts[index - 1]):
                parent = order[cursor]
                cursor += 1
                meta = parents[parent]
                seeds.append({"id": f"rlver-{split}-{len(seeds):04d}", "split": split, "goal": goal, "source_parent": parent,
                              "emotion": meta.get("emotion_type", ""), "problem": meta.get("problem_type", ""), "situation": meta["situation"].strip(),
                              "gender": rng.choice(["female", "female", "male", "male", "non-binary"]), "age": rng.randint(18, 62),
                              "negativity": rng.choice(["the person has clearly negative traits (pessimistic, easily discouraged) and is not optimistic",
                                                        "the person has both negative and positive traits, not all optimistic"]),
                              "style": rng.choice(["passive and reserved (waits to be asked, does not ask questions on their own)",
                                                   "active and talkative (brings up details and asks the listener questions)"])})
    if args.fake:
        rows = [{**{k: s[k] for k in ("id", "split", "goal", "source_parent")}, "player": f"Alex, {s['age']}, a {s['gender']} office worker who is {s['style']}.",
                 "scene": f"{s['situation']}\nTopics the player may want to confide to the NPC: {s['problem']}\nHidden theme (never disclose it directly): {s['goal']}\n"
                          "Reactions of the player at each emotion level:\nEmotion-S: thanks and says goodbye.\nEmotion-A: warm.\nEmotion-B: neutral.\nEmotion-C: guarded.\nEmotion-F: leaves.",
                 "raw": {}} for s in seeds]
    else:
        rows = asyncio.run(write_all(seeds, args.url, args.model, args.concurrency))
    write_jsonl(args.out, rows)
    print({"requested": len(seeds), "written": len(rows), "train": sum(r["split"] == "train" for r in rows), "test": sum(r["split"] == "test" for r in rows)})


if __name__ == "__main__":
    main()
