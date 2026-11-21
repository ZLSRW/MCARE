"""RLVER "heart-in-the-loop" environment for ms-swift multi-turn GRPO (gym_scheduler): a local LLM plays the user (Sentient-Agent style).

Per exchange the assistant reply (text after the last </think> when the think template is used) is shown to the simulator, which
  1. runs the emotion analyser (planning_reply): emotion change -> emo_point (start 30, cap 100), state S/A/B/C/F by the released thresholds;
  2. runs the actor (player_reply) to write the next user message.
The episode ends after max_turns exchanges, when the actor says goodbye, or when emo_point <= 0. The only reward is the terminal
emotion score / 100 (paper Sec. 2.1); in think mode a reply that breaks the <think> format sets the episode reward to 0.
Environment variables: RLVER_PROFILES, RLVER_SIM_URL, RLVER_SIM_MODEL, RLVER_THINK, RLVER_MAX_TURNS, RLVER_LOG, RLVER_FAKE (smoke tests only).
"""

from __future__ import annotations

import json
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, Tuple

from openai import AsyncOpenAI
from swift.rollout.gym_env import Env, envs

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[2]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
from baselines.external.rlver_prompts_en import (EMO_COUNT, EMO_START, OPENING_HISTORY, OPENING_PLANNING, PLANNER_TEMPLATE, PLAYER_TEMPLATE, SYSTEM_THINK,  # noqa: E402
                              SYSTEM_UNTHINK, TARGET_EQ)

from common.config import get_paths as _get_paths
ROOT = _get_paths().root
PROFILES = Path(os.environ.get("RLVER_PROFILES", _get_paths().baselines_dir / "external/rlver_repro/data/profiles.jsonl"))
SIM_URL = os.environ.get("RLVER_SIM_URL", "http://127.0.0.1:8030/v1")
SIM_MODEL = os.environ.get("RLVER_SIM_MODEL", "sim")
THINK = os.environ.get("RLVER_THINK", "0") == "1"
MAX_TURNS = int(os.environ.get("RLVER_MAX_TURNS", "8"))
LOG = os.environ.get("RLVER_LOG")
FAKE = os.environ.get("RLVER_FAKE", "0") == "1"
BREVITY = os.environ.get("RLVER_BREVITY", "0") == "1"  # ask the simulator for short analysis sections (latency: the runs are bound by sequential simulator decoding)
BREVITY_PLANNER = "\n\nKeep every section short: at most two sentences each (the Change section is only the number).\n"
BREVITY_PLAYER = "\n\nKeep the Thinking, Origin and Change parts short: at most two sentences each; the Response stays as specified above.\n"
GOODBYE = re.compile(r"\b(good-?bye|bye(-bye)?)\b", re.I)
_PROFILE_CACHE: dict[str, dict] = {}
STATS = {"calls": 0, "completion_tokens": 0, "prompt_tokens": 0}


def profile(profile_id: str) -> dict:
    if not _PROFILE_CACHE:
        for line in PROFILES.open(encoding="utf-8"):
            row = json.loads(line)
            _PROFILE_CACHE[row["id"]] = row
    return _PROFILE_CACHE[profile_id]


def extract_content(text: str) -> str:
    return text.split("</think>")[-1].strip() if "</think>" in text else text.strip()


def format_ok(text: str) -> bool:
    return text.find("<think>") != -1 and text.find("<think>") < text.find("</think>") and bool(extract_content(text))


def dump(turns: list[dict]) -> str:
    return json.dumps(turns, ensure_ascii=False, indent=2)


async def chat(prompt: str, temperature: float = 0.5, max_tokens: int = 900) -> str:
    client = AsyncOpenAI(base_url=SIM_URL, api_key="EMPTY", timeout=900, max_retries=3)  # one client per call: swift drives several short-lived event loops
    try:
        r = await client.chat.completions.create(model=SIM_MODEL, messages=[{"role": "user", "content": prompt}], temperature=temperature, max_tokens=max_tokens,
                                                 extra_body={"chat_template_kwargs": {"enable_thinking": False}})
        STATS["calls"] += 1
        STATS["completion_tokens"] += r.usage.completion_tokens
        STATS["prompt_tokens"] += r.usage.prompt_tokens
        return r.choices[0].message.content or ""
    finally:
        await client.close()


def section(text: str, start: str, end: str | None) -> str:
    text = text.replace("：", ":").replace("*", "")
    pattern = rf"{start}:\s*(.*?)\s*(?={end}:|$)" if end else rf"{start}:\s*(.*)$"
    matches = re.findall(pattern, text, re.S)
    return matches[-1].strip().strip("[]").strip() if matches else ""


class RLVEREnv(Env):
    def __init__(self, env_config):
        super().__init__(env_config)
        self.emo_point = EMO_START
        self.emo_state = "Emotion-C"
        self.history: list[dict] = []
        self.profile: dict = {}
        self.turns = 0
        self.bad_format = 0
        self.trace: list[dict] = []

    def state_for(self, point: int) -> str:
        for name, threshold in EMO_COUNT.items():
            if point >= threshold:
                return name
        return "Emotion-F"

    def view(self) -> tuple[str, str]:
        mapping = {"user": "you", "assistant": "NPC"}
        rows = [{"role": mapping[m["role"]], "content": m["content"]} for m in self.history]
        return dump(rows[:-2]), dump(rows[-2:])

    async def plan(self) -> dict:
        context, latest = self.view()
        prompt = (PLANNER_TEMPLATE.replace("{{emotion}}", str(self.emo_point)).replace("{{player_type}}", self.profile["player"])
                  .replace("{{player_topic}}", self.profile["scene"]).replace("{{target}}", TARGET_EQ)
                  .replace("{{dialog_history}}", context).replace("{{new_history}}", latest))
        if BREVITY:
            prompt += BREVITY_PLANNER
        if FAKE:
            return {"reason": "fake", "analyse": "fake", "change": random.randint(-8, 14)}
        for _ in range(3):
            text = await chat(prompt)
            tail = text.replace("：", ":").split("Change:")[-1] if "Change:" in text else ""
            number = re.search(r"[-+]?\d+", tail)
            if number:
                return {"reason": section(text, "Reason", "Activity"), "analyse": section(text, "Analyse", "Change"), "change": int(number.group(0))}
        return {"reason": "", "analyse": "", "change": 0, "parse_failed": True}

    async def player(self, planning: dict | None) -> str:
        if planning is None:
            context, latest, feeling = OPENING_HISTORY, "", OPENING_PLANNING
        else:
            context, latest = self.view()
            feeling = "Objective analysis of the NPC's reply:\n" + planning["reason"] + "\nSubjective analysis of the NPC's reply:\n" + planning["analyse"]
        prompt = (PLAYER_TEMPLATE.replace("{{planning}}", feeling).replace("{{target}}", TARGET_EQ).replace("{{emotion}}", self.emo_state)
                  .replace("{{player_type}}", self.profile["player"]).replace("{{player_topic}}", self.profile["scene"])
                  .replace("{{dialog_history}}", context).replace("{{new_history}}", latest))
        if BREVITY:
            prompt += BREVITY_PLAYER
        if FAKE:
            return random.choice(["Hmm, I guess so.", "It just keeps bothering me.", "Thanks, bye.", "Not sure what to say."])
        for _ in range(3):
            text = await chat(prompt, max_tokens=1100)
            if "Response:" in text:
                reply = text.split("Response:")[-1].strip().strip("[]").strip().strip("\"“”").strip()
                if reply:
                    return reply
        return "..."

    async def reset(self, config) -> Tuple[str, Dict[str, Any], str]:
        self.emo_point = EMO_START
        self.emo_state = "Emotion-C"
        self.turns = 0
        self.bad_format = 0
        self.trace = []
        self.profile = profile(config.data_dict["profile_id"])
        if self.profile.get("initial_history"):
            # Full-data adaptation: continue an actual train history, with no synthetic opening.
            self.history = [dict(m) for m in self.profile["initial_history"]]
            opening = "Continue this visible conversation as the supporter.\n" + "\n".join(
                f"{'User' if m['role']=='user' else 'Assistant'}: {m['content']}" for m in self.history)
            return opening, {"profile_id": self.profile["id"]}, self.profile["initial_system"]
        opening = await self.player(None)
        self.history = [{"role": "user", "content": opening}]
        return opening, {"profile_id": self.profile["id"]}, SYSTEM_THINK if THINK else SYSTEM_UNTHINK

    async def step(self, action) -> Tuple[str, float, bool, Dict[str, Any]]:
        raw = action[-1]["content"] or ""
        if THINK and not format_ok(raw):
            self.bad_format += 1
        reply = extract_content(raw)  # drops a think block; Qwen3.5 non-thinking outputs also carry the empty <think></think> prefix
        self.history.append({"role": "assistant", "content": reply or "..."})
        planning = await self.plan()
        self.emo_point = min(100, self.emo_point + planning["change"])
        self.emo_state = self.state_for(self.emo_point)
        text = await self.player(planning)
        self.history.append({"role": "user", "content": text})
        self.turns += 1
        self.trace.append({"emo": self.emo_point, "change": planning["change"]})
        done = self.turns >= MAX_TURNS or bool(GOODBYE.search(text)) or self.emo_point <= 0
        reward = 0.0
        if done:
            reward = max(0, self.emo_point) / 100.0
            if THINK and self.bad_format:
                reward = 0.0
            if LOG:
                with open(LOG, "a", encoding="utf-8") as handle:
                    handle.write(json.dumps({"time": time.time(), "profile_id": self.profile["id"], "emo_final": self.emo_point, "turns": self.turns,
                                             "bad_format": self.bad_format, "reward": reward, "trace": self.trace, "history": self.history}, ensure_ascii=False) + "\n")
        return text, reward, done, {"emo_point": self.emo_point}

    async def close(self):
        pass


envs["rlver_env"] = RLVEREnv
