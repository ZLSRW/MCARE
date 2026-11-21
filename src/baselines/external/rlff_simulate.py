"""RLFF-ESC multi-agent dialogue simulation (paper Algorithm 1) -> reward-model dataset D_r.

For every ESConv context c_{1:t-1} and each of m = 4 sampled counselor responses u_{t,j}:
  * the system agent M_sys (Qwen2.5-7B-Instruct, tau=1.1, top_k=80, top_p=1.0) proposes u_{t,j};
  * a user simulator U (Table 7 prompt) replies; then for k = 1..T (T = 8): M_sys answers, U replies, the critic M_crt (Table 9 prompt,
    tau=1.1, 10 samples) rates the conversation on the 7-level scale {-1,-.5,-.25,0,.25,.5,1} (mean of the samples = r_{t+k});
    stop when r_{t+k} > 0.5 or k = T;
  * r_sys = (r_{t+k} + 1/AvgT) / 2 with AvgT = k + 1.
Deviations from the paper: U and M_crt are one local model (Qwen3-30B-A3B-GPTQ-Int4, non-thinking) instead of Qwen2.5-72B-Instruct; the critic labels the seven
levels with the letters A-G and the 10 sampled answers (tau = 1.1) are drawn locally from the model's first-token distribution over the letters (one request with
log-probabilities instead of ten sampled requests: the server prefills the sibling samples of a request separately, which made the critic 10x as expensive).
The situation text is given to U as its first utterance (as in Table 7); the real ESConv turns follow.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import random
import re
import time
from pathlib import Path

from openai import AsyncOpenAI

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[2]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
from baselines.external.esc_repro_common import E, read_jsonl

SYS_PROMPT = ("Now enter the role-playing mode. In the following conversation, you will play as a therapist in a counselling conversation "
              "with a patient. Your goal is to help the patient reduce their emotional distress and support them working through their challenges.")
USER_SYSTEM = "Now enter the role-playing mode. In the following conversation, you will play as a patient in a counselling conversation with a therapist."
USER_INSTRUCTION = ("You are the patient who is looking for the help from the therapist, because you have the emotional issue about {emotion} "
                    "regarding {problem}. Please reply with only one short and succinct sentence. Now tell me your issue.")
CRITIC_SYSTEM = "Given a conversation between a therapist and a patient, please assess whether the patient' emotional issue has been solved after the conversation."
# Table 9 text; the conversation is placed last (after the requirement and the seven level definitions) with a one-line reminder behind it, so that the
# critic prompt grows by appending across rounds and the server can reuse its cache (the original layout puts the long requirement after the conversation).
CRITIC_HEAD = ("Question : Has the patient's issue been solved?\n"
               "Requirement : You can only reply with the letter of one of the following seven descriptive levels without explanation:\n"
               "A. Same: The patient's feelings remain unchanged.\n"
               "B. Slightly Better: The patient feels a slight, barely noticeable improvement.\n"
               "C. Moderately Better: The patient feels somewhat better, with a small but noticeable improvement.\n"
               "D. Significantly Better: The patient feels significantly better, indicating a clear improvement.\n"
               "E. Slightly Worse: The patient feels a slight increase in tension, barely noticeable decline.\n"
               "F. Moderately Worse: The patient feels somewhat more stressed or worried, with a small but noticeable decline.\n"
               "G. Significantly Worse: The patient feels significantly more distressed or upset, indicating a clear decline.\n\n"
               "The following is a conversation about {emotion} regarding {problem}:\n")
CRITIC_TAIL = "\n\nQuestion : Has the patient's issue been solved? Reply with one letter (A-G) only."
LEVELS = {"A": 0.0, "B": 0.25, "C": 0.5, "D": 1.0, "E": -0.25, "F": -0.5, "G": -1.0}  # Same, Slightly/Moderately/Significantly Better, Slightly/Moderately/Significantly Worse
CRITIC_TEMPERATURE = 1.1


def merge_roles(messages: list[dict]) -> list[dict]:
    merged: list[dict] = []
    for m in messages:
        if merged and merged[-1]["role"] == m["role"] and m["role"] != "system":
            merged[-1]["content"] += "\n" + m["content"]
        else:
            merged.append(dict(m))
    return merged


def conversation_text(turns: list[dict]) -> str:
    return "\n".join(f"{'Patient' if t['role'] == 'user' else 'Therapist'}: {t['content'].strip()}" for t in turns)


class Simulator:
    def __init__(self, args):
        self.args = args
        self.sys = AsyncOpenAI(base_url=args.sys_url, api_key="EMPTY", timeout=600, max_retries=2)
        self.sim = AsyncOpenAI(base_url=args.sim_url, api_key="EMPTY", timeout=600, max_retries=2)
        self.sys_gate, self.sim_gate = asyncio.Semaphore(args.sys_concurrency), asyncio.Semaphore(args.sim_concurrency)
        self.calls = {"sys": 0, "sim": 0, "critic": 0}
        self.rng = random.Random(args.seed)

    async def sys_reply(self, turns: list[dict]) -> str:
        messages = [{"role": "system", "content": SYS_PROMPT}] + merge_roles(turns)
        async with self.sys_gate:
            r = await self.sys.chat.completions.create(model="sys", messages=messages, temperature=1.1, top_p=1.0, max_tokens=320,
                                                      extra_body={"top_k": 80})
        self.calls["sys"] += 1
        return (r.choices[0].message.content or "").strip() or "I hear you."

    async def user_reply(self, ctx: dict, turns: list[dict]) -> str:
        # simulator's own view: patient = assistant role, therapist = user role
        flipped = [{"role": "assistant" if t["role"] == "user" else "user", "content": t["content"]} for t in turns]
        messages = [{"role": "system", "content": USER_SYSTEM},
                    {"role": "user", "content": USER_INSTRUCTION.format(emotion=ctx["emotion_type"], problem=ctx["problem_type"])},
                    {"role": "assistant", "content": ctx["situation"]}] + flipped
        async with self.sim_gate:
            r = await self.sim.chat.completions.create(model="sim", messages=merge_roles(messages), temperature=0.7, top_p=0.8, max_tokens=120,
                                                      extra_body={"chat_template_kwargs": {"enable_thinking": False}})
        self.calls["sim"] += 1
        return (r.choices[0].message.content or "").strip() or "I don't know what to say."

    async def critic(self, ctx: dict, turns: list[dict]) -> tuple[float, dict]:
        messages = [{"role": "system", "content": CRITIC_SYSTEM},
                    {"role": "user", "content": CRITIC_HEAD.format(emotion=ctx["emotion_type"], problem=ctx["problem_type"]) + conversation_text(turns) + CRITIC_TAIL}]
        async with self.sim_gate:
            r = await self.sim.chat.completions.create(model="sim", messages=messages, temperature=1.0, max_tokens=1, logprobs=True, top_logprobs=20,
                                                      extra_body={"chat_template_kwargs": {"enable_thinking": False}})
        self.calls["critic"] += 1
        mass = {}
        for item in r.choices[0].logprobs.content[0].top_logprobs:
            token = item.token.strip()
            if token in LEVELS:
                mass[token] = mass.get(token, 0.0) + math.exp(item.logprob / CRITIC_TEMPERATURE)  # tempered: q_i proportional to p_i ** (1/tau)
        total = sum(mass.values())
        if total <= 0:
            return 0.0, {"unparsed": True}
        letters = sorted(mass)
        probs = [mass[k] / total for k in letters]
        draws = self.rng.choices(letters, weights=probs, k=self.args.critic_samples)  # the paper's 10 samples at tau = 1.1
        return sum(LEVELS[d] for d in draws) / len(draws), {"probs": {k: round(p, 4) for k, p in zip(letters, probs)}, "draws": "".join(draws)}

    async def run_chunk(self, jobs: list[tuple[dict, int]]) -> list[dict]:
        """Simulate a chunk of (context, j) jobs in lock-step rounds: all system replies, then all user replies, then all critic calls.
        The two model servers share one GPU, and running them at the same time makes both slow; alternating phases keeps only one of them busy."""
        states = [{"ctx": c, "j": j, "turns": list(c["turns"]), "rounds": [], "done": False} for c, j in jobs]

        async def each(fn, items):
            return await asyncio.gather(*(fn(x) for x in items))

        firsts = await each(lambda st: self.sys_reply(st["turns"]), states)
        for st, text in zip(states, firsts):
            st["first"] = text
            st["turns"].append({"role": "assistant", "content": text})
        users = await each(lambda st: self.user_reply(st["ctx"], st["turns"]), states)
        for st, text in zip(states, users):
            st["turns"].append({"role": "user", "content": text})
        for k in range(1, self.args.max_turns + 1):
            active = [st for st in states if not st["done"]]
            if not active:
                break
            t0 = time.time()
            for st, text in zip(active, await each(lambda st: self.sys_reply(st["turns"]), active)):
                st["turns"].append({"role": "assistant", "content": text})
            t1 = time.time()
            for st, text in zip(active, await each(lambda st: self.user_reply(st["ctx"], st["turns"]), active)):
                st["turns"].append({"role": "user", "content": text})
            t2 = time.time()
            critic_out = await each(lambda st: self.critic(st["ctx"], st["turns"]), active)
            print(f"  phase seconds: sys {t1 - t0:.0f}, user {t2 - t1:.0f}, critic {time.time() - t2:.0f}", flush=True)
            for st, (reward, labels) in zip(active, critic_out):
                st["rounds"].append({"k": k, "r": reward, "probs": labels})
                if reward > self.args.success_threshold or k == self.args.max_turns:
                    st["done"], st["k_end"], st["r_final"] = True, k, reward
            print(f"  round {k}: {len(active)} active -> {sum(not st['done'] for st in active)} continue, calls {self.calls}", flush=True)
        rows = []
        for st in states:
            avg_turn = st["k_end"] + 1
            rows.append({"context_id": st["ctx"]["id"], "j": st["j"], "response": st["first"], "r_final": st["r_final"], "k_end": st["k_end"],
                         "success": st["r_final"] > self.args.success_threshold, "r_sys": (st["r_final"] + 1.0 / avg_turn) / 2, "rounds": st["rounds"],
                         "simulated_turns": st["turns"][len(st["ctx"]["turns"]):]})
        return rows


async def main_async(args) -> None:
    contexts = read_jsonl(args.contexts)[:args.limit]
    out_path = args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = {(r["context_id"], r["j"]) for r in read_jsonl(out_path)} if out_path.exists() else set()
    jobs = [(c, j) for c in contexts for j in range(args.n_responses) if (c["id"], j) not in done]
    print(f"{len(jobs)} simulations to run ({len(done)} done), chunks of {args.chunk}", flush=True)
    sim = Simulator(args)
    started = time.time()
    finished = 0
    with out_path.open("a", encoding="utf-8") as handle:
        for start in range(0, len(jobs), args.chunk):
            chunk = jobs[start:start + args.chunk]
            try:
                rows = await sim.run_chunk(chunk)
            except Exception as error:  # the chunk is retried on the next invocation
                print("CHUNK ERROR", repr(error)[:300], flush=True)
                continue
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            finished += len(rows)
            print(f"{finished}/{len(jobs)} done, {time.time() - started:.0f}s, success rate of this chunk {sum(r['success'] for r in rows) / len(rows):.3f}", flush=True)
    print(f"finished {finished}/{len(jobs)} in {time.time() - started:.0f}s", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--contexts", type=Path, default=E / "rlff_repro/data/sim_contexts.jsonl")
    parser.add_argument("--out", type=Path, default=E / "rlff_repro/sim/simulations.jsonl")
    parser.add_argument("--sys-url", default="http://127.0.0.1:8031/v1")
    parser.add_argument("--sim-url", default="http://127.0.0.1:8030/v1")
    parser.add_argument("--n-responses", type=int, default=4)
    parser.add_argument("--max-turns", type=int, default=8)
    parser.add_argument("--critic-samples", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260930)
    parser.add_argument("--success-threshold", type=float, default=0.5)
    parser.add_argument("--chunk", type=int, default=800, help="simulations per lock-step chunk")
    parser.add_argument("--sys-concurrency", type=int, default=256)
    parser.add_argument("--sim-concurrency", type=int, default=256)
    parser.add_argument("--limit", type=int, default=10**9)
    asyncio.run(main_async(parser.parse_args()))
