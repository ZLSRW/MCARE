"""RLFF-ESC future-oriented reward model (paper Sec. 3.3.1, Appendix B): a Llama-3.2-1B binary classifier trained on the simulated data D_r.

Input template (Table 5): 'Instruction: ...\\nConversation: {context}\\nTherapist's response: {response}'; label y = 1 iff r_sys > delta.
The paper does not give delta; with the goal threshold 0.5 on the mean critic score (multiples of 0.025) and r_sys = (r + 1/AvgT)/2 with AvgT <= T + 1 = 9,
delta = 0.31 separates 'goal reached within T turns' (r_sys >= 0.318) from 'not reached' (r_sys <= 0.3056) exactly. R_fut = P(y = 1).
Hyper-parameters follow Table 8 (batch 1, gradient accumulation 8, 2 epochs, max length 2048); the learning rate is selectable because the paper's 1e-4
is a head-only value that is unstable for full fine-tuning (the choice is made on validation AUC and reported).
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[2]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
from baselines.external.esc_repro_common import E, ROOT, read_jsonl

BACKBONE = os.environ.get("MCARE_LLAMA32_1B", "meta-llama/Llama-3.2-1B")
SIM = E / "rlff_repro/sim"
SIM_FILE = Path(os.environ.get("RLFF_SIM_FILE", SIM / "simulations.jsonl"))
OUT = E / "rlff_repro/rm"
DELTA = 0.31
MAX_LEN = 2048
INSTRUCTION = ("Instruction: You are a helpful assistant tasked with evaluating whether a patient's emotional problem has been effectively "
               "addressed following a conversation with a therapist.\n")


def context_text(turns: list[dict]) -> str:
    return "\n".join(f"{'Patient' if t['role'] == 'user' else 'Therapist'}: {t['content'].strip()}" for t in turns)


def build_input(turns: list[dict], response: str) -> str:
    return f"{INSTRUCTION}Conversation: {context_text(turns)}\nTherapist's response: {response.strip()}"


def load_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(BACKBONE)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def dataset() -> tuple[list[dict], list[dict]]:
    contexts = {c["id"]: c for c in read_jsonl(E / "rlff_repro/data/sim_contexts.jsonl")}
    rows = []
    for r in read_jsonl(SIM_FILE):
        rows.append({"context_id": r["context_id"], "text": build_input(contexts[r["context_id"]]["turns"], r["response"]),
                     "label": int(r["r_sys"] > DELTA), "r_sys": r["r_sys"]})
    ids = sorted({r["context_id"] for r in rows})
    random.Random(7).shuffle(ids)
    valid_ids = set(ids[:max(1, len(ids) // 10)])
    return [r for r in rows if r["context_id"] not in valid_ids], [r for r in rows if r["context_id"] in valid_ids]


@torch.no_grad()
def predict(model, tokenizer, texts: list[str], device: str, batch_size: int = 16) -> np.ndarray:
    model.eval()
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    out = np.zeros(len(texts), dtype=np.float32)
    for start in range(0, len(order), batch_size):
        idx = order[start:start + batch_size]
        batch = tokenizer([texts[i] for i in idx], padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt").to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(**batch).logits.float()
        out[idx] = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
    return out


def train(lr: float, epochs: int, accumulation: int, device: str, tag: str, seed: int) -> dict:
    torch.manual_seed(seed)
    random.seed(seed)
    tokenizer = load_tokenizer()
    train_rows, valid_rows = dataset()
    model = AutoModelForSequenceClassification.from_pretrained(BACKBONE, num_labels=2, torch_dtype=torch.float32).to(device)
    model.config.pad_token_id = tokenizer.pad_token_id
    model.gradient_checkpointing_enable()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    total = epochs * len(train_rows) // accumulation
    schedule = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: min(1.0, (s + 1) / max(1, int(0.05 * total))) * max(0.0, 1 - s / max(1, total)))
    log = []
    step = 0
    for epoch in range(epochs):
        order = list(range(len(train_rows)))
        random.shuffle(order)
        model.train()
        running = []
        for n, i in enumerate(order, start=1):
            batch = tokenizer(train_rows[i]["text"], truncation=True, max_length=MAX_LEN, return_tensors="pt").to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(**batch).logits.float()
            loss = torch.nn.functional.cross_entropy(logits, torch.tensor([train_rows[i]["label"]], device=device))
            (loss / accumulation).backward()
            running.append(float(loss))
            if n % accumulation == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                schedule.step()
                optimizer.zero_grad()
                step += 1
        probs = predict(model, tokenizer, [r["text"] for r in valid_rows], device)
        labels = np.array([r["label"] for r in valid_rows])
        auc = float(roc_auc_score(labels, probs)) if 0 < labels.sum() < len(labels) else float("nan")
        log.append({"epoch": epoch + 1, "train_loss": float(np.mean(running)), "valid_auc": auc, "valid_acc": float(((probs > 0.5) == labels).mean()),
                    "valid_pos_rate": float(labels.mean()), "mean_prob": float(probs.mean())})
        print(json.dumps(log[-1]), flush=True)
    target = OUT / tag
    target.mkdir(parents=True, exist_ok=True)
    model.to(torch.bfloat16).save_pretrained(target)
    tokenizer.save_pretrained(target)
    result = {"tag": tag, "lr": lr, "epochs": epochs, "accumulation": accumulation, "train_rows": len(train_rows), "valid_rows": len(valid_rows),
              "positive_rate_train": float(np.mean([r["label"] for r in train_rows])), "log": log}
    (target / "train_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


class RLFFRM:
    """Scorer used inside the GRPO reward function: R_fut = P(goal reached | context, response)."""

    def __init__(self, path: Path, device: str = "cuda"):
        self.device = device
        self.tokenizer = load_tokenizer()
        self.model = AutoModelForSequenceClassification.from_pretrained(path, torch_dtype=torch.bfloat16).to(device).eval()
        self.model.config.pad_token_id = self.tokenizer.pad_token_id

    def score(self, contexts: list[str], responses: list[str]) -> list[float]:
        texts = [f"{INSTRUCTION}Conversation: {c.strip()}\nTherapist's response: {r.strip()}" for c, r in zip(contexts, responses)]
        return predict(self.model, self.tokenizer, texts, self.device).tolist()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--accumulation", type=int, default=8)
    parser.add_argument("--tag", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260930)
    args = parser.parse_args()
    train(args.lr, args.epochs, args.accumulation, args.device, args.tag or f"lr{args.lr:g}", args.seed)
