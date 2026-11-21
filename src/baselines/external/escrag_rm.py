"""ESCRAG-R1 reward model: frozen Qwen2.5-7B backbone + linear scoring head (paper Sec. 3.3, Appendix C).

Bradley-Terry loss on the released ESC-Preference pairs with the reward-centering term
    L = mean[ -log sigmoid(r+ - r-) + lambda * (r+ + r-)^2 ],  lambda = 0.01,
5 epochs, per-device batch 1 x gradient accumulation 16 (= 16 pairs per step), learning rate 5e-5. Because the backbone is frozen, the
last-token hidden states are extracted once and only the head is trained on the cached features.

  extract:  features for the (leak-filtered) train / valid pairs
  train:    head training + validation pairwise accuracy
The class EscragRM scores (dialogue history, candidate <think>/<response> output) pairs during GRPO.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[2]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
from baselines.external.esc_repro_common import E, RM_INSTRUCTION, ROOT, read_jsonl

BACKBONE = os.environ.get("MCARE_QWEN25_7B", "Qwen/Qwen2.5-7B")
DATA = Path(os.environ.get("ESCRAG_DATA_DIR", str(E / "escrag_repro/data")))
OUT = Path(os.environ.get("ESCRAG_RM_DIR", str(E / "escrag_repro/rm")))
MAX_LEN = 3072


def load_backbone(device: str):
    tokenizer = AutoTokenizer.from_pretrained(BACKBONE)
    tokenizer.padding_side = "right"
    model = AutoModel.from_pretrained(BACKBONE, torch_dtype=torch.bfloat16).to(device).eval()
    return tokenizer, model


def build_text(tokenizer, history: str, response: str) -> str:
    messages = [{"role": "user", "content": RM_INSTRUCTION + history.rstrip("\n")}, {"role": "assistant", "content": response.strip()}]
    return tokenizer.apply_chat_template(messages, tokenize=False)


@torch.no_grad()
def last_token_features(tokenizer, model, texts: list[str], device: str, token_budget: int = 20000) -> torch.Tensor:
    encoded = [tokenizer(t, truncation=True, max_length=MAX_LEN)["input_ids"] for t in texts]
    order = sorted(range(len(texts)), key=lambda i: len(encoded[i]))
    feats = torch.zeros(len(texts), model.config.hidden_size, dtype=torch.float32)
    start = 0
    while start < len(order):
        end, longest = start, 0
        while end < len(order) and max(longest, len(encoded[order[end]])) * (end - start + 1) <= token_budget:
            longest = max(longest, len(encoded[order[end]]))
            end += 1
        end = max(end, start + 1)
        batch = [encoded[i] for i in order[start:end]]
        width = max(len(b) for b in batch)
        ids = torch.full((len(batch), width), tokenizer.pad_token_id, dtype=torch.long)
        mask = torch.zeros((len(batch), width), dtype=torch.long)
        for row, b in enumerate(batch):
            ids[row, :len(b)] = torch.tensor(b)
            mask[row, :len(b)] = 1
        hidden = model(input_ids=ids.to(device), attention_mask=mask.to(device)).last_hidden_state
        last = mask.sum(dim=1) - 1
        pooled = hidden[torch.arange(len(batch), device=device), last.to(device)].float().cpu()
        for row, index in enumerate(order[start:end]):
            feats[index] = pooled[row]
        start = end
    return feats


def split_history(user_content: str) -> str:
    assert user_content.startswith(RM_INSTRUCTION), user_content[:200]
    return user_content[len(RM_INSTRUCTION):]


def extract(device: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    tokenizer, model = load_backbone(device)
    for split in ("train", "valid"):
        pairs = read_jsonl(DATA / f"preference_pairs_{split}.jsonl")
        texts = []
        for pair in pairs:
            history = split_history(pair["chosen"][0]["content"])
            texts.append(build_text(tokenizer, history, pair["chosen"][1]["content"]))
            texts.append(build_text(tokenizer, history, pair["rejected"][1]["content"]))
        started = time.time()
        feats = last_token_features(tokenizer, model, texts, device).view(len(pairs), 2, -1)
        torch.save(feats, OUT / f"features_{split}.pt")
        print(f"{split}: {len(pairs)} pairs, features {tuple(feats.shape)}, {time.time() - started:.0f}s", flush=True)


def train(device: str, lr: float, epochs: int, batch: int, lam: float, seed: int) -> None:
    torch.manual_seed(seed)
    random.seed(seed)
    tr, va = torch.load(OUT / "features_train.pt"), torch.load(OUT / "features_valid.pt")
    head = torch.nn.Linear(tr.shape[-1], 1).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=0.0)
    steps_per_epoch = math.ceil(len(tr) / batch)
    total = steps_per_epoch * epochs
    schedule = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: min(1.0, (s + 1) / max(1, int(0.05 * total))) * max(0.0, 1 - s / total))
    log = []

    def evaluate(feats: torch.Tensor) -> tuple[float, float]:
        with torch.no_grad():
            r = head(feats.to(device)).squeeze(-1)
            return float((r[:, 0] > r[:, 1]).float().mean()), float((r[:, 0] - r[:, 1]).mean())

    for epoch in range(epochs):
        order = torch.randperm(len(tr))
        for start in range(0, len(tr), batch):
            idx = order[start:start + batch]
            r = head(tr[idx].to(device)).squeeze(-1)
            loss = (-torch.nn.functional.logsigmoid(r[:, 0] - r[:, 1]) + lam * (r[:, 0] + r[:, 1]) ** 2).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            schedule.step()
        train_acc, train_margin = evaluate(tr)
        valid_acc, valid_margin = evaluate(va)
        log.append({"epoch": epoch + 1, "loss": float(loss), "train_acc": train_acc, "valid_acc": valid_acc, "train_margin": train_margin, "valid_margin": valid_margin})
        print(json.dumps(log[-1]), flush=True)
    torch.save({"weight": head.weight.detach().cpu(), "bias": head.bias.detach().cpu()}, OUT / "head.pt")
    (OUT / "train_log.json").write_text(json.dumps({"lr": lr, "epochs": epochs, "batch_pairs": batch, "lambda": lam, "seed": seed, "log": log}, indent=2) + "\n")


class EscragRM:
    """Scorer used inside the GRPO reward function."""

    def __init__(self, device: str = "cuda"):
        self.device = device
        self.tokenizer, self.model = load_backbone(device)
        head = torch.load(OUT / "head.pt")
        self.weight = head["weight"].to(device).float().squeeze(0)
        self.bias = head["bias"].to(device).float()

    @torch.no_grad()
    def score(self, histories: list[str], responses: list[str]) -> list[float]:
        texts = [build_text(self.tokenizer, h, r) for h, r in zip(histories, responses)]
        feats = last_token_features(self.tokenizer, self.model, texts, self.device, token_budget=12000).to(self.device)
        return (feats @ self.weight + self.bias).tolist()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("extract", "train"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--lam", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=20260930)
    args = parser.parse_args()
    extract(args.device) if args.stage == "extract" else train(args.device, args.lr, args.epochs, args.batch, args.lam, args.seed)
