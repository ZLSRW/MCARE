"""Score frozen MCARE generations with ArmoRM's four raw reward heads.

Each arm uses the same original history for evaluation. The generation prompt
is intentionally not given to this external evaluator. Results are resumable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path


import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
from common.config import get_paths as _get_paths
_CFG = _get_paths()
ROOT = _CFG.root
PANEL = _CFG.eval_dir / "panel_all.jsonl"
MODEL = _CFG.armorm_model
REVISION = "eb2676d20da2f2d41082289d23c59b9f7427f955"
HEADS = {"helpfulness": 0, "correctness": 1, "coherence": 2, "instruction_following": 6}


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def encode_preserving_reply(tokenizer, history: list[dict], response: str, max_length: int):
    # Remove the oldest complete message(s), keeping the system and latest user.
    system = history[:1] if history and history[0]["role"] == "system" else []
    others = history[len(system):]
    removed = 0
    while True:
        messages = system + others[removed:] + [{"role": "assistant", "content": response}]
        encoded = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
        # Transformers 5 returns a BatchEncoding here; older releases returned
        # the input_ids list directly.
        ids = encoded["input_ids"] if hasattr(encoded, "__getitem__") and "input_ids" in encoded else encoded
        if len(ids) <= max_length:
            return ids, removed
        if len(others) - removed <= 1:
            raise ValueError(f"Latest user plus response exceeds {max_length} tokens")
        removed += 1


def score_one(model, tokenizer, history: list[dict], response: str, max_length: int) -> dict:
    import torch

    ids, removed = encode_preserving_reply(tokenizer, history, response, max_length)
    inputs = torch.tensor([ids], dtype=torch.long, device=model.device)
    with torch.inference_mode():
        output = model(inputs)
    rewards = output.rewards[0].detach().float().cpu()
    if rewards.numel() < 7:
        raise ValueError(f"ArmoRM returned {rewards.numel()} reward heads, expected at least 7")
    values = {name: float(rewards[index]) for name, index in HEADS.items()}
    if not all(math.isfinite(value) for value in values.values()):
        raise ValueError("Nonfinite ArmoRM reward")
    return {"raw_rewards": values, "tokens": len(ids), "dropped_history_messages": removed}


def patch_armorm_transformers_compat() -> None:
    """Keep ArmoRM's remote class importable on Transformers 5.5+.

    The missing symbol is used only as a docstring decorator argument in the
    published ArmoRM source; an empty docstring preserves model computation.
    """
    from transformers.models.llama import modeling_llama

    if not hasattr(modeling_llama, "LLAMA_INPUTS_DOCSTRING"):
        modeling_llama.LLAMA_INPUTS_DOCSTRING = ""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", type=Path, default=PANEL)
    parser.add_argument("--generation", action="append", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--max-rows", type=int, default=0, help="Positive value runs a smoke subset; zero means all rows")
    args = parser.parse_args()

    import torch
    import transformers
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    if torch.cuda.device_count() != 1:
        raise RuntimeError("Expose exactly one free GPU with CUDA_VISIBLE_DEVICES")
    patch_armorm_transformers_compat()
    panel = read_jsonl(args.panel)
    ids = [row["id"] for row in panel]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate panel IDs")
    tokenizer = AutoTokenizer.from_pretrained(MODEL, use_fast=True, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL, trust_remote_code=True, local_files_only=True,
        device_map={"": 0}, torch_dtype=torch.bfloat16,
    )
    model.eval()
    print(f"loaded ArmoRM on GPU {torch.cuda.get_device_name(0)}", flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for generation_path in args.generation:
        rows = read_jsonl(generation_path)
        if [row["id"] for row in rows] != ids:
            raise ValueError(f"Generation IDs do not match panel: {generation_path}")
        name = generation_path.stem
        suffix = f"_smoke{args.max_rows}" if args.max_rows else ""
        output_path = args.output_dir / f"armorm_{name}{suffix}.jsonl"
        manifest_path = args.output_dir / f"armorm_{name}{suffix}_manifest.json"
        expected = rows[:args.max_rows] if args.max_rows else rows
        existing = read_jsonl(output_path) if output_path.exists() else []
        if [row["id"] for row in existing] != [row["id"] for row in expected[:len(existing)]]:
            raise ValueError(f"Existing score file has different ID order: {output_path}")
        started = time.monotonic()
        with output_path.open("a", encoding="utf-8") as handle:
            for index in range(len(existing), len(expected)):
                row = expected[index]
                result = score_one(model, tokenizer, panel[index]["messages"], row["response"], args.max_length)
                record = {"id": row["id"], "panel": row["panel"], "family": row["family"], **result}
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                if (index + 1) % 25 == 0 or index + 1 == len(expected):
                    handle.flush()
                    print(f"{name}: {index+1}/{len(expected)} elapsed={time.monotonic()-started:.0f}s", flush=True)
        scored = read_jsonl(output_path)
        mean = {key: sum(row["raw_rewards"][key] for row in scored) / len(scored) for key in HEADS}
        manifest = {
            "status": "complete" if len(scored) == len(rows) and not args.max_rows else "smoke_complete",
            "model": "RLHFlow/ArmoRM-Llama3-8B-v0.1", "revision": REVISION,
            "transformers_version": transformers.__version__, "torch_version": torch.__version__,
            "panel_sha256": sha256(args.panel), "generation_sha256": sha256(generation_path),
            "scores_sha256": sha256(output_path), "n": len(scored), "max_length": args.max_length,
            "dropped_history_rows": sum(row["dropped_history_messages"] > 0 for row in scored),
            "mean_raw_rewards": mean,
            "note": "Raw predicted reward heads; not percentages, human ratings, or task-specific pass rates.",
        }
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"arm": name, "n": len(scored), "mean_raw_rewards": mean}), flush=True)


if __name__ == "__main__":
    main()
