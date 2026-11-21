"""Score complete English MCARE conversations with ESC-RANK adapters.

Uses the authors' English rubrics, but parses a single final digit strictly.
Ambiguous model output is recorded as unknown, never coerced into a score.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
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
BASE = _CFG.escrank_base_model
ADAPTERS = Path(_CFG.escrank_adapter_dir)
AUTHOR_SCORE = Path(_CFG.esc_eval_score)
BASE_REVISION = "c2ba64483dc50b3f8eb2d8271c4b9877a79ed2e2"
ADAPTER_REVISION = "450bf2eb5376c79e371aaf432925810243de1527"
DIMENSIONS = {
    "empathy": (2, "empathic_en"),
    "skill": (5, "tech_en"),
    "overall": (6, "overall_en"),
}


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def author_prompts(score_path: Path | None = None) -> list[str]:
    spec = importlib.util.spec_from_file_location("esc_eval_author_score", score_path or AUTHOR_SCORE)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot import author score.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.prompt_EN


def render_query(messages: list[dict], response: str, dimension: str, rubric: str) -> str:
    transcript = []
    for message in messages:
        if message["role"] == "system":
            continue
        role = "User" if message["role"] == "user" else "AI assistant"
        transcript.append(f"{role}: {message['content']}")
    transcript.append(f"AI assistant: {response}")
    return (
        rubric + "Dialogue between user and AI assistant:\n" + "\n\n".join(transcript).strip()
        + f"\nBased on the rules, give your {dimension} score (the number only) to the dialogue."
    )


def strict_score(raw: str) -> int | None:
    match = re.fullmatch(
        r"\s*(?:(?:Score\s*:\s*)|(?:The\s+)?(?:empathy|skill|overall|human\s+preference|emotional\s+knowledge)\s+score\s+is\s+)?([0-4])\.?\s*",
        raw,
        flags=re.I,
    )
    return int(match.group(1)) if match else None


def score_one(model, base, tokenizer, messages: list[dict], response: str, dimension: str, rubric: str, max_input_tokens: int, max_new_tokens: int) -> dict:
    import torch

    original = [dict(message) for message in messages]
    kept = original
    removed = 0
    while True:
        query = render_query(kept, response, dimension, rubric)
        encoded = base.build_inputs(tokenizer, query, history=[])
        input_ids = encoded["input_ids"]
        if input_ids.shape[-1] <= max_input_tokens:
            break
        # Keep the last user message and the newly generated assistant reply.
        first_non_system = 1 if kept and kept[0]["role"] == "system" else 0
        if len(kept) - first_non_system <= 1:
            raise ValueError("Latest user and rubric exceed score input limit")
        kept = kept[:first_non_system] + kept[first_non_system + 1:]
        removed += 1
    inputs = {key: value.to(model.device) for key, value in encoded.items() if torch.is_tensor(value)}
    eos_ids = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|im_end|>")]
    with torch.inference_mode():
        output = model.generate(
            **inputs, do_sample=False, max_new_tokens=max_new_tokens,
            eos_token_id=eos_ids, pad_token_id=tokenizer.pad_token_id,
            use_cache=True,
        )
    new_ids = output[0, input_ids.shape[-1]:]
    raw = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
    return {
        "raw": raw, "score": strict_score(raw), "input_tokens": int(input_ids.shape[-1]),
        "generated_tokens": int(new_ids.numel()), "dropped_history_messages": removed,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", type=Path, default=PANEL)
    parser.add_argument("--generation", action="append", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-input-tokens", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=12)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--m1-only", action="store_true", default=True)
    args = parser.parse_args()

    import torch
    import transformers
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if torch.cuda.device_count() != 1:
        raise RuntimeError("Expose exactly one free GPU with CUDA_VISIBLE_DEVICES")
    panel = read_jsonl(args.panel)
    selected = [(i, row) for i, row in enumerate(panel) if row["panel"] == "M1_natural"] if args.m1_only else list(enumerate(panel))
    if args.max_rows:
        selected = selected[:args.max_rows]
    tokenizer = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True, local_files_only=True)
    base = AutoModelForCausalLM.from_pretrained(
        BASE, trust_remote_code=True, local_files_only=True, torch_dtype=torch.float16,
        device_map={"": 0},
    ).eval()
    model = PeftModel.from_pretrained(base, ADAPTERS / "empathic_en", adapter_name="empathy", is_trainable=False)
    model.load_adapter(ADAPTERS / "tech_en", adapter_name="skill", is_trainable=False)
    model.load_adapter(ADAPTERS / "overall_en", adapter_name="overall", is_trainable=False)
    model.eval()
    rubrics = author_prompts()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print("loaded ESC-RANK with three English adapters", flush=True)

    for generation_path in args.generation:
        rows = read_jsonl(generation_path)
        if [row["id"] for row in rows] != [row["id"] for row in panel]:
            raise ValueError(f"Generation IDs do not match panel: {generation_path}")
        name = generation_path.stem
        suffix = f"_smoke{args.max_rows}" if args.max_rows else ""
        output_path = args.output_dir / f"esc_rank_{name}{suffix}.jsonl"
        manifest_path = args.output_dir / f"esc_rank_{name}{suffix}_manifest.json"
        existing = read_jsonl(output_path) if output_path.exists() else []
        if [row["id"] for row in existing] != [row["id"] for _, row in selected[:len(existing)]]:
            raise ValueError(f"Existing score file is not a panel prefix: {output_path}")
        started = time.monotonic()
        with output_path.open("a", encoding="utf-8") as handle:
            for index in range(len(existing), len(selected)):
                source_index, row = selected[index]
                values = {}
                for dimension, (rubric_index, adapter) in DIMENSIONS.items():
                    model.set_adapter(dimension)
                    values[dimension] = score_one(
                        model, base, tokenizer, row["messages"], rows[source_index]["response"],
                        dimension, rubrics[rubric_index], args.max_input_tokens, args.max_new_tokens,
                    )
                record = {"id": row["id"], "panel": row["panel"], "family": row["family"], "scores": values}
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                if (index + 1) % 20 == 0 or index + 1 == len(selected):
                    handle.flush()
                    print(f"{name}: {index+1}/{len(selected)} elapsed={time.monotonic()-started:.0f}s", flush=True)
        scored = read_jsonl(output_path)
        manifest = {
            "status": "complete" if not args.max_rows else "smoke_complete",
            "base_revision": BASE_REVISION, "adapter_revision": ADAPTER_REVISION,
            "local_modeling_internlm2_sha256": sha256(BASE / "modeling_internlm2.py"),
            "author_score_sha256": sha256(AUTHOR_SCORE),
            "transformers_version": transformers.__version__, "torch_version": torch.__version__,
            "panel_sha256": sha256(args.panel), "generation_sha256": sha256(generation_path),
            "scores_sha256": sha256(output_path), "n": len(scored),
            "scope": "M1 full dialogue" if args.m1_only else "M1 and M2 full dialogue",
            "parse": "anchored single 0-4 digit or one short explicit dimension-score sentence; free explanations are unknown",
            "max_input_tokens": args.max_input_tokens, "max_new_tokens": args.max_new_tokens,
            "dimensions": {
                dim: {
                    "valid": sum(record["scores"][dim]["score"] is not None for record in scored),
                    "mean_valid": (
                        sum(record["scores"][dim]["score"] for record in scored if record["scores"][dim]["score"] is not None)
                        / max(1, sum(record["scores"][dim]["score"] is not None for record in scored))
                    ),
                }
                for dim in DIMENSIONS
            },
            "note": "Predicted whole-dialogue scores, not new human satisfaction or single-turn task pass rates.",
        }
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"arm": name, "n": len(scored), "dimensions": manifest["dimensions"]}), flush=True)


if __name__ == "__main__":
    main()
