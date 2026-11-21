"""Cut the PEER release down to its English text-only parts for a text-only adaptation.

PEER's original pipeline is multimodal (Qwen3-VL, facial frames). The base models used here are text-only, so rows
that reference images and rows containing Chinese are dropped; nothing else is changed. The row counts before and
after each filter are written to manifest.json so the adaptation is auditable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[2]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
from common.config import get_paths as _get_paths
_CFG = _get_paths()
ROOT = _CFG.root
SOURCE = _CFG.data_dir / "external_datasets/PEER-data"
CJK = re.compile(r"[\u3400-\u9fff]")


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write(path: Path, rows: list[dict]) -> str:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def has_cjk(row: dict) -> bool:
    return bool(CJK.search(json.dumps(row, ensure_ascii=False)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    manifest = {"source": str(SOURCE.relative_to(ROOT)), "license": "cc-by-nc-4.0", "files": {}}

    sft = read_jsonl(SOURCE / "sft/sft_train.jsonl")
    kept = [{"messages": row["messages"]} for row in sft if not row.get("images") and "<image>" not in json.dumps(row) and not has_cjk(row)]
    manifest["files"]["sft_text_en.jsonl"] = {"before": len(sft), "after": len(kept), "sha256": write(out / "sft_text_en.jsonl", kept)}

    for split in ("train", "test"):
        ser = read_jsonl(SOURCE / f"ser/en/text_{split}.jsonl")
        kept = [{"messages": row["messages"]} for row in ser if not has_cjk(row)]
        manifest["files"][f"ser_en_text_{split}.jsonl"] = {"before": len(ser), "after": len(kept), "sha256": write(out / f"ser_en_text_{split}.jsonl", kept)}

    grpo = read_jsonl(SOURCE / "grpo/textual.jsonl")
    kept = [row for row in grpo if row.get("language") == "english" and row.get("modality") == "textual" and not row.get("images") and not has_cjk(row)]
    manifest["files"]["grpo_text_en.jsonl"] = {"before": len(grpo), "after": len(kept), "sha256": write(out / "grpo_text_en.jsonl", kept)}

    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
