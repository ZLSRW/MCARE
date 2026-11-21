"""Shared helpers for the reproduced RLFF-ESC / ESCRAG-R1 baselines.

Both papers use the same therapist role-play system prompt (RLFF-ESC Table 6, ESCRAG-R1 Appendix B.2) with a
<think></think><response></response> output format, so the prompt, the transcript layout and the format checks live here.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[2]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
import os
from common.config import get_paths as _get_paths

_CFG = _get_paths()
ROOT = _CFG.root
E = _CFG.baselines_dir / "external"
POLICY = _CFG.policy_dir
PANEL = _CFG.eval_dir / "panel_all.jsonl"
ESCRAG_REPO = Path(os.environ.get("ESCRAG_REPO_DIR", str(ROOT / "external/ESCRAG-R1")))

# Identical text in RLFF-ESC (Table 6) and ESCRAG-R1 (rag-generation-system.txt / esconv.py); the RLFF-ESC table ends with " ." after the tag example.
THERAPIST_SYSTEM = (
    "Now enter the role-playing mode. In the following conversation, you will play as a therapist in a counselling conversation "
    "with a patient. Your goal is to help the patient reduce their emotional distress and support them working through their "
    "challenges. You first think about the reasoning process in the mind and then provide the patient with the response. The "
    "reasoning process and response are enclosed within <think> </think> and <response> </response> tags, respectively, i.e., "
    "<think> reasoning process here </think> <response> supportive response here </response>"
)

# ESCRAG-R1 reward-model input (ESC-Preference pairs): user = instruction + transcript, assistant = candidate <think>/<response> output
RM_INSTRUCTION = (
    "Instruction: You are a helpful assistant tasked with evaluating whether a patient's emotional problem has been effectively "
    "addressed following a conversation with a therapist.\nConversation History:"
)


def read_jsonl(path: Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def transcript(turns: list[dict], trailing_newline: bool = True) -> str:
    """ESCRAG-R1 esconv.py layout: 'Client: ...\\nCounselor: ...\\n' (user = Client, assistant = Counselor)."""
    text = "".join(f"{'Client' if t['role'] == 'user' else 'Counselor'}: {t['content'].strip()}\n" for t in turns)
    return text if trailing_newline else text.rstrip("\n")


def panel_excluded_parents() -> set[str]:
    panel = read_jsonl(PANEL)
    return {row["family"].split(":", 1)[1] for row in panel} | {row["reference_parent_id"] for row in panel if row.get("reference_parent_id")}


def normalize_output(text: str) -> str:
    """Qwen3.5 in thinking mode continues after the '<think>\\n' that its generation prompt already contains, so a completion is
    'reasoning</think>\\n\\n<response>..</response>'. Restore the opening tag so the think/response protocol can be checked and scored uniformly."""
    if text and "<think>" not in text and "</think>" in text:
        return "<think>\n" + text.lstrip()
    return text


def format_reward(text: str) -> float:
    """ESCRAG-R1 naive.py compute_format_reward: 1.0 only when <think>..</think><response>..</response> appear in this order."""
    text = normalize_output(text)
    if not text:
        return 0.0
    a, b, c, d = (text.find(tag) for tag in ("<think>", "</think>", "<response>", "</response>"))
    return 1.0 if 0 <= a < b < c < d else 0.0


THINK_RESPONSE = re.compile(r"<response>(.*?)</response>", re.S)


def extract_response(text: str) -> str | None:
    """Text of the last complete <response> block; None when the block is missing."""
    hits = THINK_RESPONSE.findall(text or "")
    return hits[-1].strip() if hits else None
