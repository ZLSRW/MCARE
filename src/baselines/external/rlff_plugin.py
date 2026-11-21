"""ms-swift reward functions of the RLFF-ESC reproduction: R_rlff = R_fut + 0.5 * R_thk-fmt (paper Eq. 6, reward weights 1 and 0.5).

R_fut      P(goal reached | context, final response) from the Llama-3.2-1B classifier trained on the simulated data (rlff_rm.py)
R_thk-fmt  1 if the output follows <think>...</think><response>...</response>, else 0
The reward model reads only the final response (the <response> block); without one it reads the text after </think>, then the raw text.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import List

from swift.rewards import ORM, orms

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[2]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from baselines.external.esc_repro_common import format_reward  # noqa: E402

RESPONSE = re.compile(r"<response>(.*?)</response>", re.S)


def final_response(text: str) -> str:
    blocks = [b.strip() for b in RESPONSE.findall(text) if b.strip()]
    if blocks:
        return blocks[-1]
    if "</think>" in text and text.split("</think>")[-1].strip():
        return text.split("</think>")[-1].strip()
    return text.strip() or "..."


class RLFFFuture(ORM):
    _rm = None

    def __call__(self, completions, **kwargs) -> List[float]:
        if RLFFFuture._rm is None:
            from baselines.external.rlff_rm import RLFFRM

            RLFFFuture._rm = RLFFRM(Path(os.environ["RLFF_RM_PATH"]), "cuda")
        return [float(x) for x in RLFFFuture._rm.score(list(kwargs["history"]), [final_response(c) for c in completions])]


class RLFFFormat(ORM):
    def __call__(self, completions, **kwargs) -> List[float]:
        return [format_reward(c) for c in completions]


orms["rlff_future"] = RLFFFuture
orms["rlff_format"] = RLFFFormat
