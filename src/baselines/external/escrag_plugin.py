"""ms-swift reward functions of the ESCRAG-R1 reproduction: r = r_RM + r_format (paper Eq. 5, format_reward_weight = scale = 1.0 in the released script).

r_RM   frozen Qwen2.5-7B backbone + linear head trained on the ESC-Preference pairs (escrag_rm.py), scored on the full <think>/<response> output
r_fmt  1.0 if <think>..</think><response>..</response> appear in this order, else 0.0 (released compute_format_reward)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List

from swift.rewards import ORM, orms

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[2]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from baselines.external.esc_repro_common import format_reward, normalize_output  # noqa: E402


class EscragRMReward(ORM):
    _rm = None

    def __call__(self, completions, **kwargs) -> List[float]:
        if EscragRMReward._rm is None:
            from baselines.external.escrag_rm import EscragRM

            EscragRMReward._rm = EscragRM("cuda")
        histories = kwargs["history"]
        return [float(x) for x in EscragRMReward._rm.score(list(histories), [normalize_output(c) for c in completions])]


class EscragFormatReward(ORM):
    def __call__(self, completions, **kwargs) -> List[float]:
        return [format_reward(c) for c in completions]


orms["escrag_rm"] = EscragRMReward
orms["escrag_format"] = EscragFormatReward
