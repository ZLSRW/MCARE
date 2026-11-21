"""Training prompt file for the RLVER environment: one row per training scenario (the simulator's first message is produced at reset time)."""
import json
import sys
from pathlib import Path

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[2]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
from common.config import get_paths as _get_paths
D = _get_paths().baselines_dir / "external/rlver_repro/data"
def main():
    profiles = [json.loads(line) for line in (D / "profiles.jsonl").open(encoding="utf-8")]
    train = [p for p in profiles if p["split"] == "train"]
    with (D / "train_prompts.jsonl").open("w", encoding="utf-8") as handle:
        for p in train:
            handle.write(json.dumps({"messages": [{"role": "user", "content": "(the conversation starts)"}], "profile_id": p["id"]}) + "\n")
    print({"profiles": len(profiles), "train_prompts": len(train), "test": len(profiles) - len(train)})



if __name__ == '__main__':
    main()
