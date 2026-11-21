"""Exact LoRA adapter parameter interpolation baseline (Table VI: 0.75 * pi_E + 0.25 * pi_1).

Concatenates LoRA A rows and weight-scaled LoRA B columns across adapters so the
effective delta equals w_E * Delta_E + w_1 * Delta_1 exactly without SVD error,
doubling rank (r = 16 -> 32) and alpha (alpha = 32 -> 64) to preserve scaling
alpha / r = 2.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from common.config import get_paths
from common.io_utils import dump_json, sha256_file


def interpolate_lora_adapters(
    adapter_specs: list[tuple[Path, float]],
    out_dir: Path,
    base_model: str,
) -> dict[str, Any]:
    """Concatenate LoRA modules preserving exact weighted linear combinations."""
    configs = [json.loads((p / "adapter_config.json").read_text(encoding="utf-8")) for p, _ in adapter_specs]
    state_dicts = [load_file(str(p / "adapter_model.safetensors")) for p, _ in adapter_specs]
    keys = sorted(state_dicts[0].keys())
    for sd in state_dicts[1:]:
        assert sorted(sd.keys()) == keys, "Adapter tensor key sets do not match"

    merged: dict[str, torch.Tensor] = {}
    for k in keys:
        tensors = []
        for sd, weight in zip(state_dicts, [w for _, w in adapter_specs]):
            t = sd[k].float()
            if ".lora_A." in k:
                tensors.append(t)
            elif ".lora_B." in k:
                tensors.append(t * float(weight))
            else:
                raise ValueError(f"Unexpected LoRA parameter key: {k}")
        axis = 0 if ".lora_A." in k else 1
        merged[k] = torch.cat(tensors, dim=axis).to(state_dicts[0][k].dtype)

    n = len(adapter_specs)
    cfg0 = dict(configs[0])
    cfg0["r"] = int(cfg0["r"]) * n
    cfg0["lora_alpha"] = int(cfg0["lora_alpha"]) * n
    cfg0["base_model_name_or_path"] = base_model

    out_dir.mkdir(parents=True, exist_ok=True)
    save_file(merged, str(out_dir / "adapter_model.safetensors"))
    (out_dir / "adapter_config.json").write_text(json.dumps(cfg0, indent=2) + "\n", encoding="utf-8")

    meta = {
        "adapters": [{"path": str(p), "weight": w} for p, w in adapter_specs],
        "effective_rank": cfg0["r"],
        "effective_alpha": cfg0["lora_alpha"],
        "sha256": sha256_file(out_dir / "adapter_model.safetensors"),
    }
    dump_json(out_dir / "INTERPOLATION_COMPLETE.json", meta)
    return meta


def main() -> None:
    cfg = get_paths()
    parser = argparse.ArgumentParser(description="Exact LoRA adapter parameter interpolation (Table VI).")
    parser.add_argument("--specialist", type=Path, default=cfg.specialist_dir / "piE_s42" / "final")
    parser.add_argument("--stage1", type=Path, default=cfg.stage1_dir / "pi1_s42" / "final")
    parser.add_argument("--alpha-stage1", type=float, default=0.25, help="Weight on Stage-1 policy (default 0.25; 0.75 on pi_E)")
    parser.add_argument("--out-dir", type=Path, default=cfg.stage2_dir / "ablation_interpolation_075_025")
    args = parser.parse_args()

    meta = interpolate_lora_adapters(
        [(args.specialist, 1.0 - args.alpha_stage1), (args.stage1, args.alpha_stage1)],
        args.out_dir,
        cfg.base_model,
    )
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
