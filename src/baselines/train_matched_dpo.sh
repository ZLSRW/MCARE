#!/usr/bin/env bash
# Matched-pool internal baselines: Vanilla-SFT, Joint-DPO, and SafeDPO.
# Usage: bash train_matched_dpo.sh <name> <sft|dpo|safe> <dataset.jsonl> [output_dir] [seed]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

NAME="${1:?Usage: $0 <name> <sft|dpo|safe> <dataset.jsonl> [output_dir] [seed]}"
KIND="${2:?}"
DATA="${3:?}"
OUT_DIR="${4:-$REPO_ROOT/outputs/baselines/$NAME}"
SEED="${5:-42}"
BASE_MODEL="${MCARE_BASE_MODEL:-Qwen/Qwen3.5-9B}"
SWIFT_BIN="${SWIFT_BIN:-swift}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

mkdir -p "$OUT_DIR"

if [[ "$KIND" == "sft" ]]; then
  CMD=("$SWIFT_BIN" sft)
  LR="${LR:-1e-4}"
  EXTRA=()
elif [[ "$KIND" == "dpo" ]]; then
  CMD=("$SWIFT_BIN" rlhf)
  LR="${LR:-5e-5}"
  EXTRA=(--rlhf_type dpo --beta 0.1 --loss_type sigmoid --label_smoothing 0)
elif [[ "$KIND" == "safe" ]]; then
  CMD=("$PYTHON_BIN" "$SCRIPT_DIR/safedpo_rlhf.py")
  LR="${LR:-5e-5}"
  EXTRA=(--rlhf_type dpo --beta 0.1 --loss_type sigmoid --label_smoothing 0)
else
  echo "Unknown kind: $KIND (expected sft|dpo|safe)" >&2
  exit 2
fi

PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}" \
"${CMD[@]}" "${EXTRA[@]}" \
  --model "$BASE_MODEL" \
  --dataset "$DATA" \
  --output_dir "$OUT_DIR" \
  --tuner_type lora \
  --lora_rank 16 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  --target_modules all-linear \
  --torch_dtype bfloat16 \
  --num_train_epochs 2 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --learning_rate "$LR" \
  --max_length 4096 \
  --gradient_checkpointing true \
  --attn_impl sdpa \
  --enable_thinking false \
  --loss_scale last_round+ignore_empty_think \
  --warmup_ratio 0.05 \
  --lr_scheduler_type cosine \
  --weight_decay 0.1 \
  --max_grad_norm 1 \
  --logging_steps 5 \
  --save_steps 1000 \
  --save_total_limit 2 \
  --dataset_num_proc 8 \
  --dataloader_num_workers 2 \
  --split_dataset_ratio 0 \
  --seed "$SEED" \
  --data_seed "$SEED" \
  --report_to none
