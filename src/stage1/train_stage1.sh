#!/usr/bin/env bash
# Stage 1 MCARE Training: Need-First Policy (pi_1) via DPO + positive NLL.
# Hyperparameters (Section IV-C): Qwen3.5-9B, LoRA r=16 alpha=32 dropout=0.05
# on all linear layers, beta=0.1, loss_type=sigmoid, rpo_alpha=0.15, lr=2e-5,
# effective batch=16, 2 epochs.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

DATASET="${1:?Usage: $0 <stage1_pairs.jsonl> [output_dir] [seed]}"
OUT_DIR="${2:-$REPO_ROOT/outputs/stage1/pi1_s${3:-42}}"
SEED="${3:-42}"
BASE_MODEL="${MCARE_BASE_MODEL:-Qwen/Qwen3.5-9B}"
SWIFT_BIN="${SWIFT_BIN:-swift}"

mkdir -p "$OUT_DIR"

"$SWIFT_BIN" rlhf \
  --rlhf_type dpo \
  --model "$BASE_MODEL" \
  --dataset "$DATASET" \
  --output_dir "$OUT_DIR" \
  --tuner_type lora \
  --lora_rank 16 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  --target_modules all-linear \
  --torch_dtype bfloat16 \
  --beta 0.1 \
  --loss_type sigmoid \
  --rpo_alpha 0.15 \
  --num_train_epochs 2 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --learning_rate 2e-5 \
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
  --save_steps 500 \
  --save_total_limit 2 \
  --dataset_num_proc 8 \
  --dataloader_num_workers 2 \
  --split_dataset_ratio 0 \
  --seed "$SEED" \
  --data_seed "$SEED" \
  --report_to none
