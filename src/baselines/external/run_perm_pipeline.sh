#!/usr/bin/env bash
# PERM external baseline reproduction using GRPO with an OpenAI-compatible local judge.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
EXP_DIR="${MCARE_EXP_DIR:-$REPO_ROOT/outputs}/baselines/external/perm_repro"
BASE_MODEL="${MCARE_BASE_MODEL:-Qwen/Qwen3.5-9B}"
PERM_DIR="${PERM_REPO_DIR:-$REPO_ROOT/external/PERM/perm}"
PROMPTS="${PERM_PROMPTS:-1000}"
LR="${PERM_LR:-1e-5}"
OUT="${1:-$EXP_DIR/run_9b_lora}"
PERM_REWARD_URL="${PERM_REWARD_URL:-http://127.0.0.1:8020/v1}"
PERM_REWARD_MODEL="${PERM_REWARD_MODEL:-reward-model}"

mkdir -p "$OUT"
PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}" \
accelerate launch --config_file "$PERM_DIR/deepspeed_zero3.yaml" \
  --num_processes 1 "$SCRIPT_DIR/perm_grpo_trl29.py" \
  --train_data_path "$EXP_DIR/data/train.jsonl" \
  --num_train_samples "$PROMPTS" \
  --output_dir "$OUT" \
  --model_name "$BASE_MODEL" \
  --reward_model_type oai \
  --reward_generation_kwargs "{\"model_name\": \"$PERM_REWARD_MODEL\", \"api_key\": \"EMPTY\", \"base_url\": \"$PERM_REWARD_URL\"}" \
  --calculate_score_mode harmonic \
  --learning_rate "$LR" --beta 0.01 --epochs 1 \
  --use_lora --peft_r 16 --peft_alpha 32 --peft_dropout 0.1 \
  --per_device_train_batch_size 4 --gradient_accumulation_steps 4 --warmup_ratio 0.1 \
  --save_steps 100 --save_total_limit 10 --max_completion_length 1024 \
  --use_length_penalty --length_penalty_limit 768 --length_penalty_coef 0.001 \
  --use_standby_reward --standby_reward_coef 0.511
