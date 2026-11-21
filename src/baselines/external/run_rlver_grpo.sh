#!/usr/bin/env bash
# RLVER multi-turn GRPO external baseline reproduction.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
D="${MCARE_EXP_DIR:-$REPO_ROOT/outputs}/baselines/external/rlver_repro"
BASE_MODEL="${MCARE_BASE_MODEL:-Qwen/Qwen3.5-9B}"
SWIFT_BIN="${SWIFT_BIN:-swift}"

STEPS="${RLVER_STEPS:-64}"
LR="${RLVER_LR:-5e-5}"
PROFILES_PER_STEP="${RLVER_GA:-16}"
BS="${RLVER_BS:-2}"
GA=$(( PROFILES_PER_STEP * 4 / BS ))
N_PROMPTS=$(( STEPS * PROFILES_PER_STEP ))
OUT="${1:-$D/grpo_9b}"
DATA="${RLVER_DATA:-$D/data/train_prompts.jsonl}"

mkdir -p "$OUT"
PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}" \
RLVER_LOG="$OUT/episodes.jsonl" \
RLVER_THINK="${RLVER_THINK:-0}" \
RLVER_FAKE="${RLVER_FAKE:-0}" \
RLVER_PROFILES="${RLVER_PROFILES:-$D/data/profiles.jsonl}" \
"$SWIFT_BIN" rlhf --rlhf_type grpo --model "$BASE_MODEL" --output_dir "$OUT/out" \
  --external_plugins "$SCRIPT_DIR/rlver_env.py" --multi_turn_scheduler gym_scheduler --gym_env rlver_env --max_turns 8 \
  --use_vllm true --vllm_mode colocate --vllm_gpu_memory_utilization 0.30 --vllm_max_model_len 6144 \
  --offload_model true --offload_optimizer true --tuner_type lora --lora_rank 32 --lora_alpha 64 --target_modules all-linear \
  --torch_dtype bfloat16 --attn_impl sdpa --dataset "${DATA}#${N_PROMPTS}" \
  --max_completion_length "${RLVER_MAX_COMPLETION:-320}" --max_length 5120 --num_train_epochs 1 \
  --per_device_train_batch_size "$BS" --gradient_accumulation_steps "$GA" --learning_rate "$LR" \
  --gradient_checkpointing true --save_steps 16 --logging_steps 1 --warmup_ratio 0.05 \
  --dataloader_num_workers 2 --dataset_num_proc 4 --num_generations 4 --temperature 1.0 --top_p 1.0 \
  --num_iterations 1 --beta 0.0 --enable_thinking false
