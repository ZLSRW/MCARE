#!/usr/bin/env bash
# RLFF-ESC external baseline reproduction (sim -> rm -> grpo).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
EXP_DIR="${MCARE_EXP_DIR:-$REPO_ROOT/outputs}/baselines/external/rlff_repro"
BASE_MODEL="${MCARE_BASE_MODEL:-Qwen/Qwen3.5-9B}"
SWIFT_BIN="${SWIFT_BIN:-swift}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

STAGE="${1:?Usage: $0 <sim|rm|grpo> ...}"
mkdir -p "$EXP_DIR"

case "$STAGE" in
  sim)
    PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}" \
      "$PYTHON_BIN" -u "$SCRIPT_DIR/rlff_simulate.py" "${@:2}"
    ;;
  rm)
    LR="${2:?Usage: $0 rm <lr> [tag]}"
    TAG="${3:-lr${LR}}"
    PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}" \
      "$PYTHON_BIN" -u "$SCRIPT_DIR/rlff_rm.py" --lr "$LR" --tag "$TAG"
    ;;
  grpo)
    OUT="${2:-$EXP_DIR/grpo_9b}"
    RM="${RLFF_RM_PATH:-$EXP_DIR/rm/best}"
    mkdir -p "$OUT"
    PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}" RLFF_RM_PATH="$RM" \
    "$SWIFT_BIN" rlhf --rlhf_type grpo \
      --model "$BASE_MODEL" --output_dir "$OUT/out" \
      --external_plugins "$SCRIPT_DIR/rlff_plugin.py" --reward_funcs rlff_future rlff_format --reward_weights 1.0 0.5 \
      --use_vllm true --vllm_mode colocate --vllm_gpu_memory_utilization 0.35 --vllm_max_model_len 6144 \
      --offload_model true --offload_optimizer true --tuner_type lora --lora_rank 8 --lora_alpha 32 --target_modules all-linear \
      --torch_dtype bfloat16 --attn_impl sdpa --dataset "$EXP_DIR/data/rl_prompts.jsonl" --max_completion_length 512 \
      --max_length 6144 --num_train_epochs 1 --per_device_train_batch_size 1 --gradient_accumulation_steps 16 \
      --learning_rate "${RLFF_LR:-1e-5}" --gradient_checkpointing true --save_steps 100 --logging_steps 5 --warmup_ratio 0.05 \
      --dataloader_num_workers 2 --dataset_num_proc 8 --num_generations 4 --temperature 1.1 --top_p 1.0 --top_k 80 \
      --num_iterations 1 --beta 0.04 --enable_thinking false
    ;;
  *)
    echo "Unknown stage: $STAGE (expected sim|rm|grpo)" >&2
    exit 2
    ;;
esac
