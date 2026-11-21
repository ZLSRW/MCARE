#!/usr/bin/env bash
# PEER-SFT & PEER-RUR English text-only baseline reproduction.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
D="${MCARE_EXP_DIR:-$REPO_ROOT/outputs}/baselines/external/peer_repro_clean"
BASE_MODEL="${MCARE_BASE_MODEL:-Qwen/Qwen3.5-9B}"
EMB_MODEL="${MCARE_MINILM_MODEL:-sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2}"
PEER_PLUGIN="${PEER_PLUGIN_PATH:-$REPO_ROOT/external/PEER/peer_grpo/plugin.py}"
SWIFT_BIN="${SWIFT_BIN:-swift}"
STAGE="${1:?Usage: $0 <unifireward|sft|merge|grpo> [grpo_prompts]}"
GRPO_PROMPTS="${2:-1000}"

mkdir -p "$D"
latest_ckpt() { ls -d "$1"/v0-*/checkpoint-* 2>/dev/null | sort -V | tail -1; }

case "$STAGE" in
  unifireward)
    "$SWIFT_BIN" sft --model "$BASE_MODEL" --dataset "$D/data/ser_en_text_train.jsonl" --output_dir "$D/unifireward/out" \
      --torch_dtype bfloat16 --tuner_type lora --lora_rank 16 --lora_alpha 64 --target_modules all-linear --num_train_epochs 1 \
      --per_device_train_batch_size 1 --gradient_accumulation_steps 32 --attn_impl sdpa --learning_rate 2e-4 \
      --gradient_checkpointing true --save_steps 1000 --logging_steps 5 --max_length 4096 --warmup_ratio 0.05 \
      --dataset_num_proc 8 --dataloader_num_workers 4 --split_dataset_ratio 0
    ;;
  sft)
    "$SWIFT_BIN" sft --model "$BASE_MODEL" --dataset "$D/data/sft_text_en.jsonl" --output_dir "$D/sft/out" \
      --tuner_type lora --lora_rank 32 --lora_alpha 64 --target_modules all-linear --torch_dtype bfloat16 --num_train_epochs 3 \
      --per_device_train_batch_size 2 --gradient_accumulation_steps 12 --learning_rate 1e-4 --max_length 8192 \
      --gradient_checkpointing true --attn_impl sdpa --warmup_ratio 0.05 --logging_steps 5 --save_steps 1000 \
      --dataset_num_proc 8 --dataloader_num_workers 4 --split_dataset_ratio 0
    ;;
  merge)
    "$SWIFT_BIN" export --adapters "$(latest_ckpt "$D/sft/out")" --merge_lora true --output_dir "$D/sft/merged"
    ;;
  grpo)
    REWARD_CKPT="$(latest_ckpt "$D/unifireward/out")"
    "$SWIFT_BIN" deploy --model "$BASE_MODEL" --adapters "$REWARD_CKPT" --infer_backend pt --max_new_tokens 4096 \
      --attn_impl sdpa --torch_dtype bfloat16 --served_model_name Qwen3-VL-8B-Unified --port "${PEER_REWARD_PORT:-8013}" &
    SERVER_PID=$!
    trap 'kill $SERVER_PID 2>/dev/null || true' EXIT
    until curl -s -m 3 "http://127.0.0.1:${PEER_REWARD_PORT:-8013}/v1/models" | grep -q "Qwen3-VL-8B-Unified"; do sleep 5; done
    MODEL_API_ADDRESS=127.0.0.1 MODEL_API_PORT="${PEER_REWARD_PORT:-8013}" EMB_MODEL="$EMB_MODEL" \
    USE_REWEIGHT=1 REDUNDANCY_ALPHA=0.5 REDUNDANCY_BETA=5 GROUP_SIZE=4 \
    "$SWIFT_BIN" rlhf --rlhf_type grpo --model "$D/sft/merged" --output_dir "$D/grpo_rur/out" \
      --external_plugins "$PEER_PLUGIN" --reward_funcs UnifiedReward OurFormat \
      --use_vllm true --vllm_mode colocate --vllm_gpu_memory_utilization 0.35 --vllm_max_model_len 8192 \
      --offload_model true --offload_optimizer true --tuner_type lora --lora_rank 32 --lora_alpha 64 --target_modules all-linear \
      --torch_dtype bfloat16 --attn_impl sdpa --dataset "$D/data/grpo_text_en.jsonl#$GRPO_PROMPTS" --max_completion_length 2048 \
      --max_length 8192 --num_train_epochs 1 --per_device_train_batch_size 1 --gradient_accumulation_steps 8 --learning_rate 5e-6 \
      --gradient_checkpointing true --save_steps 250 --logging_steps 5 --warmup_ratio 0.05 --num_generations 4 \
      --temperature 1.0 --num_iterations 2 --beta 0
    ;;
  *)
    echo "Unknown stage: $STAGE" >&2
    exit 2
    ;;
esac
