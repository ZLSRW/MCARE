#!/usr/bin/env bash
# ESCRAG-R1 external baseline reproduction (sft -> merge -> pregen -> grpo).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
D="${MCARE_EXP_DIR:-$REPO_ROOT/outputs}/baselines/external/escrag_repro"
BASE_MODEL="${MCARE_BASE_MODEL:-Qwen/Qwen3.5-9B}"
SWIFT_BIN="${SWIFT_BIN:-swift}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
STAGE="${1:?Usage: $0 <sft|merge|pregen|grpo>}"

mkdir -p "$D"
latest_ckpt() { ls -d "$1"/v0-*/checkpoint-* 2>/dev/null | sort -V | tail -1; }

case "$STAGE" in
  sft)
    "$SWIFT_BIN" sft --model "$BASE_MODEL" --dataset "$D/data/sft_demos.jsonl" --output_dir "$D/sft/out" \
      --tuner_type lora --lora_rank 32 --lora_alpha 64 --target_modules all-linear --torch_dtype bfloat16 --num_train_epochs 3 \
      --per_device_train_batch_size 2 --gradient_accumulation_steps 8 --learning_rate 1e-4 --max_length 4096 \
      --gradient_checkpointing true --attn_impl sdpa --warmup_ratio 0.05 --logging_steps 2 --save_steps 1000 \
      --dataset_num_proc 8 --dataloader_num_workers 2 --split_dataset_ratio 0
    ;;
  merge)
    CK="$(latest_ckpt "$D/sft/out")"
    "$SWIFT_BIN" export --adapters "$CK" --merge_lora true --output_dir "$D/merge/merged"
    ;;
  pregen)
    PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}" \
      "$PYTHON_BIN" "$SCRIPT_DIR/escrag_pregen.py" \
      --policy "$D/merge/merged" --family qwen35 \
      --prompts "$D/data/grpo_prompts.jsonl" --out "$D/data/grpo_rag.jsonl" --thinking
    ;;
  grpo)
    PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}" \
    "$SWIFT_BIN" rlhf --rlhf_type grpo --model "$D/merge/merged" --output_dir "$D/grpo/out" \
      --external_plugins "$SCRIPT_DIR/escrag_plugin.py" --reward_funcs escrag_rm escrag_format --reward_weights 1.0 1.0 \
      --use_vllm true --vllm_mode colocate --vllm_gpu_memory_utilization 0.30 --vllm_max_model_len 7168 \
      --offload_model true --offload_optimizer true --tuner_type lora --lora_rank 32 --lora_alpha 64 --target_modules all-linear \
      --torch_dtype bfloat16 --attn_impl sdpa --dataset "$D/data/grpo_rag.jsonl" --max_completion_length 512 \
      --max_length 7168 --num_train_epochs 1 --per_device_train_batch_size 1 --gradient_accumulation_steps 16 \
      --learning_rate "${ESCRAG_LR:-2e-5}" --gradient_checkpointing true --save_steps 100 --logging_steps 5 --warmup_ratio 0.05 \
      --dataloader_num_workers 2 --dataset_num_proc 8 --num_generations 4 --temperature 0.9 --top_p 0.95 --num_iterations 1 --beta 0.05
    ;;
  *)
    echo "Unknown stage: $STAGE" >&2
    exit 2
    ;;
esac
