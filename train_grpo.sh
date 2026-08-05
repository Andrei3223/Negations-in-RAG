#!/bin/bash
# GRPO launcher. Invoked by train_grpo.job (which does module load / conda
# activate / .env sourcing and sets the variables read below).
#
# All hyperparameters now live in verl/trainer/config/subsearch_grpo.yaml, which
# composes on top of ppo_trainer.yaml. Only per-run values (paths, retriever URL)
# are passed here. That config is handed verbatim to wandb.init(config=...) by
# RayPPOTrainer, so every knob is recorded with the run.
#
# Inspect the composed config without launching anything:
#   python3 -m verl.trainer.main_ppo --config-name subsearch_grpo --cfg job --resolve
#
# Override any parameter ad hoc, e.g.:
#   EXTRA_OVERRIDES="reward_model.schedule.beta=0.3 max_turns=2" bash train_grpo.sh
set -euo pipefail

# 1. Retriever endpoint, published by retrieval_launch.job
IP_FILE="$RETRIEVER_IP_PATH"
if [ -f "$IP_FILE" ]; then
    RETRIEVER_IP=$(cat "$IP_FILE")
    echo "Connecting to Retriever at: $RETRIEVER_IP"
else
    echo "ERROR: Retriever IP file not found at $IP_FILE"
    exit 1
fi

# 2. Caches on scratch, not the home quota
export HF_HOME="${HF_HOME:-/scratch-shared/$USER/hf_cache}"
export TRITON_CACHE_DIR="/scratch-shared/$USER/triton_cache"
export PYTHONUNBUFFERED=1
mkdir -p "$HF_HOME" "$TRITON_CACHE_DIR"

# 3. NCCL and GPU configuration
export NCCL_SOCKET_IFNAME=ib-bond0
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_P2P_DISABLE=1
export NCCL_TIMEOUT=1800
export NCCL_DEBUG=WARN
export CUDA_VISIBLE_DEVICES=0,1,2,3

# vllm + qwen2 with flash_attn has issues; XFORMERS is the working backend
export VLLM_ATTENTION_BACKEND=XFORMERS

# 4. Launch. Hydra resolves trainer.experiment_name from the values actually in
#    force, so the run name cannot drift from the config the way the old
#    shell-built EXPERIMENT_NAME did.
python3 -m verl.trainer.main_ppo \
    --config-name subsearch_grpo \
    data.train_files="$DATA_PATH/train.parquet" \
    data.val_files="$DATA_PATH/val.parquet" \
    actor_rollout_ref.model.path="$BASE_MODEL" \
    retriever.url="http://${RETRIEVER_IP}:8000/retrieve" \
    trainer.default_local_dir="$SAVE_DIR_BASE/\${trainer.experiment_name}" \
    ${EXTRA_OVERRIDES:-}
