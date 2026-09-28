#!/bin/bash
# Shared configuration for the SLURM jobs in run/. Sourced by run/retrieval.job
# and run/train.job -- edit paths here, not in the job files, so the retriever
# and the trainer can never disagree about where things live.
#
# Not a standalone script: it expects to be sourced.

export REPO_ROOT="${REPO_ROOT:-/home/16207041/Negations-in-RAG}"

# --- Secrets: HF_TOKEN, WANDB_API_KEY (+ Azure keys used by run/eval.py) ------
set -a
if   [ -f "$REPO_ROOT/run/.env" ]; then source "$REPO_ROOT/run/.env"
elif [ -f "$REPO_ROOT/.env" ];     then source "$REPO_ROOT/.env"
fi
set +a

# --- Conda ---------------------------------------------------------------------
source /home/16207041/miniconda3/etc/profile.d/conda.sh
export TRAIN_ENV=searchr1        # torch 2.4 / vllm 0.6.3 / ray / verl (editable)
export RETRIEVER_ENV=retriever   # faiss-gpu 1.8 / pyserini / fastapi

# --- Wiki-18 corpus + FAISS index (shared project space; 60G index, 14G jsonl) --
export RETRIEVAL_DATA="/fnwi_fs/ivi/irlab/projects/chatr1/collection"

# --- Training data (parquet built by scripts/data_process) ----------------------
# _upsampled is the one reward_model.data_sources in ppo_trainer.yaml documents:
# 189k rows, nevir-with-answers + negations-free-sentence-with-answer upsampled
# 11x over the non-upsampled build. Swap to combined_nq_hotpot_nevir_negfree for
# the natural mix.
export DATA_PATH="$REPO_ROOT/data/combined_nq_hotpot_nevir_negfree_upsampled"

# --- Caches on shared storage, not the 100G home quota --------------------------
# The wiki-18 Arrow build the retriever needs is already materialised under
# $HF_HOME/datasets (~27G), as are Qwen2.5-3B, e5-base-v2 and MiniLM-L12-v2.
export HF_HOME="/fnwi_fs/ivi/irlab/personal/16207041/hf_cache"
export TRITON_CACHE_DIR="/fnwi_fs/ivi/irlab/personal/16207041/triton_cache"

# --- Model and checkpoint destination -------------------------------------------
export BASE_MODEL="Qwen/Qwen2.5-3B"
export SAVE_DIR_BASE="/fnwi_fs/ivi/irlab/personal/16207041/verl_checkpoints"

# --- Retriever handshake --------------------------------------------------------
# retrieval.job writes "http://<node-ip>:<port>" here once it has reserved a port;
# train.job and smoke.job poll it. The port is deliberately NOT fixed here: the
# retriever derives one from its job id and walks forward if it is taken, because
# a hard-coded 8000 collided with other users' services (job 347161). Export
# RETRIEVER_PORT yourself only to pin the starting port.
export RETRIEVER_IP_PATH="$REPO_ROOT/retriever_url.txt"

mkdir -p "$HF_HOME" "$TRITON_CACHE_DIR" "$SAVE_DIR_BASE" "$REPO_ROOT/logs"
