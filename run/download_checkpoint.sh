#!/bin/bash
set -e

# Load tokens from env file
REPO_ROOT="${REPO_ROOT:-/home/16207041/Negations-in-RAG}"
set -a
[ -f "$REPO_ROOT/run/.env" ] && source "$REPO_ROOT/run/.env"
set +a

source "${CONDA_DIR:-/home/16207041/miniconda3}/etc/profile.d/conda.sh"
conda activate searchr1

MODEL_ID="PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-3b-it-em-grpo"
LOCAL_DIR="checkpoints/SearchR1-qwen2.5-3b-it-em-grpo"

mkdir -p "$LOCAL_DIR"

echo "Downloading $MODEL_ID to $LOCAL_DIR ..."
huggingface-cli download "$MODEL_ID" --local-dir "$LOCAL_DIR"
echo "Done. Checkpoint saved to $LOCAL_DIR"
