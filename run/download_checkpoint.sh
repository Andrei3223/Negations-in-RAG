#!/bin/bash
set -e

# Load tokens from env file
set -a
source "/home/16207041/Negations-in-RAG/run/.env"
set +a

source /home/16207041/miniconda3/etc/profile.d/conda.sh
conda activate searchr1

MODEL_ID="PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-3b-it-em-grpo"
LOCAL_DIR="checkpoints/SearchR1-qwen2.5-3b-it-em-grpo"

mkdir -p "$LOCAL_DIR"

echo "Downloading $MODEL_ID to $LOCAL_DIR ..."
huggingface-cli download "$MODEL_ID" --local-dir "$LOCAL_DIR"
echo "Done. Checkpoint saved to $LOCAL_DIR"
