#!/bin/bash
#SBATCH --job-name=searchr1-eval
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Usage: sbatch run/eval_negations_with_answers.sh [hydra overrides...]
#
# Config groups (pass as key=value):
#   eval/dataset:   nevir_with_answers (default)
#                   negations_free_with_answer | negation_controlled_with_answers |
#                   negations_free_sentence_with_answer
#   eval/model:     searchr1 (default) | searchr1_it | subsearch
#   eval/retriever: wiki (default) | custom
#
# Examples:
#   sbatch run/eval_negations_with_answers.sh
#   sbatch run/eval_negations_with_answers.sh "eval/dataset=negations_free_with_answer" "eval/retriever=custom"
#   sbatch run/eval_negations_with_answers.sh "eval/model=subsearch" data.val_batch_size=100

set -e

set -a; source "/home/16207041/Negations-in-RAG/run/.env"; set +a
source "${HOME}/.bashrc"
source /home/16207041/miniconda3/etc/profile.d/conda.sh
conda activate searchr1

export CUDA_VISIBLE_DEVICES=0
export VLLM_ATTENTION_BACKEND=XFORMERS
export NCCL_P2P_DISABLE=1

mkdir -p logs

# Readable log name: <job-name>-<job-id> (e.g. nevir-subsearch-custom-r2-329207).
# SLURM_JOB_NAME is set from sbatch --job-name; falls back to "local" off-SLURM.
RUN_TAG="${SLURM_JOB_NAME:-local}-${SLURM_JOB_ID:-local}"
PYTHONUNBUFFERED=1 python3 run/eval.py "$@" \
    2>&1 | tee "logs/eval-${RUN_TAG}.log"

exit "${PIPESTATUS[0]}"
