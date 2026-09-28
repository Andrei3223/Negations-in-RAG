#!/bin/bash
#SBATCH --job-name=searchr1-eval
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=03:00:00
# SLURM needs a fixed, pre-existing path here, so its own stdout/stderr go to a
# shared folder; everything else for the run lands in eval_logs/<job-name>-<jobid>/.
#SBATCH --output=eval_logs/slurm/%x-%j.out
#SBATCH --error=eval_logs/slurm/%x-%j.err

# Usage: sbatch run/eval_negations_with_answers.sh [hydra overrides...]
#
# Config groups (pass as key=value):
#   eval/dataset:   nevir_with_answers (default)
#                   negations_free_with_answer | negation_controlled_with_answers |
#                   negations_free_sentence_with_answer |
#                   negations_free_with_answer_docs (test_clean2 + golden docs)
#   eval/model:     searchr1 (default) | searchr1_it | subsearch
#   eval/retriever: wiki (default) | custom | wiki_5
#                   custom_2  (custom corpus, top-2)
#                   custom_x2 (custom corpus, top-1 shown twice; avoids the lone-passage failure)
#                   running   (reuse run/retrieval.job's server)
#                   golden    (no retrieval: <search> returns the gold doc; needs a *_docs dataset)
#                   golden_hn (no retrieval: gold doc + hard negative; needs a *_docs dataset)
#                   golden_rn (no retrieval: gold doc + unrelated doc from another topic)
#
# Examples:
#   sbatch run/eval_negations_with_answers.sh
#   sbatch run/eval_negations_with_answers.sh "eval/dataset=negations_free_with_answer" "eval/retriever=custom"
#   sbatch run/eval_negations_with_answers.sh "eval/model=subsearch" data.val_batch_size=100
#   sbatch run/eval_negations_with_answers.sh "eval/dataset=negations_free_with_answer_docs" "eval/retriever=golden_hn"
#
# Repeats in ONE job (sampling is temperature 1): set REPS. Rep k runs with
# actor_rollout_ref.rollout.seed=k -- the seed must differ, since vLLM's default
# seed 0 makes every run identical. Reps run back to back, so scale --time.
#   sbatch --export=ALL,REPS=3 --time=06:00:00 --job-name=nfs-subsearch-golden \
#     run/eval_negations_with_answers.sh "eval/dataset=negations_free_with_answer_docs" "eval/retriever=golden"
# Don't also pass actor_rollout_ref.rollout.seed when REPS > 1.
#
# Logs (one folder per run; --job-name sets <job-name>, so name your runs):
#   eval_logs/<job-name>-<jobid>/eval.log          full output, incl. final metrics
#   eval_logs/<job-name>-r<k>-<jobid>/eval.log     the same, per repeat when REPS > 1
#   eval_logs/<job-name>-<jobid>/.hydra/           resolved config + overrides
#   eval_logs/slurm/<job-name>-<jobid>.{out,err}   SLURM's own streams
# Submit from the repo root: eval_logs/slurm must exist before sbatch opens it.

set -e

set -a; source "/home/16207041/Negations-in-RAG/run/.env"; set +a
source "${HOME}/.bashrc"
source /home/16207041/miniconda3/etc/profile.d/conda.sh
conda activate searchr1

export CUDA_VISIBLE_DEVICES=0
export VLLM_ATTENTION_BACKEND=XFORMERS
export NCCL_P2P_DISABLE=1

REPS="${REPS:-1}"
JOB_NAME="${SLURM_JOB_NAME:-local}"
# Off-SLURM there is no job id, so a timestamp keeps local runs apart.
JOB_ID="${SLURM_JOB_ID:-$(date +%Y%m%d-%H%M%S)}"
mkdir -p eval_logs/slurm

# One run folder per repeat: <job-name>-<job-id>, or <job-name>-r<k>-<job-id>
# when REPS > 1 (the name scripts/eval_3x.py parses). A failed repeat doesn't
# stop the others; the job exits non-zero if any failed.
failed=()
for k in $(seq 1 "$REPS"); do
    if [ "$REPS" -gt 1 ]; then
        RUN_TAG="${JOB_NAME}-r${k}-${JOB_ID}"
        rep_args=("actor_rollout_ref.rollout.seed=${k}")
    else
        RUN_TAG="${JOB_NAME}-${JOB_ID}"
        rep_args=()
    fi
    # eval.yaml points hydra.run.dir at eval_logs/${EVAL_RUN_TAG}.
    export EVAL_RUN_TAG="$RUN_TAG"
    RUN_DIR="eval_logs/${RUN_TAG}"
    mkdir -p "$RUN_DIR"
    echo "=== repeat ${k}/${REPS} -> ${RUN_DIR} ${rep_args[*]}"

    set +e
    PYTHONUNBUFFERED=1 python3 run/eval.py "$@" "${rep_args[@]}" \
        2>&1 | tee "${RUN_DIR}/eval.log"
    status="${PIPESTATUS[0]}"
    set -e
    [ "$status" -eq 0 ] || failed+=("r${k}:exit${status}")
done

if [ "${#failed[@]}" -gt 0 ]; then
    echo "FAILED repeats: ${failed[*]}"
    exit 1
fi
