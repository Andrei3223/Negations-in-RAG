#!/bin/bash
#SBATCH --job-name=searchr1-eval-negation
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=03:00:00
#SBATCH --output=eval_logs/slurm/%x-%j.out
#SBATCH --error=eval_logs/slurm/%x-%j.err

set -e

# Load tokens from env file
set -a
source "/home/16207041/Negations-in-RAG/run/.env"
set +a

# Set-up the environment.
source "${HOME}/.bashrc"
source /home/16207041/miniconda3/etc/profile.d/conda.sh

NEGATION_DATA='data/negation_rag/gpt4o-negation-controlled'
RETRIEVAL_DATA="/fnwi_fs/ivi/irlab/projects/chatr1/collection"

# Caches live on shared storage, not the 100G home quota. The wiki-18 Arrow build
# (~9G) is already materialised under $HF_HOME/datasets from $RETRIEVAL_DATA/wiki-18.jsonl.
export HF_HOME="/fnwi_fs/ivi/irlab/personal/16207041/hf_cache"
export TRITON_CACHE_DIR="/fnwi_fs/ivi/irlab/personal/16207041/triton_cache"

export BASE_MODEL='checkpoints/SearchR1-qwen2.5-3b-it-em-grpo'
export EXPERIMENT_NAME=negation-controlled-eval-qwen2.5-3b-it-em-grpo

export CUDA_VISIBLE_DEVICES=0
export VLLM_ATTENTION_BACKEND=XFORMERS
export NCCL_P2P_DISABLE=1

RETRIEVAL_PORT=8001

# One folder per run, as in run/eval_negations_with_answers.sh.
RUN_DIR="eval_logs/${SLURM_JOB_NAME:-local}-${SLURM_JOB_ID:-$(date +%Y%m%d-%H%M%S)}"
mkdir -p "$RUN_DIR" eval_logs/slurm

# --- Launch retrieval server in background (retriever conda env) ---
echo "Starting retrieval server..."
/home/16207041/miniconda3/envs/retriever/bin/python -u search_r1/search/retrieval_server.py \
        --index_path  "$RETRIEVAL_DATA/e5_Flat.index" \
        --corpus_path "$RETRIEVAL_DATA/wiki-18.jsonl" \
        --topk 3 \
        --retriever_name e5 \
        --retriever_model intfloat/e5-base-v2 \
        --port $RETRIEVAL_PORT \
    > $RUN_DIR/retrieval.log 2>&1 &
RETRIEVAL_PID=$!

# Ensure retrieval server is killed when this script exits (for any reason)
trap "echo 'Killing retrieval server (PID $RETRIEVAL_PID)'; kill $RETRIEVAL_PID 2>/dev/null" EXIT

echo "Waiting for retrieval server on port $RETRIEVAL_PORT (up to 20 min)..."
for i in $(seq 1 240); do
    if curl -sf http://127.0.0.1:$RETRIEVAL_PORT/retrieve \
            -d '{"queries":["test"],"topk":1}' \
            -H 'Content-Type: application/json' > /dev/null 2>&1; then
        echo "Retrieval server is ready (waited ~$((i*5))s)."
        break
    fi
    if ! kill -0 $RETRIEVAL_PID 2>/dev/null; then
        echo "ERROR: Retrieval server process died. Check $RUN_DIR/retrieval.log"
        exit 1
    fi
    if [ $i -eq 240 ]; then
        echo "ERROR: Retrieval server did not start in time. Check $RUN_DIR/retrieval.log"
        exit 1
    fi
    sleep 5
done

# --- Evaluation on gpt4o-negation-controlled (val_only, no gradient updates) ---
# max_prompt_length = max_start_length + max_response_length * (max_turns - 1) + max_obs_length * max_turns
# = 4096 + 1500 * 1 + 2048 * 2 = 9692 -> 10240

conda activate searchr1

PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    data.train_files=$NEGATION_DATA/test.parquet \
    data.val_files=$NEGATION_DATA/test.parquet \
    data.train_data_num=null \
    data.val_data_num=null \
    data.train_batch_size=8 \
    data.val_batch_size=32 \
    data.max_prompt_length=10240 \
    data.max_response_length=1500 \
    data.max_start_length=4096 \
    data.max_obs_length=2048 \
    data.shuffle_train_dataloader=False \
    algorithm.adv_estimator=grpo \
    actor_rollout_ref.model.path=$BASE_MODEL \
    actor_rollout_ref.model.enable_gradient_checkpointing=false \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.0 \
    actor_rollout_ref.actor.use_kl_loss=false \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.ppo_micro_batch_size=2 \
    actor_rollout_ref.actor.fsdp_config.param_offload=false \
    actor_rollout_ref.actor.fsdp_config.grad_offload=false \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=false \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.ref.log_prob_micro_batch_size=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=false \
    algorithm.no_think_rl=false \
    actor_rollout_ref.rollout.n_agent=1 \
    actor_rollout_ref.rollout.temperature=1 \
    actor_rollout_ref.actor.state_masking=true \
    trainer.logger=[] \
    +trainer.val_only=true \
    +trainer.val_before_train=true \
    trainer.default_hdfs_dir=null \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=1 \
    trainer.project_name='Search-R1' \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.total_epochs=1 \
    trainer.total_training_steps=0 \
    trainer.default_local_dir=verl_checkpoints/$EXPERIMENT_NAME \
    max_turns=2 \
    retriever.url="http://127.0.0.1:$RETRIEVAL_PORT/retrieve" \
    retriever.topk=3 \
    2>&1 | tee "$RUN_DIR/eval.log"
