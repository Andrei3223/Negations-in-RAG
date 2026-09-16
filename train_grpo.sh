#!/bin/bash
# GRPO launcher. Invoked by run/train.job (which sources run/env.sh, activates
# conda, waits for the retriever, and sets the variables read below).
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

# 1. Retriever endpoint, published by retrieval_launch.sh. Accepts either a full
#    "http://host:port" (current format) or a bare IP (older files), so an
#    endpoint file left over from a previous cluster still resolves.
IP_FILE="${RETRIEVER_IP_PATH:?set RETRIEVER_IP_PATH}"
if [ ! -f "$IP_FILE" ]; then
    echo "ERROR: Retriever endpoint file not found at $IP_FILE"
    echo "       Start run/retrieval.job first."
    exit 1
fi
RETRIEVER_BASE=$(tr -d '[:space:]' < "$IP_FILE")
case "$RETRIEVER_BASE" in
    http*) ;;
    *) RETRIEVER_BASE="http://${RETRIEVER_BASE}:${RETRIEVER_PORT:-8000}" ;;
esac
echo "Connecting to Retriever at: $RETRIEVER_BASE"

# 2. Caches on shared storage, not the home quota. run/env.sh sets these; the
#    defaults keep the script runnable on its own.
export HF_HOME="${HF_HOME:-/fnwi_fs/ivi/irlab/personal/$USER/hf_cache}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/fnwi_fs/ivi/irlab/personal/$USER/triton_cache}"
export PYTHONUNBUFFERED=1
mkdir -p "$HF_HOME" "$TRITON_CACHE_DIR"

# 3. NCCL. Single node here, so no InfiniBand fabric settings: export
#    NCCL_SOCKET_IFNAME/NCCL_IB_* before calling this script if you move to a
#    cluster that needs them.
export NCCL_P2P_DISABLE=1   # no NVLink on these nodes
export NCCL_TIMEOUT=1800
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

# GPUs come from the SLURM allocation; only pin them when running bare.
if [ -z "${SLURM_JOB_ID:-}" ] && [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
    export CUDA_VISIBLE_DEVICES=0,1,2,3
fi

# vllm + qwen2 with flash_attn has issues; XFORMERS is the working backend
export VLLM_ATTENTION_BACKEND=XFORMERS

# vLLM hands its KV cache back to the caching allocator between generation and
# the log_prob pass, but as blocks too small to serve one big logits tensor
# (job 339222 died with 21.4 GiB free and 21.4 GiB reserved-but-unallocated).
# Expandable segments let the allocator coalesce that space instead.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# Ray's memory monitor kills its own workers once node RAM crosses this fraction
# (default 0.95) -- that is what ended job 339329 mid-checkpoint. Secondary to
# the --mem=400G node filter in run/train.job, which is the actual fix; this only
# matters if the job ever lands somewhere tighter. Read by the raylet at startup,
# so it has to be exported here rather than in main_ppo.py's runtime_env, which
# only reaches worker processes. Not raised further, and the monitor is NOT
# disabled: it is what turns an OOM into a traceback naming the guilty task
# instead of the kernel silently shooting the raylet.
export RAY_memory_usage_threshold="${RAY_memory_usage_threshold:-0.97}"

# 4. Decide which inputs come from the environment and which the config owns.
#
#    A Hydra CLI override always beats the config file, so passing these
#    unconditionally silently discarded any train_files / val_files / model.path a
#    config set for itself. That is what happened to job 342902: it was launched
#    with CONFIG_NAME=subsearch_negation_ft but trained base Qwen2.5-3B on the old
#    189k combined parquet, because $DATA_PATH and $BASE_MODEL overrode all three.
#    Only the batch size and LR from that config took effect, and nothing said so.
#
#    So: if the named config declares a key itself, the config wins and we do not
#    pass it. Otherwise it comes from the environment, as before. Configs that set
#    none of them (subsearch_grpo, subsearch_smoke, subsearch_overfit) behave
#    exactly as they did, so run/train.job and run/smoke.job are unaffected.
CONFIG_NAME="${CONFIG_NAME:-subsearch_grpo}"
# Resolved from this script's own location, not $0 or the cwd, so it is correct
# however the script is invoked. CONFIG_DIR is overridable for testing.
CONFIG_DIR="${CONFIG_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/verl/trainer/config}"
CONFIG_FILE="$CONFIG_DIR/${CONFIG_NAME}.yaml"

# Loud, because a missing file here would silently fall back to the environment
# and reintroduce exactly the bug this block exists to prevent.
if [ ! -f "$CONFIG_FILE" ]; then
    echo "ERROR: config not found: $CONFIG_FILE"
    echo "       Cannot tell whether it owns its own data/model paths; refusing to"
    echo "       guess. Set CONFIG_DIR, or check CONFIG_NAME=$CONFIG_NAME."
    exit 1
fi

# Reads the named file only, not the composed defaults: the question is whether
# THIS config deliberately claims the key, not whether some ancestor has a value.
owns() {
    python3 - "$CONFIG_FILE" "$1" <<'PY'
import sys, yaml
cfg = yaml.safe_load(open(sys.argv[1])) or {}
data = cfg.get('data') or {}
model = ((cfg.get('actor_rollout_ref') or {}).get('model') or {})
sys.exit(0 if {'train_files': 'train_files' in data,
               'val_files':   'val_files'   in data,
               'model_path':  'path'        in model}[sys.argv[2]] else 1)
PY
}

launch_args=()
if owns train_files; then
    echo "config: ${CONFIG_NAME}.yaml declares data.train_files -- keeping it (\$DATA_PATH ignored)"
elif [ -n "${DATA_PATH:-}" ]; then
    launch_args+=( "data.train_files=$DATA_PATH/train.parquet" )
fi
if owns val_files; then
    echo "config: ${CONFIG_NAME}.yaml declares data.val_files -- keeping it (\$DATA_PATH ignored)"
elif [ -n "${DATA_PATH:-}" ]; then
    launch_args+=( "data.val_files=$DATA_PATH/val.parquet" )
fi
if owns model_path; then
    echo "config: ${CONFIG_NAME}.yaml declares actor_rollout_ref.model.path -- keeping it (\$BASE_MODEL ignored)"
elif [ -n "${BASE_MODEL:-}" ]; then
    launch_args+=( "actor_rollout_ref.model.path=$BASE_MODEL" )
fi

# 5. Launch. Hydra resolves trainer.experiment_name from the values actually in
#    force, so the run name cannot drift from the config the way the old
#    shell-built EXPERIMENT_NAME did.
#
#    EXTRA_OVERRIDES stays last so an ad-hoc override still beats everything above.
python3 -m verl.trainer.main_ppo \
    --config-name "$CONFIG_NAME" \
    ${launch_args[@]+"${launch_args[@]}"} \
    retriever.url="${RETRIEVER_BASE}/retrieve" \
    trainer.default_local_dir="$SAVE_DIR_BASE/\${trainer.experiment_name}" \
    ${EXTRA_OVERRIDES:-}
