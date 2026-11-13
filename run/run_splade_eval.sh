#!/bin/bash
# Evaluate generator checkpoint(s) with SPLADE-v3 (naver/splade-v3) as the retriever:
#   splade_custom_2   nfs + nevir, custom doc1/doc2 corpus, top-2
#   splade_custom_x2  nfs + nevir, custom corpus, top-1 shown twice
#   splade_wiki       hotpotqa-1k (and nq-1k on request, EVALS=nq:wiki), wiki-18, top-3
# REPS seeded repeats per job, every row logged (EM / F1 in-run; BERTScore afterwards
# via scripts/eval_3x.py). Compare with the stock-e5 runs (custom_2 / custom_x2 / wiki)
# and the fine-tuned e5 runs (negft_*).
#
# SPLADE needs its own sparse indexes (subsearch/search/splade.py):
#   1. build-splade-custom  data/custom_index_splade_v3        (1 GPU, minutes)
#   2. build-splade-wiki    $WIKI_INDEX_DIR                    (4 GPUs, hours; resumable)
#   3. splade-retriever     run/retrieval.job serving the wiki index, publishing to
#                           retriever_url_splade.txt          (after 2 succeeds)
#   4. custom evals         after 1 succeeds; each starts its own small server
#   5. wiki evals           once 3 has started (eval.py waits for the index to load)
#   6. stop-retriever       cancels 3 after every wiki eval has ended
#
#   DRYRUN=1 bash run/run_splade_eval.sh   # print what would be submitted
#   bash run/run_splade_eval.sh            # submit
#   SKIP_BUILD=1 ...                       # indexes already built: skip steps 1-2
#
# Once, before the first run: naver/splade-v3 is gated (accept its terms on
# huggingface.co with the HF_TOKEN account) and ships only pytorch_model.bin, which
# the retriever env's transformers 5.4 refuses to load under torch 2.4. Convert it
# to safetensors on the login node:
#   conda activate searchr1 && source run/env.sh
#   python scripts/convert_to_safetensors.py naver/splade-v3 $SPLADE_MODEL (path below)
#
# Fine-tuned SPLADE (retriever_finetune, model=splade_v3): same pipeline with its own
# indexes, server and conditions (spladeft_custom_2 / spladeft_custom_x2 / spladeft_wiki):
#   SPLADE_RUN=ft MODELS=exp1 bash run/run_splade_eval.sh
# SPLADE_RUN=ftneg: the same fine-tuning without MS MARCO replay (spladeftneg_*), trained
# on exactly the data of the fine-tuned e5 (negft_*).
#
# Only some evals, e.g. NevIR and NFS over wiki-18 with indexes already built:
#   SPLADE_RUN=ft SKIP_BUILD=1 MODELS=exp1 EVALS="nfs:wiki nevir:wiki" bash run/run_splade_eval.sh
#
# Env: SPLADE_RUN=base|ft|ftneg  MODELS="searchr1 exp1"  REPS=3  DRYRUN=0  SKIP_BUILD=0
#      EVALS="nfs:custom_2 nfs:custom_x2 nevir:custom_2 nevir:custom_x2 hotpotqa:wiki"
set -euo pipefail
REPO_ROOT=${REPO_ROOT:-/home/16207041/Negations-in-RAG}
STORE_ROOT=${STORE_ROOT:-/fnwi_fs/ivi/irlab/personal/16207041}          # checkpoints, models, indexes
RETRIEVAL_DATA=${RETRIEVAL_DATA:-/fnwi_fs/ivi/irlab/projects/chatr1/collection}  # wiki-18 corpus + e5 index
cd "$REPO_ROOT"

MODELS=(${MODELS:-searchr1 exp1})
REPS=${REPS:-3}
DRYRUN=${DRYRUN:-0}
SKIP_BUILD=${SKIP_BUILD:-0}
SUBMIT=run/eval_negations_with_answers.sh

WIKI_CORPUS=$RETRIEVAL_DATA/wiki-18.jsonl
# SPLADE_RUN picks the retriever; TAG is the condition prefix (eval/retriever=${TAG}_*,
# whose configs hold the same model and index paths) and names the jobs.
case "${SPLADE_RUN:-base}" in
  base)  # stock naver/splade-v3 (safetensors copy)
    TAG=splade
    SPLADE_MODEL=$STORE_ROOT/models/splade-v3
    CUSTOM_INDEX_DIR=data/custom_index_splade_v3
    WIKI_INDEX_DIR=$STORE_ROOT/indexes/wiki18_splade-v3
    MAX_TERMS=256 ;;   # ~211 terms per wiki passage: the cap rarely binds
  ft)    # fine-tuned on NevIR + NFS pairs + MS MARCO (lambda_d=0.003, picked by val pair acc)
    TAG=spladeft
    SPLADE_MODEL=$STORE_ROOT/retriever_checkpoints/splade-v3-negft-lam0.3-neg2-marco1782/20261014-140623/best
    CUSTOM_INDEX_DIR=data/custom_index_spladeft
    WIKI_INDEX_DIR=$STORE_ROOT/indexes/wiki18_spladeft-lamd0.003
    MAX_TERMS=0 ;;     # no cap: ~334 terms per wiki passage (p99 561), 256 would truncate
  ftneg) # same as ft but negation pairs only (marco=none), like the fine-tuned e5 (negft_*)
    TAG=spladeftneg
    SPLADE_MODEL=$STORE_ROOT/retriever_checkpoints/splade-v3-negft-lam0.3-neg2-nomarco/20261101-190055/best
    CUSTOM_INDEX_DIR=data/custom_index_spladeftneg
    WIKI_INDEX_DIR=$STORE_ROOT/indexes/wiki18_spladeftneg-lamd0.003
    MAX_TERMS=0 ;;
  *) echo "SPLADE_RUN must be base, ft or ftneg, got '$SPLADE_RUN'"; exit 1 ;;
esac
[ -f "$SPLADE_MODEL/model.safetensors" ] || {
  echo "missing $SPLADE_MODEL/model.safetensors (base: run scripts/convert_to_safetensors.py, see the header)"; exit 1; }
URL_FILE=$REPO_ROOT/retriever_url_${TAG}.txt

# condition -> datasets it runs on; dataset -> eval config and wall clock (REPS=3)
declare -A DATASET=([nfs]=negations_free_sentence_with_answer_docs [nevir]=nevir_with_answers_docs
                    [hotpotqa]=hotpotqa_1k [nq]=nq_1k)
declare -A TIME=([nfs]=03:00:00 [nevir]=08:00:00 [hotpotqa]=07:00:00 [nq]=05:00:00)
# Which evals to submit, as <dataset>:<setting>; the condition is ${TAG}_<setting>.
# Only a :wiki entry starts the wiki retriever.
read -ra EVAL_LIST <<< "${EVALS:-nfs:custom_2 nfs:custom_x2 nevir:custom_2 nevir:custom_x2 hotpotqa:wiki}"
for e in "${EVAL_LIST[@]}"; do
  [[ -n "${DATASET[${e%%:*}]:-}" && "${e#*:}" =~ ^(custom_2|custom_x2|wiki)$ ]] || {
    echo "bad EVALS entry '$e': want <nfs|nevir|hotpotqa|nq>:<custom_2|custom_x2|wiki>"; exit 1; }
done

mkdir -p eval_logs/slurm logs
MANIFEST=eval_logs/run_manifest_${TAG}_$(date +%Y%m%d-%H%M%S).csv
[ "$DRYRUN" = "1" ] && MANIFEST=/dev/null
echo "timestamp,model,dataset,condition,job_id,job_name" > "$MANIFEST"

submit() {  # submit <job-name> <sbatch args...> ; prints the job id
  local name=$1; shift
  if [ "$DRYRUN" = "1" ]; then
    echo "  DRYRUN sbatch --job-name=$name $*" >&2
    echo "<$name>"
  else
    sbatch --parsable --job-name="$name" "$@"
  fi
}
record() { echo "$(date '+%F %T'),$1,$2,$3,$4,$5" >> "$MANIFEST"; }

echo "== indexes"
custom_dep=(); wiki_dep=()
if [ "$SKIP_BUILD" != "1" ]; then
  B1=$(submit build-${TAG}-custom --gres=gpu:nvidia_rtx_a6000:1 --mem=32G --time=01:00:00 \
         --export=ALL,SPLADE_MODEL="$SPLADE_MODEL",INDEX_CORPUS=data/custom_index/corpus.jsonl,INDEX_DIR="$CUSTOM_INDEX_DIR",SPLADE_MAX_TERMS="$MAX_TERMS" \
         run/build_splade_index.job)
  # Typed GPUs: an untyped gpu:4 can land on the Titan X/Xp nodes (no bf16, fp32
  # fallback, several times slower). 4 L40s built the base index in 82 min.
  B2=$(submit build-${TAG}-wiki --gres=${WIKI_BUILD_GRES:-gpu:nvidia_l40:4} --cpus-per-task=32 --mem=128G --time=24:00:00 \
         --export=ALL,SPLADE_MODEL="$SPLADE_MODEL",INDEX_CORPUS="$WIKI_CORPUS",INDEX_DIR="$WIKI_INDEX_DIR",SPLADE_MAX_TERMS="$MAX_TERMS" \
         run/build_splade_index.job)
  record - custom build "$B1" build-${TAG}-custom; record - wiki build "$B2" build-${TAG}-wiki
  echo "   custom -> $B1   wiki -> $B2"
  custom_dep=(--dependency="afterok:$B1"); wiki_dep=(--dependency="afterok:$B2")
else
  for f in "$CUSTOM_INDEX_DIR/meta.json" "$WIKI_INDEX_DIR/meta.json"; do
    [ -r "$f" ] || { echo "SKIP_BUILD=1 but $f is missing"; exit 1; }
  done
fi

RETR=""
if [[ " ${EVAL_LIST[*]} " == *":wiki "* ]]; then
  echo "== SPLADE wiki retriever"
  RETR=$(submit ${TAG}-retriever "${wiki_dep[@]}" \
           --export=ALL,RETRIEVER_NAME=splade,RETRIEVER_MODEL="$SPLADE_MODEL",RETRIEVER_INDEX="$WIKI_INDEX_DIR",RETRIEVER_CORPUS="$WIKI_CORPUS",RETRIEVER_IP_PATH="$URL_FILE" \
           run/retrieval.job)
  record - wiki retriever "$RETR" ${TAG}-retriever
  echo "   retrieval.job -> $RETR"
fi

wiki_jobs=()
for model in "${MODELS[@]}"; do
  echo "== $model"
  for job in "${EVAL_LIST[@]}"; do
    ds=${job%%:*}; cond=${TAG}_${job#*:}
    name="${ds}-${model}-${cond}"
    if [ "$cond" = ${TAG}_wiki ]; then deps=(--dependency="after:$RETR"); else deps=("${custom_dep[@]}"); fi
    jid=$(submit "$name" --export=ALL,REPS="$REPS" --time="${TIME[$ds]}" "${deps[@]}" \
            "$SUBMIT" "eval/dataset=${DATASET[$ds]}" "eval/retriever=$cond" "eval/model=$model")
    record "$model" "$ds" "$cond" "$jid" "$name"
    [ "$cond" = ${TAG}_wiki ] && wiki_jobs+=("$jid")
    echo "   $name -> $jid"
  done
done

if [ -n "$RETR" ]; then
  echo "== retriever clean-up"
  CLEAN=$(submit stop-${TAG}-retriever --partition=cpu --cpus-per-task=1 --mem=1G --time=00:05:00 \
            --output=eval_logs/slurm/%x-%j.out \
            --dependency="afterany:$(IFS=:; echo "${wiki_jobs[*]}")" \
            --wrap "scancel $RETR")
  record - - cleanup "$CLEAN" stop-${TAG}-retriever
  echo "   cancels $RETR after ${#wiki_jobs[@]} wiki evals -> $CLEAN"
fi
echo "Manifest: $MANIFEST"
