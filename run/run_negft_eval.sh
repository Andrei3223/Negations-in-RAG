#!/bin/bash
# Evaluate generator checkpoint(s) with the fine-tuned e5 retriever
# (e5-negft-lam0.3-neg2, best checkpoint) on negations-free-sentence and NevIR:
#   negft_custom_2   custom doc1/doc2 corpus, top-2
#   negft_custom_x2  custom corpus, top-1 shown twice
#   negft_wiki       wiki-18, top-3
# REPS seeded repeats per job, every row logged (EM / F1 in-run; BERTScore afterwards
# via scripts/eval_3x.py). Compare with the stock-e5 runs custom_2 / custom_x2 / wiki.
#
# The passage encoder was fine-tuned too, so both corpora are re-encoded first:
#   1. build-index-custom  data/custom_index_negft            (1 GPU, minutes)
#   2. build-index-wiki    $WIKI_INDEX_DIR                     (4 GPUs, 400G, hours)
#   3. negft-retriever     run/retrieval.job on the new wiki index, publishing to
#                          retriever_url_negft.txt            (after 2 succeeds)
#   4. custom evals        after 1 succeeds; each starts its own small server
#   5. wiki evals          once 3 has started (eval.py waits for the index to load)
#   6. stop-retriever      cancels 3 after every wiki eval has ended
#
#   DRYRUN=1 bash run/run_negft_eval.sh   # print what would be submitted
#   bash run/run_negft_eval.sh            # submit
#   SKIP_BUILD=1 ...                      # indexes already built: skip steps 1-2
#
# Env: MODELS="exp1"  REPS=3  DRYRUN=0  SKIP_BUILD=0
set -euo pipefail
REPO_ROOT=${REPO_ROOT:-/home/16207041/Negations-in-RAG}
STORE_ROOT=${STORE_ROOT:-/fnwi_fs/ivi/irlab/personal/16207041}          # checkpoints, models, indexes
RETRIEVAL_DATA=${RETRIEVAL_DATA:-/fnwi_fs/ivi/irlab/projects/chatr1/collection}  # wiki-18 corpus + e5 index
cd "$REPO_ROOT"

MODELS=(${MODELS:-exp1})
REPS=${REPS:-3}
DRYRUN=${DRYRUN:-0}
SKIP_BUILD=${SKIP_BUILD:-0}
SUBMIT=run/eval_negations_with_answers.sh

CKPT=$STORE_ROOT/retriever_checkpoints/e5-negft-lam0.3-neg2/20261005-003334/best
CUSTOM_INDEX_DIR=data/custom_index_negft
WIKI_INDEX_DIR=$STORE_ROOT/indexes/wiki18_e5-negft-lam0.3-neg2
WIKI_CORPUS=$RETRIEVAL_DATA/wiki-18.jsonl
URL_FILE=$REPO_ROOT/retriever_url_negft.txt

declare -A DATASET=([nfs]=negations_free_sentence_with_answer_docs [nevir]=nevir_with_answers_docs)
declare -A TIME=([nfs]=03:00:00 [nevir]=08:00:00)
declare -A TIME_WIKI=([nfs]=05:00:00 [nevir]=10:00:00)

mkdir -p eval_logs/slurm logs
MANIFEST=eval_logs/run_manifest_negft_$(date +%Y%m%d-%H%M%S).csv
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
  B1=$(submit build-index-custom --gres=gpu:1 --mem=32G --time=01:00:00 \
         --export=ALL,INDEX_MODEL="$CKPT",INDEX_CORPUS=data/custom_index/corpus.jsonl,INDEX_DIR="$CUSTOM_INDEX_DIR" \
         run/build_index.job)
  B2=$(submit build-index-wiki --gres=gpu:4 --cpus-per-task=32 --mem=400G --time=24:00:00 \
         --export=ALL,INDEX_MODEL="$CKPT",INDEX_CORPUS="$WIKI_CORPUS",INDEX_DIR="$WIKI_INDEX_DIR" \
         run/build_index.job)
  record - custom build "$B1" build-index-custom; record - wiki build "$B2" build-index-wiki
  echo "   custom -> $B1   wiki -> $B2"
  custom_dep=(--dependency="afterok:$B1"); wiki_dep=(--dependency="afterok:$B2")
else
  for f in "$CUSTOM_INDEX_DIR/e5_Flat.index" "$WIKI_INDEX_DIR/e5_Flat.index"; do
    [ -r "$f" ] || { echo "SKIP_BUILD=1 but $f is missing"; exit 1; }
  done
fi

echo "== fine-tuned wiki retriever"
RETR=$(submit negft-retriever "${wiki_dep[@]}" \
         --export=ALL,RETRIEVER_MODEL="$CKPT",RETRIEVER_INDEX="$WIKI_INDEX_DIR/e5_Flat.index",RETRIEVER_CORPUS="$WIKI_CORPUS",RETRIEVER_IP_PATH="$URL_FILE" \
         run/retrieval.job)
record - wiki retriever "$RETR" negft-retriever
echo "   retrieval.job -> $RETR"

wiki_jobs=()
for model in "${MODELS[@]}"; do
  echo "== $model"
  for ds in nfs nevir; do
    for cond in negft_custom_2 negft_custom_x2 negft_wiki; do
      name="${ds}-${model}-${cond}"
      if [ "$cond" = negft_wiki ]; then
        deps=(--dependency="after:$RETR"); time=${TIME_WIKI[$ds]}
      else
        deps=("${custom_dep[@]}"); time=${TIME[$ds]}
      fi
      jid=$(submit "$name" --export=ALL,REPS="$REPS" --time="$time" "${deps[@]}" \
              "$SUBMIT" "eval/dataset=${DATASET[$ds]}" "eval/retriever=$cond" "eval/model=$model")
      record "$model" "$ds" "$cond" "$jid" "$name"
      [ "$cond" = negft_wiki ] && wiki_jobs+=("$jid")
      echo "   $name -> $jid"
    done
  done
done

echo "== retriever clean-up"
CLEAN=$(submit stop-negft-retriever --partition=cpu --cpus-per-task=1 --mem=1G --time=00:05:00 \
          --output=eval_logs/slurm/%x-%j.out \
          --dependency="afterany:$(IFS=:; echo "${wiki_jobs[*]}")" \
          --wrap "scancel $RETR")
record - - cleanup "$CLEAN" stop-negft-retriever
echo "   cancels $RETR after ${#wiki_jobs[@]} wiki evals -> $CLEAN"
echo "Manifest: $MANIFEST"
