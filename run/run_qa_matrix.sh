#!/bin/bash
# NQ / HotpotQA regression check: base SearchR1 and the golden-trained checkpoints
# (eval/model=searchr1|exp1|exp2|exp3) on fixed 1,000-question subsets, live wiki-18
# top-3 retrieval, REPS seeded repeats per job, every row logged.
#
#   DRYRUN=1 bash run/run_qa_matrix.sh   # print what would be submitted
#   bash run/run_qa_matrix.sh            # submit
#   RETRIEVER=negft MODELS=exp1 bash run/run_qa_matrix.sh
#                                        # fine-tuned e5 (e5-negft-lam0.3-neg2) over its own
#                                        # re-encoded wiki-18 index (built by run/run_negft_eval.sh)
#
# Submits run/retrieval.job (one shared wiki-18 server), one eval job per model x
# dataset queued behind it (run/eval.py waits for the index to load), and a 1-minute
# cpu job that cancels the retriever once every eval job has ended. Same pattern as
# run/run_ckpt_matrix.sh. Subsets: scripts/data_process/make_qa_eval_subsets.py.
#
# Env: MODELS="searchr1 exp1 exp2 exp3"  REPS=3  DRYRUN=0  RETRIEVER=stock|negft
set -euo pipefail
cd /home/16207041/Negations-in-RAG

MODELS=(${MODELS:-searchr1 exp1 exp2 exp3})
REPS=${REPS:-3}
DRYRUN=${DRYRUN:-0}
RETRIEVER=${RETRIEVER:-stock}
SUBMIT=run/eval_negations_with_answers.sh

case "$RETRIEVER" in
  stock)  # stock e5 over the standard wiki-18 index, publishing to retriever_url.txt
    RETR_NAME=subsearch-retriever; RETR_ARGS=(); RETR_CFG=running; COND=wiki; SUFFIX="" ;;
  negft)  # same paths as run/run_negft_eval.sh; publishes to retriever_url_negft.txt
    CKPT=/fnwi_fs/ivi/irlab/personal/16207041/retriever_checkpoints/e5-negft-lam0.3-neg2/20261005-003334/best
    WIKI_INDEX=/fnwi_fs/ivi/irlab/personal/16207041/indexes/wiki18_e5-negft-lam0.3-neg2/e5_Flat.index
    WIKI_CORPUS=/fnwi_fs/ivi/irlab/projects/chatr1/collection/wiki-18.jsonl
    URL_FILE=/home/16207041/Negations-in-RAG/retriever_url_negft.txt
    [ -r "$WIKI_INDEX" ] || { echo "missing $WIKI_INDEX (build it with run/run_negft_eval.sh)"; exit 1; }
    RETR_NAME=negft-qa-retriever
    RETR_ARGS=(--export=ALL,RETRIEVER_MODEL="$CKPT",RETRIEVER_INDEX="$WIKI_INDEX",RETRIEVER_CORPUS="$WIKI_CORPUS",RETRIEVER_IP_PATH="$URL_FILE")
    RETR_CFG=negft_wiki; COND=negft_wiki; SUFFIX=_negft ;;
  *) echo "RETRIEVER must be stock or negft, got '$RETRIEVER'"; exit 1 ;;
esac

declare -A DATASET=([nq]=nq_1k [hotpotqa]=hotpotqa_1k)
# Wall clock for REPS=3, including up to 90 min waiting for the index; HotpotQA is
# multi-hop and takes more search turns per question.
declare -A TIME=([nq]=05:00:00 [hotpotqa]=07:00:00)

mkdir -p eval_logs/slurm
MANIFEST=eval_logs/run_manifest_qa${SUFFIX}_$(date +%Y%m%d-%H%M%S).csv
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

record() {  # record <model> <dataset> <condition> <job id> <job name>
  echo "$(date '+%F %T'),$1,$2,$3,$4,$5" >> "$MANIFEST"
}

echo "== retriever"
RETR=$(submit "$RETR_NAME" "${RETR_ARGS[@]}" run/retrieval.job)
record - - retriever "$RETR" "$RETR_NAME"
echo "   retrieval.job -> $RETR"

eval_jobs=()
for model in "${MODELS[@]}"; do
  for ds in nq hotpotqa; do
    name="${ds}-${model}-${COND}"
    jid=$(submit "$name" --export=ALL,REPS="$REPS" --time="${TIME[$ds]}" --dependency="after:$RETR" \
            "$SUBMIT" "eval/dataset=${DATASET[$ds]}" "eval/retriever=$RETR_CFG" "eval/model=$model")
    record "$model" "$ds" "$COND" "$jid" "$name"
    eval_jobs+=("$jid")
    echo "   $name -> $jid"
  done
done

echo "== retriever clean-up"
CLEAN=$(submit stop-retriever --partition=cpu --cpus-per-task=1 --mem=1G --time=00:05:00 \
          --output=eval_logs/slurm/%x-%j.out \
          --dependency="afterany:$(IFS=:; echo "${eval_jobs[*]}")" \
          --wrap "scancel $RETR")
record - - cleanup "$CLEAN" stop-retriever
echo "   cancels $RETR after ${#eval_jobs[@]} eval jobs -> $CLEAN"

echo "Submitted ${#eval_jobs[@]} eval jobs (+ retriever, + clean-up). Manifest: $MANIFEST"
