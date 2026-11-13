#!/bin/bash
# Evaluate the golden-trained checkpoints (eval/model=exp1|exp2|exp3) under five
# retrieval conditions, REPS seeded repeats per job, every row logged
# (eval.yaml sets reward_model.log_sample_rate: 1).
#
#   DRYRUN=1 bash run/run_ckpt_matrix.sh   # print what would be submitted
#   bash run/run_ckpt_matrix.sh            # submit
#
# Submits, in order:
#   1. run/retrieval.job: one shared wiki-18 server for the wiki condition.
#   2. For every model:
#        wiki       eval/retriever=running    nfwa nfs nevir   (queued behind the retriever;
#                                                                eval.py waits for it to load)
#        custom_2   top-2 over data/custom_index   nfs nevir   (the custom corpus holds only
#        custom_x2  top-1 shown twice              nfs nevir    these two sets' documents)
#        golden     gold document x2               nfwa nfs nevir
#        golden_hn  gold + hard negative           nfwa nfs nevir
#   3. A 1-minute cpu job that cancels the retriever once every wiki job has ended,
#      so its two GPUs are not held for retrieval.job's full 15 days.
#
# Every condition uses the *_docs dataset files: the same rows as the plain files
# (negations-free: test_clean2, complete pairs), plus the columns golden modes need.
# Job names are <dataset>-<model>-<condition>, which scripts/eval_3x.py and
# scripts/build_eval_index.py parse. A manifest goes to eval_logs/.
#
# Env: MODELS="exp1 exp2 exp3"  REPS=3  DRYRUN=0
#      DATASETS="nfwa nfs nevir"  CONDS="wiki golden golden_hn custom_2 custom_x2"
#      (subsets of the matrix; the retriever is only started when CONDS has wiki), e.g.
#   MODELS=searchr1 DATASETS="nfs nevir" CONDS=wiki bash run/run_ckpt_matrix.sh
set -euo pipefail
cd /home/16207041/Negations-in-RAG

MODELS=(${MODELS:-exp1 exp2 exp3})
REPS=${REPS:-3}
DRYRUN=${DRYRUN:-0}
DATASETS=(${DATASETS:-nfwa nfs nevir})
CONDS=" ${CONDS:-wiki golden golden_hn custom_2 custom_x2} "
SUBMIT=run/eval_negations_with_answers.sh

declare -A DATASET=(
  [nfwa]=negations_free_with_answer_docs
  [nfs]=negations_free_sentence_with_answer_docs
  [nevir]=nevir_with_answers_docs
)
# Wall clock for REPS=3; wiki gets extra for the index load and a shared server.
declare -A TIME=([nfwa]=05:00:00 [nfs]=03:00:00 [nevir]=08:00:00)
declare -A TIME_WIKI=([nfwa]=07:00:00 [nfs]=05:00:00 [nevir]=10:00:00)

mkdir -p eval_logs/slurm
MANIFEST=eval_logs/run_manifest_ckpt_$(date +%Y%m%d-%H%M%S).csv
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

RETR=""
if [[ "$CONDS" == *" wiki "* ]]; then
  echo "== retriever"
  RETR=$(submit subsearch-retriever run/retrieval.job)
  record - - retriever "$RETR" subsearch-retriever
  echo "   retrieval.job -> $RETR"
fi

wiki_jobs=()
count=0
for model in "${MODELS[@]}"; do
  echo "== $model"
  for ds in "${DATASETS[@]}"; do
    conds=(wiki golden golden_hn)
    [ "$ds" != nfwa ] && conds+=(custom_2 custom_x2)
    for cond in "${conds[@]}"; do
      [[ "$CONDS" == *" $cond "* ]] || continue
      name="${ds}-${model}-${cond}"
      retriever=$cond
      deps=()
      time=${TIME[$ds]}
      if [ "$cond" = wiki ]; then
        retriever=running
        deps=(--dependency="after:$RETR")
        time=${TIME_WIKI[$ds]}
      fi
      jid=$(submit "$name" --export=ALL,REPS="$REPS" --time="$time" "${deps[@]}" \
              "$SUBMIT" "eval/dataset=${DATASET[$ds]}" "eval/retriever=$retriever" "eval/model=$model")
      record "$model" "$ds" "$cond" "$jid" "$name"
      [ "$cond" = wiki ] && wiki_jobs+=("$jid")
      count=$((count + 1))
      echo "   $name -> $jid"
    done
  done
done

if [ -n "$RETR" ]; then
  echo "== retriever clean-up"
  if [ "${#wiki_jobs[@]}" -eq 0 ]; then
    submit stop-retriever --partition=cpu --cpus-per-task=1 --mem=1G --time=00:05:00 \
      --output=eval_logs/slurm/%x-%j.out --wrap "scancel $RETR" > /dev/null
    echo "   no wiki jobs matched; retriever $RETR cancelled"
  else
    CLEAN=$(submit stop-retriever --partition=cpu --cpus-per-task=1 --mem=1G --time=00:05:00 \
              --output=eval_logs/slurm/%x-%j.out \
              --dependency="afterany:$(IFS=:; echo "${wiki_jobs[*]}")" \
              --wrap "scancel $RETR")
    record - - cleanup "$CLEAN" stop-retriever
    echo "   cancels $RETR after ${#wiki_jobs[@]} wiki jobs -> $CLEAN"
  fi
fi

echo "Submitted $count eval jobs (+ retriever, + clean-up). Manifest: $MANIFEST"
