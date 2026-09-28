#!/bin/bash
# Submit the negations_free_sentence_with_answer eval matrix (2 models x 2
# retrievers) 3x each, staggering sbatch submissions by $LAG (default 30m) so
# the runs don't contend for the shared GPU / retrieval-server resources at once.
#
# Keep this process alive for the whole submission window (~5h30m for 12 jobs):
#   nohup bash run/run_nfs_3x.sh &> eval_logs/orchestrator-nfs-$(date +%F-%H%M).log &
#
# Env overrides:
#   LAG=30m     delay between submissions (any `sleep` duration, e.g. 30s, 20m)
#   DRYRUN=1    print the sbatch commands instead of submitting (for testing)
set -u

LAG=${LAG:-30m}
DRYRUN=${DRYRUN:-0}
DATASET=negations_free_sentence_with_answer
MODELS=(searchr1 subsearch)
RETRIEVERS=(wiki custom)
REPS=3
SUBMIT=run/eval_negations_with_answers.sh

mkdir -p eval_logs/slurm
MANIFEST=eval_logs/run_manifest_nfs_$(date +%Y%m%d-%H%M%S).csv
echo "timestamp,rep,model,retriever,job_id,job_name" > "$MANIFEST"

# rep as the outer loop -> the 3 repetitions of each experiment are spread out
# in time rather than submitted back-to-back.
jobs=()
for rep in $(seq 1 "$REPS"); do
  for model in "${MODELS[@]}"; do
    for retr in "${RETRIEVERS[@]}"; do
      jobs+=("$rep|$model|$retr")
    done
  done
done

n=${#jobs[@]}
echo "Submitting $n jobs, $LAG apart (dataset=$DATASET, dryrun=$DRYRUN)"
echo "Manifest: $MANIFEST"

for i in "${!jobs[@]}"; do
  IFS='|' read -r rep model retr <<< "${jobs[$i]}"
  name="nfs-${model}-${retr}-r${rep}"
  echo "[$(date '+%F %T')] submit $((i+1))/$n: $name"

  if [ "$DRYRUN" = "1" ]; then
    echo "  DRYRUN sbatch --job-name=$name $SUBMIT eval/dataset=$DATASET eval/model=$model eval/retriever=$retr"
    jid="DRY$((i+1))"
  else
    out=$(sbatch --job-name="$name" "$SUBMIT" \
          "eval/dataset=${DATASET}" "eval/model=${model}" "eval/retriever=${retr}")
    echo "  $out"
    jid=$(echo "$out" | grep -oE '[0-9]+' | head -1)
  fi

  echo "$(date '+%F %T'),$rep,$model,$retr,${jid:-NA},$name" >> "$MANIFEST"

  # lag between submissions, but not after the last one
  if [ $((i+1)) -lt "$n" ]; then
    echo "  sleeping $LAG..."
    sleep "$LAG"
  fi
done

echo "[$(date '+%F %T')] all submitted -> $MANIFEST"
