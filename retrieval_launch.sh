#!/bin/bash
# Launcher for the e5/FAISS retrieval server. Invoked by run/retrieval.job,
# which activates the `retriever` conda env and sources run/env.sh for the
# variables read below.
#
# Publishes its own endpoint to $RETRIEVER_IP_PATH so run/train.job can find it
# without knowing which node SLURM picked.
set -eo pipefail

: "${RETRIEVAL_DATA:?set RETRIEVAL_DATA -- directory holding e5_Flat.index and wiki-18.jsonl}"
: "${RETRIEVER_IP_PATH:?set RETRIEVER_IP_PATH -- file the trainer polls for the endpoint}"
RETRIEVER_PORT="${RETRIEVER_PORT:-8000}"

index_file="$RETRIEVAL_DATA/e5_Flat.index"
corpus_file="$RETRIEVAL_DATA/wiki-18.jsonl"
for f in "$index_file" "$corpus_file"; do
    [ -r "$f" ] || { echo "ERROR: cannot read $f"; exit 1; }
done

# Routable address of this node, skipping link-local. The trainer usually lands
# on a different node, so 127.0.0.1 is not enough.
RETRIEVER_IP=$(hostname -I | awk '{for(i=1;i<=NF;i++) if($i !~ /^169\.254/) {print $i; exit}}')
printf 'http://%s:%s\n' "$RETRIEVER_IP" "$RETRIEVER_PORT" > "$RETRIEVER_IP_PATH"
echo "Retriever endpoint published to $RETRIEVER_IP_PATH -> http://$RETRIEVER_IP:$RETRIEVER_PORT"

# --faiss_gpu shards the index across every GPU visible to this job as fp16.
# Drop the flag to run on CPU: correct, but a brute-force scan of 21M vectors
# per query makes rollouts impractically slow.
exec python subsearch/search/retrieval_server.py \
    --index_path "$index_file" \
    --corpus_path "$corpus_file" \
    --topk "${RETRIEVER_TOPK:-3}" \
    --retriever_name e5 \
    --retriever_model intfloat/e5-base-v2 \
    --port "$RETRIEVER_PORT" \
    --faiss_gpu
