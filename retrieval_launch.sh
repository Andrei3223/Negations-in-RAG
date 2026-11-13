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
# Starting port. Not 8000: that is the default of `vllm serve` and most FastAPI
# apps, and it is routinely taken on these shared GPU nodes -- job 347161 loaded
# the full index on cn117 and then died on "address already in use". Derive one
# from the job id in 20000-31999, below the kernel's ephemeral range (32768+),
# so concurrent retrievers differ and outgoing connections never get handed it.
# The server walks forward from here if it is taken; RETRIEVER_PORT still pins
# the starting point.
RETRIEVER_PORT="${RETRIEVER_PORT:-$(( 20000 + ${SLURM_JOB_ID:-$$} % 12000 ))}"

# Overridable to serve another encoder over its own index (e.g. a fine-tuned e5 with
# a re-encoded wiki-18, run/run_negft_eval.sh); defaults are the stock e5 setup.
# RETRIEVER_NAME=splade serves a SPLADE index directory (subsearch/search/splade.py,
# run/run_splade_eval.sh) instead of a FAISS file; --faiss_gpu is then ignored.
index_file="${RETRIEVER_INDEX:-$RETRIEVAL_DATA/e5_Flat.index}"
corpus_file="${RETRIEVER_CORPUS:-$RETRIEVAL_DATA/wiki-18.jsonl}"
retriever_model="${RETRIEVER_MODEL:-intfloat/e5-base-v2}"
retriever_name="${RETRIEVER_NAME:-e5}"
for f in "$index_file" "$corpus_file"; do
    [ -r "$f" ] || { echo "ERROR: cannot read $f"; exit 1; }
done

# Routable address of this node, skipping link-local. The trainer usually lands
# on a different node, so 127.0.0.1 is not enough.
RETRIEVER_IP=$(hostname -I | awk '{for(i=1;i<=NF;i++) if($i !~ /^169\.254/) {print $i; exit}}')

# The endpoint is published by retrieval_server.py itself, only AFTER it has
# reserved a port -- so the file always names the port actually held, not one we
# hoped would still be free once the 60 GB index finished loading.

# --faiss_gpu shards the index across every GPU visible to this job as fp16.
# Drop the flag to run on CPU: correct, but a brute-force scan of 21M vectors
# per query makes rollouts impractically slow.
exec python subsearch/search/retrieval_server.py \
    --index_path "$index_file" \
    --corpus_path "$corpus_file" \
    --topk "${RETRIEVER_TOPK:-3}" \
    --retriever_name "$retriever_name" \
    --retriever_model "$retriever_model" \
    --port "$RETRIEVER_PORT" \
    --port_tries 100 \
    --host_ip "$RETRIEVER_IP" \
    --endpoint_file "$RETRIEVER_IP_PATH" \
    --faiss_gpu
