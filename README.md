# Negations in RAG

How well do search-augmented LLM agents handle negation? This repository evaluates
and fine-tunes reason-and-search agents (Search-R1 and SubSearch, Qwen2.5-3B) on
paired questions that differ only by a negation, such as *"Which ethologists focus on
individual behaviour?"* vs. *"Which ethologists ignore individual behaviour?"*. Each
question of a pair has its own supporting document and its own answer.

It contains:

- **An evaluation pipeline** for paired negation QA (NevIR, Negations-Free) and general
  open-domain QA (NQ, HotpotQA). It runs under several retrieval conditions: live
  Wikipedia retrieval, retrieval over a corpus of the pairs' own documents, and
  "golden" runs that hand the model the gold document directly. It reports EM, token F1,
  BERTScore, pair accuracy and empty-answer rates over seeded repeats.
- **Generator fine-tuning** with GRPO on golden documents (gold + minimally different
  hard negative + unrelated document), so the policy has to read the negation to answer.
- **Retriever fine-tuning** for e5 and SPLADE on negation minimal pairs, with a
  Plackett-Luce ranking loss that puts a query's own document above its negated twin.

The code is built on [SubSearch](https://arxiv.org/abs/2604.07415),
[Search-R1](https://github.com/PeterGriffinJin/Search-R1) and
[veRL](https://github.com/volcengine/verl).

## Repository layout

| Path | Contents |
|---|---|
| `run/eval.py` | Hydra entry point for evaluation (validation-only veRL run) |
| `run/*.job`, `run/*.sh` | SLURM jobs and launchers; `run/env.sh` holds the shared configuration |
| `verl/trainer/config/eval/` | Evaluation config groups: `dataset/`, `model/`, `retriever/` |
| `verl/trainer/config/golden_*.yaml` | Generator fine-tuning configs |
| `subsearch/llm_agent/generation.py` | Multi-turn reason-and-search loop, including the golden-document modes |
| `subsearch/search/` | Dense (e5 + FAISS) and sparse (SPLADE) retrieval servers and index builders |
| `retriever_finetune/` | Retriever fine-tuning (Hydra configs in `conf/`) |
| `scripts/` | Scoring and analysis; `scripts/data_process/` builds the evaluation and training files |
| `docs/datasets.md` | The negation datasets, their splits and how each was cleaned |

## Setup

### Environments

Two conda environments: one for the generator (training and evaluation) and one for
the retrieval server.

```bash
# Generator: torch 2.4, vLLM 0.6.3, veRL (editable)
conda create -n searchr1 python=3.10
conda activate searchr1
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121
pip install vllm==0.6.3
pip install flash-attn --no-build-isolation
pip install -e .
pip install wandb bert-score

# Retrieval server: FAISS-GPU 1.8, FastAPI
conda create -n retriever python=3.10
conda activate retriever
conda install pytorch==2.4.0 pytorch-cuda=12.1 -c pytorch -c nvidia
conda install -c pytorch -c nvidia faiss-gpu=1.8.0
pip install transformers datasets pyserini uvicorn fastapi
```

### Paths and secrets

Every machine-specific path is an environment variable. The defaults are the cluster
layout the experiments ran on, so set the ones that differ on your machine, either in
your shell or in `run/.env`:

| Variable | Used for |
|---|---|
| `REPO_ROOT` | the repository checkout |
| `STORE_ROOT` | large storage: Hugging Face / Triton caches, generator and retriever checkpoints, models, re-encoded indexes |
| `RETRIEVAL_DATA` | directory with the wiki-18 corpus (`wiki-18.jsonl`) and its e5 index (`e5_Flat.index`) |
| `CONDA_DIR` | conda installation that holds the `searchr1` and `retriever` environments |
| `RAW_DATA_DIR` | raw negation datasets read by `scripts/data_process/` |
| `MSMARCO_DIR` | MS MARCO v1 passage copy, for retriever fine-tuning with replay |

`run/env.sh` defines them in one place and derives the rest (`HF_HOME`,
`SAVE_DIR_BASE`, ...). Put credentials in `run/.env` (ignored by git):

```bash
HF_TOKEN=...
WANDB_API_KEY=...
```

SLURM jobs write their logs relative to the submission directory, so submit from the
repository root and create the log folders once:

```bash
mkdir -p logs eval_logs/slurm
```

### Models and retrieval corpus

```bash
# Search-R1 (Qwen2.5-3B, EM reward, GRPO): the baseline and the starting point for fine-tuning
huggingface-cli download PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-3b-em-grpo \
    --local-dir checkpoints/SearchR1-nq_hotpotqa_train-qwen2.5-3b-em-grpo

# wiki-18 corpus and e5 index, as released with Search-R1
python scripts/download.py --save_path $RETRIEVAL_DATA
cat $RETRIEVAL_DATA/part_* > $RETRIEVAL_DATA/e5_Flat.index
gzip -d $RETRIEVAL_DATA/wiki-18.jsonl.gz
```

## Evaluation

Evaluation is a validation-only veRL run, configured by three Hydra groups. Start a
retrieval server for live wiki-18 retrieval first. It loads the index (about 15–30
minutes) and publishes its address to `retriever_url.txt`:

```bash
sbatch run/retrieval.job
```

Then submit evaluations. `REPS=3` runs three repeats in one job with sampling seeds
1, 2 and 3:

```bash
sbatch --export=ALL,REPS=3 --job-name=nevir-exp1-golden \
    run/eval_negations_with_answers.sh \
    eval/dataset=nevir_with_answers_docs eval/retriever=golden eval/model=exp1
```

| Group | Options |
|---|---|
| `eval/dataset` | `nevir_with_answers`, `negations_free_with_answer`, `negations_free_sentence_with_answer`, `negation_controlled_with_answers`; `*_docs` variants carry the documents the golden modes need; `nq_1k`, `hotpotqa_1k` (fixed 1,000-question subsets) |
| `eval/model` | `searchr1`, `searchr1_it`, `subsearch`, `exp1`, `exp2`, `exp3` (the fine-tuned generators below) |
| `eval/retriever` | `running` (use the server from `run/retrieval.job`, wiki-18 top-3), `wiki` / `wiki_5` (start a private wiki-18 server, top-3 / top-5) |
| | `custom_2`, `custom_x2` (corpus of the negation pairs' documents: top-2, or top-1 shown twice) |
| | `golden` (gold document twice), `golden_hn` (gold + hard negative), `golden_rn` (gold + random document): no retrieval |
| | `negft_*`, `splade_*`, `spladeft_*`, `spladeftneg_*`: fine-tuned e5, SPLADE-v3 and fine-tuned SPLADE |

Each repeat writes its full log and resolved config to
`eval_logs/<job-name>-r<k>-<jobid>/`. Decoding samples at temperature 1; pass
`trainer.val_do_sample=false` for greedy decoding. Search-R1 was trained with three
passages per search and writes fake ones when it gets a single passage, so every
condition above shows it at least two.

Launchers submit whole experiment matrices, each with its own retriever and a clean-up
job. All of them accept `DRYRUN=1` to print the `sbatch` commands without submitting:

| Launcher | Runs |
|---|---|
| `run/run_ckpt_matrix.sh` | generator checkpoints × {wiki, custom_2, custom_x2, golden, golden_hn} (`MODELS`, `DATASETS`, `CONDS` select a subset) |
| `run/run_qa_matrix.sh` | NQ / HotpotQA regression check; `RETRIEVER=stock\|negft` |
| `run/run_negft_eval.sh` | re-encodes the corpora with a fine-tuned e5 and evaluates on them |
| `run/run_splade_eval.sh` | the same for SPLADE: `SPLADE_RUN=base\|ft\|ftneg` |

### Scoring

```bash
python scripts/eval_3x.py            # EM, F1, BERTScore, pair accuracy, empty answers per run and per experiment
python scripts/build_eval_index.py   # eval_logs/EVALUATIONS.md: every run with its parameters and metrics
python scripts/retrieval_recall.py intfloat/e5-base-v2 <fine-tuned retriever dir>   # retrieval-only check
```

`eval_3x.py` writes `run/eval_results_3x_runs.csv` (one row per repeat) and
`run/eval_results_3x_agg.csv` (mean ± sd over repeats). BERTScore uses
`roberta-large` with baseline rescaling, and unanswered questions score 0. Pair
accuracy counts a pair only when both of its questions are answered correctly.

## Fine-tuning the generator

The golden-document configs start from the Search-R1 checkpoint and train with GRPO
(reward 0.5·EM + 0.5·token F1). Every `<search>` on a negation question returns its
gold document, the minimally different document of its pair and an unrelated document,
in a fixed per-row order:

| Config | Training data | Evaluated as |
|---|---|---|
| `golden_nevir_nfs` | NevIR + Negations-Free-Sentence | `exp1` |
| `golden_nevir_nfwa` | NevIR + all of Negations-Free | `exp2` |
| `golden_nevir_nfwa_nq_hotpot` | as above + NQ / HotpotQA with live retrieval | `exp3` |

```bash
# golden documents only: no retrieval server needed
sbatch --time=2-00:00:00 --export=ALL,CONFIG_NAME=golden_nevir_nfs,NO_RETRIEVER=1 run/train.job

# mixed with live NQ / HotpotQA retrieval
RID=$(sbatch --parsable run/retrieval.job)
sbatch --dependency=after:$RID --time=3-00:00:00 \
    --export=ALL,CONFIG_NAME=golden_nevir_nfwa_nq_hotpot,RETRIEVER_JOB_ID=$RID run/train.job
```

Checkpoints are saved under `$SAVE_DIR_BASE/<experiment_name>/`.

## Fine-tuning the retriever

`retriever_finetune/` trains e5-base-v2 or SPLADE-v3 so that each question ranks its
own document above the other document of its pair. The loss is a two-stage
Plackett-Luce likelihood: the own document first among all candidates, then the
pair's other document first among the rest. Hard negatives are mined from the training
pool, and an optional MS MARCO replay term guards against forgetting.

```bash
python -m retriever_finetune.prepare_marco          # once: MS MARCO replay data
sbatch retriever_finetune/train_e5.job              # e5 + MS MARCO replay (defaults)
sbatch retriever_finetune/train_e5.job marco=none   # negation pairs only
sbatch retriever_finetune/train_e5.job model=splade_v3 loss.lam=0.5
```

Each run saves `best/` (chosen by validation pairwise accuracy), `last/` and
`metrics.json` under `$STORE_ROOT/retriever_checkpoints/<experiment>/<time>/`. A
fine-tuned retriever needs its own indexes, because the passage encoder changes too:
`run/run_negft_eval.sh` and `run/run_splade_eval.sh` build them and run the evaluations.

## Acknowledgements

This project builds on [SubSearch](https://arxiv.org/abs/2604.07415)
([model](https://huggingface.co/RoxanaMaria/SubSearch-qwen2.5-3b-grpo)),
[Search-R1](https://github.com/PeterGriffinJin/Search-R1) and
[veRL](https://github.com/volcengine/verl).

## License

Apache License 2.0; see [LICENSE](LICENSE) and [Notice.txt](Notice.txt).
