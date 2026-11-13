# Datasets

What is in `data/`, where each dataset came from, and how the negation sets were
cleaned at each step. Counts are **pairs** (each pair gives two eval rows, `q1` and `q2`)
and were read from the files on 2026-09-28.

## Negation evaluation sets

All four live in `data/negations_with_answer_rag/<name>/`. Each pair is a minimal pair:
two questions that differ by a negation, each with its own document and short answer.

| dataset | tag in logs | test pairs (file used by eval) | negated query | source |
|---|---|---|---|---|
| negations-free-with-answer | `nfwa` | 284 (`test_clean2`; 294 before the conflict filter) | `q1` | HF `RoxanaMaria/gpt4o-negation-free` |
| negations-free-sentence-with-answer | `nfs` | 122 (`test_clean`) | `q1` | subset of the one above |
| nevir-with-answers | `nevir` | 662 (`test_clean`) | `q2` (approximate) | HF `orionweller/NevIR` |
| negation-controlled-with-answers | — | 273 (`test_clean2`) | `q1` | HF `RoxanaMaria/gpt4o-negation-controlled` |

- **Negations-free** covers all 10 negation types (table below). **Sentence** keeps only 4 of them.
- **NevIR** pairs are contrastive rather than strictly positive/negated ("hardly ever" vs
  "always"), so its positive/negated split is approximate; pair accuracy does not depend on it.
- **Negation-controlled** has its own questions (4 of 305 test questions overlap with
  negations-free) and is not used by the current experiments.
- Which query is negated is set per dataset in `negation.q1_negated_sources`
  (`verl/trainer/config/ppo_trainer.yaml`).

### Row format

Every parquet row has `data_source`, `prompt` (the Search-R1 instruction plus the question),
`ability`, `reward_model.ground_truth.target` (the answer) and
`extra_info` (`pair_id`, `query_role` = `q1`/`q2`, `negation_type`, `split`).
Built by `scripts/data_process/negations_with_answer_rag.py`. Documents are **not** in the
prompt: the model retrieves them.

## How each set was built and cleaned

### Negations-free (and negation-controlled, same pipeline)

| step | what happened | test pairs | script |
|---|---|---|---|
| 1. raw | Downloaded from HF: `topic, q1, doc1, q2, doc2, negation` | 310 | `scripts/gen_data/` |
| 2. answers | GPT-4.1-mini writes a short answer per question from its own document (`answer1` from `doc1`, then `answer2` from `doc2` with `answer1` as reference; identical answers re-asked once) | 310 | `scripts/gen_data/generate_answers.py` |
| 3. `_clean` | **Per-record** filter, done outside this repo (no script). What it removed, reconstructed from the files: some yes/no answers and some records whose pair had identical answers, plus a few others. It is **not** a full removal of either: 149 yes/no answers and 43 identical-answer pairs remain in test. Dropping single records left 19 test pairs with only one half | 301 (583 rows) | — |
| 4. `_clean2` | **Per-pair**: keep pairs present in `_clean`, then drop any pair with a none-like answer (`not mentioned`, `none`, `n/a`, `not specified`, …). Both records of every surviving pair are re-emitted, so no pair is incomplete | 294 | `scripts/data_process/create_clean2.py` |
| 5. conflict filter | **Per-pair, in place** (2026-09-28): drop pairs whose question has a conflicting label, a missing answer, or (validation only) repeats a train or test question. See [Conflicting labels](#conflicting-labels-and-the-conflict-filter) | 284 | `scripts/data_process/drop_conflicting_pairs.py` |
| 6. `_docs` | Adds golden-document columns for the no-retrieval eval (see below) | 284 | `scripts/data_process/negations_golden_docs.py` |

`_clean2` has *more* rows than `_clean` (588 vs 583): it re-adds the missing halves of pairs
that `_clean` had cut in half.

Negation-controlled follows steps 1–4: 305 → 283 (`_clean`, 7 incomplete pairs) → 273 pairs.
It is unused and was **not** conflict-filtered.

### Negations-free-sentence

Taken from negations-free step 4 (`_clean2`), keeping only negation types **0, 1, 5, 6**
(`scripts/data_process/create_sentence_subset.py`): 294 → **122** test pairs.
`test_clean.parquet` is identical to `test.parquet` (already clean).

Why the other types were dropped (checked with an NLI model and answer grounding):

| type | problem |
|---|---|
| 2 contradiction | doc1 and doc2 contradict each other ("*All* X are P" vs "*some* X are not P"); 62% of pairs flagged by NLI, and many pairs claim both are true |
| 3 contrary, 4 subcontradiction | every answer is yes/no; in type 4, 76% of pairs have the same answer for q1 and q2, so answering "yes" twice scores |
| 7–9 antonyms | poor answers: 22–32% are not in their own document (e.g. "white-and-black cinematography" → `Color film`) |

### NevIR

| step | what happened | test pairs | script |
|---|---|---|---|
| 1. raw | HF NevIR: `id, q1, q2, doc1, doc2` (no topic, no negation type) | 1383 | `scripts/data_process/nevir_rag.py` |
| 2. answers | Generated twice with GPT-4.1-mini; the second generation regressed, answering "not mentioned" where the first had a real answer | 1383 | `scripts/gen_data/generate_answers.py` |
| 3. fix + drop | Restore the first generation's answer where the second is none-like (305 rows fixed), then **drop pairs with a none-like answer or identical answers** — 52% of test | 663 | `scripts/fix_nevir.py` (log: `~/data/nevir_with_answers_fixed/fix_log.txt`) |
| 4. `_clean` | Per-pair none-like filter (the same rule as `_clean2` above); removes 1 pair per split | 662 | `scripts/data_process/create_clean2.py` |
| 5. `_docs` | Golden-document columns | 662 | `scripts/data_process/negations_golden_docs.py` |

`test_original.parquet` keeps all 1383 pairs from before step 3. `pair_id` is NevIR's own `id`
(e.g. `1002-3`), not a row index.

### Where the remaining differences come from

Counts below are for the test files the eval uses.

- **Yes/no answers:** common in negations-free (145 of 568 rows, mostly types 2–4), rare in
  sentence (13 of 244), almost absent from NevIR (1 of 1324).
- **Identical answers within a pair:** still in negations-free (44 of 284 pairs) and
  sentence (4 of 122); none in NevIR, whose step 3 removed them.
- **Answer quality:** all answers are LLM-generated from the document, not human-written.

See [Data quality problems](#data-quality-problems) for the same measurements on every
split, and what they mean for training.

## Negation types (negations-free family)

| id | type | in sentence subset | test pairs in `_clean2` |
|---|---|---|---|
| 0 | sentential (no / not / none) | ✅ | 11 |
| 1 | exceptor (except / besides / but) | ✅ | 29 |
| 2 | contradiction (yes/no) | | |
| 3 | contrary (yes/no) | | |
| 4 | subcontradiction (yes/no) | | |
| 5 | affixal (un- / non- / in-) | ✅ | 41 |
| 6 | implicit | ✅ | 41 |
| 7 | immediate antonyms | | |
| 8 | mid antonyms | | |
| 9 | polar antonyms | | |

## Golden-document variants (`*_docs.parquet`)

Same rows as the base file, plus five columns used by the golden retriever modes
(`retriever.mode` = `golden`, `golden_hn`, `golden_rn`, `golden_hn_rn`; no retriever is
called, and each `<search>` returns these). Built for **every split** by
`python scripts/data_process/negations_golden_docs.py --dataset all --split all`. The test
files feed the golden evals, and the train/validation files feed the golden training sets
below.

| column | content |
|---|---|
| `gold_doc` | the question's own document (`q1` → `doc1`, `q2` → `doc2`) |
| `hard_neg_doc` | the other half of the pair |
| `rand_neg_doc` | a document from another pair — another **topic** where topics exist; any other pair for NevIR |
| `doc_order` | whether gold is shown first or second (fixed per row, balanced ~50/50) |
| `doc_order3` | slot order of gold / hard / random for `golden_hn_rn` (one of the 6 permutations, fixed per row, roughly balanced) |

Documents are stored as `"<title>"\n<text>`; the title is the `topic`, or the constant
`Document` for NevIR, which has no topic. Everything is a deterministic function of
`pair_id` + `query_role`, so rebuilding reproduces the same file.

- The random negative comes from the **same split's** file, so no test document reaches
  training.
- The same document text can appear in several pairs: NevIR pairs with the same id prefix
  (`102-2`, `102-3`) share a document, and negations-free reuses passages across questions
  (pairs `0` and `112`). A random draw that returns the row's own gold or hard document is
  redrawn among documents with different text. The test files never hit this case and are
  unchanged.
- Pairs with a missing answer are skipped (negations-free train pair 59). The conflict
  filter now also removes it from the base file.
- The negations-free `_docs` files were **regenerated** after the conflict filter. The
  random-negative pool shrank, so some random draws differ from the pre-filter files; the
  pre-filter versions are kept in `pre_conflict_filter/`.

| split files | pairs (nevir / negations-free / sentence) |
|---|---|
| `train_clean_docs`, `train_clean2_docs`, `train_docs` | 474 / 966 / 417 |
| `validation_clean_docs`, `validation_clean2_docs`, `validation_docs` | 118 / 129 / 58 |
| test (above) | 662 / 284 / 122 |

The sentence files are a strict subset of the negations-free files in every split (all
833 train, 116 validation and 244 test questions are also in the filtered negations-free).

| dataset config (`eval/dataset=…`) | file |
|---|---|
| `negations_free_with_answer_docs` | `negations-free-with-answer/test_clean2_docs.parquet` |
| `negations_free_sentence_with_answer_docs` | `negations-free-sentence-with-answer/test_clean_docs.parquet` |
| `nevir_with_answers_docs` | `nevir-with-answers/test_clean_docs.parquet` |

## Retrieval corpora

| corpus | contents | used by |
|---|---|---|
| wiki-18 (`/fnwi_fs/ivi/irlab/projects/chatr1/collection`) | 21M Wikipedia passages, e5 FAISS index (60 GB) | `eval/retriever=wiki`, `running` |
| `data/custom_index/` | 3,346 documents: doc1/doc2 of negations-free-sentence and NevIR (all splits), e5 index | `eval/retriever=custom`, `custom_2`, `custom_x2` |

The custom corpus holds the gold document for **100%** of sentence and NevIR test rows but only
**42%** of negations-free rows, so the custom retriever is not run on negations-free.

## Training and other data

| folder | contents |
|---|---|
| `combined_nq_hotpot_nevir_negfree/` | training mix: NQ 79k + HotpotQA 90k + NevIR train (474 pairs, `train_clean`) + sentence train (417 pairs) |
| `combined_nq_hotpot_nevir_negfree_upsampled/` | same, negation sets upsampled 11× (≈10k rows each); the default `DATA_PATH` in `run/env.sh` |
| `combined_…_upsampled_decomposed/` | same rows, prompt asking for sub-question decomposition (SubSearch) |
| `negation_only_base/`, `negation_only_decomposed/` | NevIR + sentence only (train / val / test) |
| `golden_<setup>/`, `golden_<setup>_decomposed/` | golden-document training sets (next section) |
| `nq_search/` | NQ with the Search-R1 prompt |
| `multihop_search/` | NQ, HotpotQA, MuSiQue, 2Wiki, PopQA, TriviaQA, … with the Search-R1 prompt |
| `negation_rag/gpt4o-negation-controlled/` | older framing: both documents in the prompt, answer is "document 1" / "document 2" |

Raw sources and intermediate files live outside the repo in `/home/16207041/data/`.

### Golden-document training sets

Built by `scripts/data_process/build_golden_training.py --setup <setup> [--template decomposed]`.
The script needs the train/validation `_docs` files described above. Negation rows carry
the golden-document columns and are trained with `retriever.mode=golden_hn_rn`, where each
`<search>` returns gold + hard negative + random document in `doc_order3`. NQ/HotpotQA rows
have empty document columns and are searched live on wiki-18 (`retriever.live_fallback=true`).

| setup / config | train rows | val rows | retriever job |
|---|---|---|---|
| `nevir_nfs` / `golden_nevir_nfs` | 1,782: NevIR 948 + sentence 834 | 352 | not needed (`NO_RETRIEVER=1`) |
| `nevir_nfwa` / `golden_nevir_nfwa` | 2,880: NevIR 948 + negations-free 1,932 | 494 | not needed (`NO_RETRIEVER=1`) |
| `nevir_nfwa_nq_hotpot` / `golden_nevir_nfwa_nq_hotpot` | 8,640: the above + NQ 2,880 + HotpotQA 2,880 (seed 42) | 994 (incl. NQ 250, HotpotQA 250) | needed |

- Setup 2 does not add the sentence set, since its rows are already inside negations-free.
- NQ/HotpotQA are sampled from `combined_nq_hotpot_nevir_negfree`, with as many rows each
  as there are negation rows.
- Validation rows use golden documents too. Transfer to live retrieval is measured
  separately with `run/eval.py`.
- Negations-free rows are the conflict-filtered `_clean2`. The builder refuses to write
  if a train question appears in a test set, or a validation question in train or test.
  The conflict filter removes those at the source.
- Each setup directory holds `train.parquet` and `val.parquet`. Test sets are the
  `test_clean*_docs.parquet` files (evaluate with `eval/retriever=golden_*` or live).
- Columns: the 5 trainer columns plus `gold_doc`, `hard_neg_doc`, `rand_neg_doc`,
  `doc_order`, `doc_order3`. `extra_info` is unified to `split, index, pair_id,
  query_role, negation_type` for every source. Train is shuffled (seed 42); val keeps the
  two halves of each pair adjacent.
- The builder refuses to write if a pair is incomplete, a source is missing the golden
  columns, or a train question appears in a test set. `decomposed` copies are made with
  `retemplate.convert_frame`, which fails if any prompt lacks the base prefix.
- Configs `verl/trainer/config/golden_*.yaml`: Search-R1 checkpoint, plain EM
  (`reward_model.answer_em_weight` switches to EM + F1), greedy validation, 150 steps
  (setup 3: 300).
- Setup 1 contains no negations-free-with-answer rows and was unchanged by the conflict
  filter (rebuilt byte-identical).

## Data quality problems

Measured on the final `_docs` files, after the conflict filter (answers normalized as for EM: lowercase, no punctuation
or articles). "In own doc" means the answer string appears in its question's document; "in
other doc" means it appears in the other half of the pair. Document length is in Qwen tokens.

| dataset | split | pairs | answer in own doc | in other doc | only in own | identical-answer pairs | yes/no rows | answer words | doc tokens (median / p95 / max) | doc1–doc2 word overlap |
|---|---|---|---|---|---|---|---|---|---|---|
| NevIR | train | 474 | 59% | 53% | 6% | 0 | 0 | 2.4 | 146 / 238 / 291 | 0.91 |
| NevIR | validation | 118 | 63% | 55% | 8% | 0 | 0 | 2.2 | 161 / 233 / 269 | 0.91 |
| NevIR | test | 662 | 61% | 58% | 4% | 1 | 1 | 2.3 | 157 / 245 / 297 | 0.92 |
| negations-free | train | 966 | 24% | 7% | 19% | 139 (14%) | 524 (27%) | 3.1 | 39 / 88 / 250 | 0.21 |
| negations-free | validation | 129 | 23% | 8% | 17% | 21 (16%) | 64 (25%) | 3.1 | 40 / 77 / 137 | 0.18 |
| negations-free | test | 284 | 24% | 6% | 19% | 44 (15%) | 145 (26%) | 3.0 | 39 / 83 / 174 | 0.21 |
| sentence | train | 417 | 26% | 6% | 21% | 17 (4%) | 48 (6%) | 3.9 | 51 / 106 / 250 | 0.23 |
| sentence | validation | 58 | 25% | 9% | 17% | 2 (3%) | 4 (3%) | 4.0 | 47 / 92 / 113 | 0.19 |
| sentence | test | 122 | 30% | 7% | 23% | 4 (3%) | 13 (5%) | 3.6 | 48 / 95 / 174 | 0.20 |

What this means:

1. **Most answers are not spans of the document.** Outside NevIR only ~25% of answers
   appear verbatim in their document. The LLM wrote them from the document, sometimes as a
   derived value. Exact match therefore often scores a correct paraphrase as wrong. With
   the gold document shown, the Search-R1 checkpoint reaches token F1 ≈ 0.38 but EM only
   0.15–0.24 (`eval_logs/EVALUATIONS.md`). Consider `answer_em_weight` < 1 for training.
2. **NevIR answers are mostly in both documents.** The two documents share ~91% of their
   words, and only 4–8% of answers appear in the gold document alone. The model must
   understand the negation, not find a unique string. It is the hardest set, and the
   `retrieval_score` bonus of the SubSearch reward ("gold string in a retrieved doc")
   rewards reading the wrong half as often as the right one.
3. **Identical answers within a pair** (negations-free 14–16%, sentence 3–4%) cannot reward
   telling the two questions apart. Answering the same thing twice is correct for both, so
   they inflate pair accuracy.
4. **Yes/no answers** make up 25–27% of negations-free rows (types 2–4) but ~5% of sentence
   rows. A yes/no answer scores 0.5 by guessing. Together with 3, this is why setup 2
   (`nevir_nfwa`) trades answer quality for 1.7× the data.
5. **Documents are short outside NevIR** (median ~40–50 tokens). Three golden documents fit
   easily in `max_obs_length: 900`. NevIR's are ~150 tokens, ~450 for three.
6. **Positive/negated labels are approximate for NevIR** (contrastive pairs); pair
   accuracy does not depend on them.
7. **Conflicting labels:** the same question with the same document had two different
   gold answers. These pairs are now dropped; see the next section.
8. **Repeated questions with the same answer remain.** Upstream negations-free reuses one
   half-pair in several pairs. After the filter, 157 train rows (8%), 20 validation rows and
   42 test rows repeat a question already in the same split, always with the **same**
   answer. They only upweight those rows in training, and in test the shared half counts
   in more than one pair. They are kept, because dropping a repeat also drops the other,
   unique half of its pair.
9. **Documents shared across splits without a shared question**, all upstream: NevIR
   train/test 1 and validation/test 2; negations-free train/test 1. Harmless for
   golden training, since the question differs.

## Conflicting labels and the conflict filter

**How the conflicts were created.** For the quantifier negation types (2 contradiction,
3 contrary, 4 subcontradiction), upstream negations-free reuses the **same negated question
and document** in two pairs of different types:

| pair | type | q1 (the other half, differs) | q2 (shared, same doc2) |
|---|---|---|---|
| 1042 | 2 contradiction | "For **all** data scientists, do they use programming and statistics…?" → yes | "Does there exist a data scientist that does **not** use…?" |
| 123 | 4 subcontradiction | "Are there **some** data scientists who use programming and statistics…?" → yes | *(identical)* |

The shared doc2 reads *"There exists a data scientist who does not use both programming
and statistical knowledge…"*, so the right answer to q2 is **yes**. `scripts/gen_data/generate_answers.py`
answers q2 once **per pair**, at temperature 0, but with that pair's q1, answer1 and
negation-type hint in the prompt ("Related question… Related answer: yes"). In the type-2
pair, where q2 is the logical opposite of q1, it answered **no**, following q1 rather than
doc2. In the type-4 pair ("some do" / "some don't" can both be true) it answered **yes**.
Its same-answer fix pass ("your answer MUST be different from the related answer") may
push towards "no" too; the generation logs are not available to confirm.

All 15 conflicting q2 questions in train follow exactly this pattern: the type-4 copy equals
answer1 ("yes"), and the type-2 copy differs ("no"). Another 3 conflicts are q1 wording
differences between runs ("extreme weather rising sea levels **altered** ecosystems" vs
"…ecosystems").

**Why it matters.** The question and gold document are identical, and only the distractor
differs, which should not change the answer. Every answer is then rewarded on one copy and
penalized on the other. One of the two labels is wrong, so the type-2 copy teaches the model to
answer against its document. In test, one copy is scored wrong whatever the model says.

**What the filter drops** (`scripts/data_process/drop_conflicting_pairs.py`, in place,
2026-09-28): every pair (both halves) containing a question whose label conflicts within
its split, pairs with a missing answer, and validation pairs that repeat a train or test
question. Detection is by question text + gold document on the `_docs` files; the base and
`_docs` files are filtered by pair_id, and the `_docs` files then regenerated.

| split | conflict | missing answer | val repeats train | val repeats test | pairs before → after |
|---|---|---|---|---|---|
| train | 36 pairs (18 questions; types 2: 18, 4: 15, 3: 3) | 1 (pair 59) | — | — | 1,003 → 966 |
| validation | 4 (2 questions) | — | 3 (pairs 1, 69, 113) | 4 (74, 90, 105, 107) | 139 → 129 |
| test | 10 (5 questions) | — | — | — | 294 → 284 |

Only negations-free-with-answer was affected. NevIR and the sentence set have no
conflicts, and none of the dropped questions are in the sentence set. So `combined_*`,
`negation_only_*`, `custom_index/` and golden setup 1 are unchanged. Affected downstream:
golden setups 2 and 3 (rebuilt), both negations-free eval dataset configs, and every
negations-free eval result.

- **Originals:** `data/negations_with_answer_rag/negations-free-with-answer/pre_conflict_filter/`
  (the six base and `_docs` files as they were, never overwritten).
- **Log:** `…/negations-free-with-answer/dropped_pairs.csv`: split, pair_id, reason, question,
  answer, and the conflicting answers.
- **Earlier negations-free evals** (`eval_logs/EVALUATIONS.md`, 294 test pairs) are not
  directly comparable with runs on the filtered 284-pair test set. Rerun, or compare on the
  284 surviving pairs.
- **Probably wider than the duplicates.** The same priming can mislabel type 2/3 pairs that
  are not duplicated. A text match finds raw train pairs whose doc2 contains "there
  exists / there is / there are" while answer2 is "no": type 2 has 23, type 3 has 28, type 4
  has 7. "There is **no** X" also matches, so these are suspects, not confirmed errors. The
  fix would be to re-answer q2 from doc2 alone, without the related question and answer.

## Known inconsistencies

- `scripts/gen_data/download_negation_controlled.py` says it downloads the *controlled* set
  but its `HF_REPO` points at `gpt4o-negation-free`; the files on disk are two different
  datasets, so the controlled one was fetched some other way.
- The `_clean` step has no script in this repo; its description in `create_clean2.py`
  ("removed answer1==answer2 and yes/no answers") overstates what the files show.
- The `_clean2` none-like filter does not catch a missing (`None`) answer, which it writes
  as the string `'nan'` (negations-free train pair 59; removed by the conflict filter).
- `eval/dataset=negations_free_with_answer` read `test_clean` (per-record filter, 19 half
  pairs) until 2026-09-28; it now reads the filtered `test_clean2`, like the `_docs` config.
- The `swap_q_roles` key in `verl/trainer/config/eval/dataset/*.yaml` is no longer read;
  `negation.q1_negated_sources` decides which query is negated.
