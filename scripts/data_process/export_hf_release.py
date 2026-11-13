"""Export the negation QA sets for a Hugging Face release: one pair per row, no Search-R1 prompt.

  negations-free-with-answer           from RoxanaMaria/gpt4o-negation-free
  negations-free-sentence-with-answer  its sentence-level subset (negation types 0, 1, 5, 6)
  nevir-with-answers                   from orionweller/NevIR

The pairs are exactly the ones the experiments used. Nothing is filtered here: the script only
drops the prompt, and checks record by record that every pair, question, answer and document
equals the files the experiments read before writing anything.

  set             experiments read (prompt format)               pair-format source
  negations-free  negations-free-with-answer/{split}_clean2      ~/data/negations_free_with_answer/{split}
                  (+ _docs; conflict-filtered, 966/129/284)      rows whose index is a pair_id there
  sentence        negations-free-sentence-with-answer/{split}    ~/data/negations_free_sentence_with_answer/{split}
  nevir           nevir-with-answers/{split}_clean               ~/data/nevir_with_answers/{split}_clean

The pair-format ~/data/negations_free_with_answer/{split}_clean2 files predate the conflict
filter (1,003 / 139 / 294 pairs), so they are not used.

Writes two Hugging Face repos; the README.md dataset cards in them are written by hand and
never touched:
  data/hf_release/negations-free-with-answer/{full,sentence}/{split}.parquet   (two configs)
  data/hf_release/nevir-with-answers/{split}.parquet
and, outside the repo folders so they are never uploaded: _stats/<set>.json (per-split
statistics quoted in the cards) and _review/<set>/validation.csv (for checking by hand).

    python3 scripts/data_process/export_hf_release.py
"""
import json
import os
import string

import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW = '/home/16207041/data'
RAG = os.path.join(BASE, 'data', 'negations_with_answer_rag')
OUT = os.path.join(BASE, 'data', 'hf_release')
SPLITS = ['train', 'validation', 'test']

NFWA, NFS, NEVIR = ('negations-free-with-answer', 'negations-free-sentence-with-answer',
                    'nevir-with-answers')
# set -> folder of its parquet files inside OUT (the sentence set is a config of the
# negations-free repo, since it is a subset of it), and its name for _stats / _review.
RELEASE_DIRS = {
    NFWA: ('negations-free-with-answer/full', 'negations-free-full'),
    NFS: ('negations-free-with-answer/sentence', 'negations-free-sentence'),
    NEVIR: ('nevir-with-answers', 'nevir'),
}
# What run/eval.py, the golden training sets and the retriever fine-tuning read:
# (question/answer file, golden-document file) per split.
EXPERIMENT_FILES = {
    NFWA: {s: (f'{s}_clean2', f'{s}_clean2_docs') for s in SPLITS},
    NFS: {'train': ('train', 'train_docs'), 'validation': ('validation', 'validation_docs'),
          'test': ('test', 'test_clean_docs')},
    NEVIR: {s: (f'{s}_clean', f'{s}_clean_docs') for s in SPLITS},
}
SENTENCE_TYPES = {0, 1, 5, 6}
NEGATION_TYPES = {   # same ids as scripts/gen_data/generate_answers.py
    0: 'sentential', 1: 'exceptor', 2: 'contradiction', 3: 'contrary', 4: 'subcontradiction',
    5: 'affixal', 6: 'implicit', 7: 'immediate antonyms', 8: 'mid antonyms', 9: 'polar antonyms',
}


def norm(s):
    s = str(s).lower()
    s = ''.join(' ' if c in string.punctuation else c for c in s)
    return ' '.join(s.split())


def experiment_file(name, split, which):
    return os.path.join(RAG, name, EXPERIMENT_FILES[name][split][which] + '.parquet')


# Each loader returns the released pair_id plus exp_pair_id, the id the experiment files use
# (dropped before writing). pair_id: the row in the upstream HF split for both negations-free
# sets, so a pair has the same id in both; NevIR's own id for NevIR. They differ only for the
# sentence set, whose experiment pair_id is the row in its own file (= release row order).
def load_nfwa(split):
    raw = pd.read_parquet(f'{RAW}/negations_free_with_answer/{split}.parquet').reset_index(drop=True)
    exp = pd.read_parquet(experiment_file(NFWA, split, 0), columns=['extra_info'])
    keep = {str(e['pair_id']) for e in exp.extra_info}
    df = raw[raw.index.astype(str).isin(keep)].copy()
    df.insert(0, 'pair_id', df.index.astype(str))
    df['exp_pair_id'] = df['pair_id']
    return df.reset_index(drop=True)


def load_nfs(split, nfwa_raw):
    df = pd.read_parquet(f'{RAW}/negations_free_sentence_with_answer/{split}.parquet')
    cols = ['q1', 'doc1', 'q2', 'doc2', 'negation', 'answer1', 'answer2']
    rows = {tuple(r): i for i, r in enumerate(nfwa_raw[cols].itertuples(index=False))}
    src = [rows.get(tuple(r)) for r in df[cols].itertuples(index=False)]
    assert None not in src, f'{split}: sentence pair not found in negations-free'
    df.insert(0, 'pair_id', [str(i) for i in src])
    df['exp_pair_id'] = df.index.astype(str)
    return df.reset_index(drop=True)


def load_nevir(split):
    df = pd.read_parquet(f'{RAW}/nevir_with_answers/{split}_clean.parquet')
    df.insert(0, 'pair_id', df['id'].astype(str))
    df['exp_pair_id'] = df['pair_id']
    return df.drop(columns=['id', 'WorkerId']).reset_index(drop=True)


def check_matches_experiments(name, split, df):
    """Same pairs, questions, answers and documents as the files the experiments read."""
    pairs = df.set_index('exp_pair_id')
    qa = pd.read_parquet(experiment_file(name, split, 0))
    assert len(qa) == 2 * len(df) and set(pairs.index) == {str(e['pair_id']) for e in qa.extra_info}, \
        f'{name} {split}: pairs differ from the experiment file'
    for p, rm, ei in zip(qa.prompt, qa.reward_model, qa.extra_info):
        row, i = pairs.loc[str(ei['pair_id'])], ei['query_role'][1]
        q = str(row[f'q{i}']).strip()
        q = q if q.endswith('?') else q + '?'   # create_clean2.make_prompt appends '?'
        assert p[0]['content'].rsplit('Question: ', 1)[1].strip() == q, (name, split, ei)
        assert str(rm['ground_truth']['target'][0]).strip() == str(row[f'answer{i}']).strip(), (name, split, ei)
    docs = pd.read_parquet(experiment_file(name, split, 1))
    assert len(docs) == 2 * len(df), f'{name} {split}: golden-doc file has {len(docs)} rows'
    for ei, gold, hard in zip(docs.extra_info, docs.gold_doc, docs.hard_neg_doc):
        row, i = pairs.loc[str(ei['pair_id'])], ei['query_role'][1]
        own, other = (row.doc1, row.doc2) if i == '1' else (row.doc2, row.doc1)
        # golden documents are '"<title>"\n<text>'
        assert gold.split('\n', 1)[1].strip() == own.strip(), (name, split, ei)
        assert hard.split('\n', 1)[1].strip() == other.strip(), (name, split, ei)


def finish(name, df):
    cols = ['pair_id']
    if name != NEVIR:
        df['negation'] = df['negation'].astype(int)
        df['negation_type'] = df['negation'].map(NEGATION_TYPES)
        cols += ['topic', 'negation', 'negation_type']
    return df[cols + ['q1', 'doc1', 'answer1', 'q2', 'doc2', 'answer2']].reset_index(drop=True)


def stats(df, name):
    a1, a2 = df.answer1.map(norm), df.answer2.map(norm)
    answers, docs = pd.concat([a1, a2]), pd.concat([df.doc1.map(norm), df.doc2.map(norm)])
    in_doc = [a in d for a, d in zip(answers, docs)]
    s = {
        'pairs': len(df), 'questions': 2 * len(df),
        'identical_answer_pairs': int((a1 == a2).sum()),
        'yes_no_answers': int(answers.isin(['yes', 'no']).sum()),
        'answer_in_own_document': round(sum(in_doc) / len(in_doc), 3),
        'mean_answer_words': round(answers.str.split().str.len().mean(), 2),
    }
    if name != NEVIR:
        s['topics'] = int(df.topic.nunique())
        s['pairs_by_negation_type'] = {NEGATION_TYPES[t]: int(n) for t, n in
                                       df.negation.value_counts().sort_index().items()}
    return s


def main():
    nfwa_raw = {s: pd.read_parquet(f'{RAW}/negations_free_with_answer/{s}.parquet') for s in SPLITS}
    loaders = {NFWA: load_nfwa, NFS: lambda s: load_nfs(s, nfwa_raw[s]), NEVIR: load_nevir}
    result = {}
    for name, load in loaders.items():
        data_dir, short = RELEASE_DIRS[name]
        for d in (os.path.join(OUT, data_dir), os.path.join(OUT, '_stats'), os.path.join(OUT, '_review', short)):
            os.makedirs(d, exist_ok=True)
        summary = {}
        for split in SPLITS:
            df = load(split)
            check_matches_experiments(name, split, df)
            df = finish(name, df)
            result[(name, split)] = df
            df.to_parquet(os.path.join(OUT, data_dir, f'{split}.parquet'), index=False)
            if split == 'validation':
                df.to_csv(os.path.join(OUT, '_review', short, 'validation.csv'), index=False)
            summary[split] = stats(df, name)
            print(f'{data_dir:36} {split:10} {len(df):4} pairs (= experiments)')
        with open(os.path.join(OUT, '_stats', f'{short}.json'), 'w') as f:
            json.dump(summary, f, indent=2)

    # The sentence set must be exactly the sentence-type pairs of negations-free.
    for split in SPLITS:
        full, sent = result[(NFWA, split)], result[(NFS, split)]
        expect = set(full.pair_id[full.negation.isin(SENTENCE_TYPES)])
        assert set(sent.pair_id) == expect, f'{split}: sentence set is not the subset of negations-free'
    print(f'checks passed; written to {OUT}')


if __name__ == '__main__':
    main()
