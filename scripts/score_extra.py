import sys
from pathlib import Path
import pandas as pd
BASE=Path('.').resolve()
sys.path.insert(0,str(BASE/'scripts')); sys.path.insert(0,str(BASE))
from eval_3x import extract_run, metrics_for_run
from bertscore_eval import compute_bertscore_best_of
from verl.utils.reward_score.qa_em import em_check, token_f1_score

RUNS=[
 {"path":"eval_logs/searchr1-eval-329951/eval.log","ds_tag":"nevir","dataset":"nevir_with_answers",
  "model":"searchr1","retr":"wiki-top5","rep":1,"label":"329951"},
 {"path":"eval_logs/searchr1-eval-329952/eval.log","ds_tag":"nevir","dataset":"nevir_with_answers",
  "model":"searchr1_it","retr":"wiki-top5","rep":1,"label":"329952"},
]
frames=[]
for r in RUNS:
    recs=extract_run(r)
    g=pd.DataFrame(recs); g["_run"]=r["label"]; g["_model"]=r["model"]; g["_retr"]=r["retr"]
    frames.append(g)
df=pd.concat(frames,ignore_index=True)
hyps=df["extracted_answer"].fillna("").tolist(); refs=df["golden"].tolist()
print(f"BERTScore over {len(df)} records...")
df["bs_f1"]=compute_bertscore_best_of(hyps,refs)
df["em"]=[em_check(h,r) for h,r in zip(hyps,refs)]
df["tok_f1"]=[token_f1_score(h,r) for h,r in zip(hyps,refs)]
rows=[]
for label,gg in df.groupby("_run"):
    m=metrics_for_run(gg.reset_index(drop=True),"nevir_with_answers")
    rows.append({"run":label,"model":gg["_model"].iloc[0],"retr":gg["_retr"].iloc[0],**m})
out=pd.DataFrame(rows)
out.to_csv("run/eval_results_extra.csv",index=False)
print(out.to_string(index=False))
