#!/usr/bin/env python3
"""Pick the best run of a grid on validation (decodes val at fp16 for each run if needed).

  python scripts/select_run.py --grid rw --runs rw_base_a0.5_b0 ... --baseline ft_base
  python scripts/select_run.py --grid tl --baseline full_base__fp16__full_val \
      --runs fulltl_base_g0.5 fulltl_base_g1 --control fulltl_base_g0 [--tag lam --cwer_margin 1.0]
Criteria (fixed before looking at test):
  rw   lowest pooled val embedded error (cs_E_emb) among runs whose pooled val MER is at most
       1 point above the baseline's
  sem  lowest pooled val sMER among runs whose pooled val CWER is not above the baseline's
  tl   (third run, 26 Oct; docs/PROCESS_LOG.md) all metrics are the mean over ascend_val and
       nsc_val. Eligible = val MER <= baseline MER + 0.5 points AND val CWER (rule tagger) <=
       baseline CWER + --cwer_margin points. Choose the eligible run with the lowest val
       translit_rate; none eligible = negative result. Controls are reported but never selected.
rw/sem write results/select_<grid>.json and print the chosen run name; tl writes
results/chosen_<tag>.json and results/<tag>_val.csv.
"""
import argparse
import json
import os
import subprocess
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from csasr.data import manifest  # noqa: E402
from csasr.score import report as R  # noqa: E402
from csasr.score.metrics import corpus_metrics  # noqa: E402

VAL = ("ascend_val", "nsc_val")
KEYS = ("MER", "CWER", "E_E", "cs_E_emb", "translit_rate", "conf_rate")


def val_metrics(run):
    name = f"{run}__fp16__val"
    if not os.path.exists(manifest.path(f"preds/{name}.meta.json")):
        subprocess.run([sys.executable, manifest.path("scripts/decode.py"), "--model", f"ckpt/{run}/best",
                        "--model_tag", run, "--precision", "fp16", "--lang", "free", "--method", "val",
                        "--sets", *manifest.EVAL_VAL, "--batch", "48"], check=True)
    st = R.stats(name)
    pooled = [x for s in manifest.EVAL_VAL for x in st[s][1]]
    return corpus_metrics(pooled)


def val_row(name, system, role):
    st = R.stats(name)
    per = {s: corpus_metrics(st[s][1]) for s in VAL}
    row = dict(system=system, pred_file=name, role=role)
    for k in KEYS:
        for s in VAL:
            row[f"{k}_{s}"] = per[s][k]
        row[f"{k}_mean"] = sum(per[s][k] for s in VAL) / len(VAL)
    return row


def select_grid(a):
    base = val_metrics(a.baseline)
    res = {a.baseline: base}
    for r in a.runs:
        res[r] = val_metrics(r)
    if a.grid == "rw":
        ok = [r for r in a.runs if res[r]["MER"] <= base["MER"] + 0.01]
        best = min(ok or a.runs, key=lambda r: res[r]["cs_E_emb"])
    else:
        ok = [r for r in a.runs if res[r]["CWER"] <= base["CWER"]]
        best = min(ok or a.runs, key=lambda r: res[r]["sMER"])
    keep = ("MER", "sMER", "CWER", "cs_E_emb", "cs_E_mat", "conf_rate")
    json.dump(dict(grid=a.grid, baseline=a.baseline, chosen=best, eligible=ok,
                   val={k: {m: v[m] for m in keep} for k, v in res.items()}),
              open(manifest.path(f"results/select_{a.grid}.json"), "w"), indent=1)
    print(best)


def select_tl(a):
    base = val_row(a.baseline, a.baseline.split("__")[0], "baseline")
    rows = [base]
    for r in a.control:
        rows.append(val_row(f"{r}__fp16__fulltl_val", r, "control (not selectable)"))
    for r in a.runs:
        x = val_row(f"{r}__fp16__fulltl_val", r, "candidate")
        x["mer_ok"] = x["MER_mean"] <= base["MER_mean"] + 0.005
        x["cwer_ok"] = x["CWER_mean"] <= base["CWER_mean"] + a.cwer_margin / 100
        x["eligible"] = x["mer_ok"] and x["cwer_ok"]
        rows.append(x)
    el = [x for x in rows if x.get("eligible")]
    chosen = min(el, key=lambda x: x["translit_rate_mean"])["system"] if el else None
    for x in rows:
        x["chosen"] = x["system"] == chosen
    df = pd.DataFrame(rows)
    df.to_csv(manifest.path(f"results/{a.tag}_val.csv"), index=False)
    out = dict(chosen=chosen, negative_result=chosen is None,
               gamma=float(chosen.split("_g")[-1].rstrip("r")) if chosen else None, baseline=a.baseline,
               rule=f"mean over ascend_val+nsc_val; MER <= base+0.5pt and CWER <= base+{a.cwer_margin}pt; min translit_rate")
    json.dump(out, open(manifest.path(f"results/chosen_{a.tag}.json"), "w"), indent=1)
    cols = ["system", "role", "MER_mean", "CWER_mean", "E_E_mean", "translit_rate_mean", "conf_rate_mean"]
    cols += [c for c in ("eligible", "chosen") if c in df]
    print(df[cols].to_string(index=False))
    print(json.dumps(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", required=True, choices=["rw", "sem", "tl"])
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--control", nargs="*", default=[], help="tl: runs reported but never selected")
    ap.add_argument("--tag", default="tl", help="tl: outputs results/chosen_<tag>.json and results/<tag>_val.csv")
    ap.add_argument("--cwer_margin", type=float, default=0.0, help="tl: points of val CWER allowed above baseline")
    a = ap.parse_args()
    select_tl(a) if a.grid == "tl" else select_grid(a)


if __name__ == "__main__":
    main()
