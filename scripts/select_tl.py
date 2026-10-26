#!/usr/bin/env python3
"""Third run (26 Oct): apply the transliteration-loss selection rule on validation.

  python scripts/select_tl.py --baseline full_base__fp16__full_val \
      --runs fulltl_base_g0.5 fulltl_base_g1 --control fulltl_base_g0
Rule (docs/PROCESS_LOG.md, fixed before training): all metrics are the mean over ascend_val and
nsc_val. Eligible = val MER <= baseline MER + 0.5 points AND val CWER (rule tagger) <= baseline
CWER. Choose the eligible run with the lowest val translit_rate; none eligible = negative result.
The gamma = 0 control is reported but never selected.
Writes results/chosen_tl.json and results/tl_val.csv.
"""
import argparse
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from csasr.data import manifest  # noqa: E402
from csasr.score import report as R  # noqa: E402
from csasr.score.metrics import corpus_metrics  # noqa: E402

VAL = ("ascend_val", "nsc_val")
KEYS = ("MER", "CWER", "E_E", "cs_E_emb", "translit_rate", "conf_rate")


def val_row(name, system, role):
    st = R.stats(name)
    per = {s: corpus_metrics(st[s][1]) for s in VAL}
    row = dict(system=system, pred_file=name, role=role)
    for k in KEYS:
        for s in VAL:
            row[f"{k}_{s}"] = per[s][k]
        row[f"{k}_mean"] = sum(per[s][k] for s in VAL) / len(VAL)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--control", nargs="*", default=[])
    a = ap.parse_args()
    base = val_row(a.baseline, a.baseline.split("__")[0], "baseline")
    rows = [base]
    for r in a.control:
        rows.append(val_row(f"{r}__fp16__fulltl_val", r, "control (not selectable)"))
    for r in a.runs:
        x = val_row(f"{r}__fp16__fulltl_val", r, "candidate")
        x["mer_ok"] = x["MER_mean"] <= base["MER_mean"] + 0.005
        x["cwer_ok"] = x["CWER_mean"] <= base["CWER_mean"]
        x["eligible"] = x["mer_ok"] and x["cwer_ok"]
        rows.append(x)
    el = [x for x in rows if x.get("eligible")]
    chosen = min(el, key=lambda x: x["translit_rate_mean"])["system"] if el else None
    for x in rows:
        x["chosen"] = x["system"] == chosen
    df = pd.DataFrame(rows)
    df.to_csv(manifest.path("results/tl_val.csv"), index=False)
    out = dict(chosen=chosen, negative_result=chosen is None,
               gamma=float(chosen.split("_g")[-1].rstrip("r")) if chosen else None, baseline=a.baseline,
               rule="mean over ascend_val+nsc_val; MER <= base+0.5pt and CWER <= base; min translit_rate")
    json.dump(out, open(manifest.path("results/chosen_tl.json"), "w"), indent=1)
    cols = ["system", "role", "MER_mean", "CWER_mean", "E_E_mean", "translit_rate_mean", "conf_rate_mean"]
    cols += [c for c in ("eligible", "chosen") if c in df]
    print(df[cols].to_string(index=False))
    print(json.dumps(out))


if __name__ == "__main__":
    main()
