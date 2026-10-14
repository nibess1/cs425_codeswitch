#!/usr/bin/env python3
"""Pick the best run of a grid on validation (decodes val at fp16 for each run).

  python scripts/select_run.py --grid rw --runs rw_base_a0.5_b0 ... --baseline ft_base
Criteria (fixed before looking at test):
  rw   lowest pooled val embedded error (cs_E_emb) among runs whose pooled val MER is at most
       1 point above the baseline's
  sem  lowest pooled val sMER among runs whose pooled val CWER is not above the baseline's
Writes results/select_<grid>.json and prints the chosen run name.
"""
import argparse
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from csasr.data import manifest  # noqa: E402
from csasr.score import report as R  # noqa: E402
from csasr.score.metrics import corpus_metrics  # noqa: E402


def val_metrics(run):
    name = f"{run}__fp16__val"
    if not os.path.exists(manifest.path(f"preds/{name}.meta.json")):
        subprocess.run([sys.executable, manifest.path("scripts/decode.py"), "--model", f"ckpt/{run}/best",
                        "--model_tag", run, "--precision", "fp16", "--lang", "free", "--method", "val",
                        "--sets", *manifest.EVAL_VAL, "--batch", "48"], check=True)
    st = R.stats(name)
    pooled = [x for s in manifest.EVAL_VAL for x in st[s][1]]
    return corpus_metrics(pooled)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", required=True, choices=["rw", "sem"])
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--baseline", required=True)
    a = ap.parse_args()
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


if __name__ == "__main__":
    main()
