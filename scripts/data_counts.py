#!/usr/bin/env python3
"""Size check before training (proposal §3.1): utterances, mixed utterances, embedded units,
critical words per split -> results/data_counts.{md,csv}."""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from csasr.data import manifest  # noqa: E402
from csasr.text.units import label  # noqa: E402


def main():
    rows = []
    for name in ["ascend_train", "ascend_val", "ascend_test", "nsc_train", "nsc_val", "nsc_test", "nsc_diff_test",
                 "stress_test"]:
        p = manifest.path(manifest.SETS[name])
        if not os.path.exists(p):
            continue
        rs = manifest.load(name)
        labs = [label(r["ref"], already_normalized=True) for r in rs]
        for main in ["all", "zh", "en", "balanced"]:
            sel = [(r, l) for r, l in zip(rs, labs) if main == "all" or l.main == main]
            if not sel:
                continue
            rows.append(dict(
                set=name, main=main, utts=len(sel), hours=round(sum(r["duration"] for r, _ in sel) / 3600, 2),
                mixed_utts=sum(l.mixed for _, l in sel), units=sum(len(l.units) for _, l in sel),
                embedded_units=sum(l.roles.count("E") for _, l in sel),
                Ms_units=sum(l.roles.count("Ms") for _, l in sel),
                critical_words=sum(len(l.critical) for _, l in sel),
                speakers=len({r["speaker"] for r, _ in sel})))
    df = pd.DataFrame(rows)
    os.makedirs(manifest.path("results"), exist_ok=True)
    df.to_csv(manifest.path("results/data_counts.csv"), index=False)
    with open(manifest.path("results/data_counts.md"), "w") as f:
        f.write("# Data counts (size check before training, proposal §3.1)\n\n")
        f.write("`main` = sentence main language from jieba word counts; `embedded_units` = E units.\n"
                "Detectability: at 20% error on 400 embedded units the 95% CI is about ±4 points.\n\n")
        f.write(df.to_markdown(index=False))
        f.write("\n")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
