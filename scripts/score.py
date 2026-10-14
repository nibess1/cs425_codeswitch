#!/usr/bin/env python3
"""Score prediction files. Every reported number comes from here.

  python scripts/score.py small__fp16__offshelf_free base__fp16__ft        # table
  python scripts/score.py --compare A B --set nsc_test                    # paired diff
  python scripts/score.py --fragility A16 Aq --set ascend_test --main zh  # R
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from csasr.score import report as R  # noqa: E402

COLS = ["MER", "sMER", "CWER", "E_E", "E_Ms", "E_Mi", "E_sw", "conf_rate"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("systems", nargs="*")
    ap.add_argument("--sets", nargs="*", default=None)
    ap.add_argument("--set", default=None)
    ap.add_argument("--main", default=None, choices=[None, "zh", "en"])
    ap.add_argument("--compare", nargs=2, default=None)
    ap.add_argument("--fragility", nargs=2, default=None)
    ap.add_argument("--no_labse", action="store_true")
    ap.add_argument("--csv", default=None)
    a = ap.parse_args()
    kw = dict(use_labse=not a.no_labse)
    if a.compare:
        res = R.compare(*a.compare, a.set, main=a.main, **kw)
        for m, d in res.items():
            print(f"{m:10s} {R.fmt(d['diff'], d['lo'], d['hi'])}  {'*' if d['sig'] else ''}")
        return
    if a.fragility:
        f = R.fragility(*a.fragility, a.set, main=a.main, **kw)
        print(f)
        return
    df = R.table(a.systems, a.sets, main=a.main, **kw)
    if a.csv:
        df.to_csv(a.csv, index=False)
    for _, r in df.iterrows():
        cells = [f"{c}={R.fmt(r[c], r[c + '_lo'], r[c + '_hi'])}" for c in COLS]
        print(f"{r.system} | {r.set} | n={r.n_utt} N={int(r.N)} N_E={int(r.N_E)} | " + "  ".join(cells))


if __name__ == "__main__":
    main()
