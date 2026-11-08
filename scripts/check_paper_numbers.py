#!/usr/bin/env python3
"""Number check (final checklist): each decimal number in a paper section should appear in the result
tables that section draws on (SOURCES below) or in results/KEY_NUMBERS.md.

  python scripts/check_paper_numbers.py <paper_dir> <results_dir>
Values are matched at 1-3 decimals, as stored and x100 (some tables store fractions). Checking each section
only against its own tables keeps chance matches rare (the script prints the chance-match rate of random
decoys). Expected misses: hyper-parameters, thresholds, hardware facts and numbers derived in the text
(differences, ranges); review the printed list by hand.
"""
import argparse
import csv
import glob
import os
import random
import re

NUM = re.compile(r"(?<![\w.])\d+\.\d+(?![\w.])")
SOURCES = {  # section file -> result tables (glob patterns relative to results_dir)
    "00_abstract": ["06_*", "27b_*", "14_*"],
    "01_introduction": ["02_*", "06_*", "13_*", "14_*", "27b_*", "figures/fig6_*"],
    "03_task_metrics": [],
    "04_data": ["01_*", "19_*", "20_*", "22_*", "23_*"],
    "05_methods": [],
    "06_setup": ["04_*"],
    "06b_running": [],
    "07_rq1": ["02_*", "11_*", "12_*", "23_*", "figures/fig2_*"],
    "08_rq2": ["02_*", "03_*", "09_*", "15_*", "25*", "26*"],
    "09_rq3": ["05_*", "06_*", "14_*", "figures/fig4_*"],
    "09b_human": ["27*", "d5h_*", "17_*"],
    "09c_examples_table": ["28_*"],
    "10_deployment": ["13_*", "figures/fig6_*"],
    "11_conclusion": [],
    "12_limitations": ["20*", "22_*"],
    "13_ethics": [],
    "appendix": ["04_*", "09_*", "13_*", "14_*", "17_*", "18*", "19_*", "20*", "21_*", "23_*", "25*", "26*",
                 "06_*", "figures/fig6_*"],
}


def variants(v):
    return {f"{abs(x):.{d}f}" for x in (v, 100 * v) for d in (1, 2, 3)}


def values(paths):
    seen = set()
    for p in paths:
        if p.endswith(".csv"):
            for row in csv.reader(open(p, encoding="utf-8-sig")):
                for cell in row:
                    for m in re.findall(r"-?\d+\.?\d*(?:e-?\d+)?", cell):
                        try:
                            seen |= variants(float(m))
                        except ValueError:
                            pass
        else:
            seen |= set(NUM.findall(open(p, encoding="utf-8").read()))
    return seen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paper_dir")
    ap.add_argument("results_dir")
    a = ap.parse_args()
    key = next(p for p in (os.path.join(a.results_dir, "KEY_NUMBERS.md"),
                           os.path.join(os.path.dirname(a.results_dir.rstrip("/")), "KEY_NUMBERS.md")) if os.path.exists(p))
    n = miss = 0
    rng = random.Random(0)
    for p in sorted(glob.glob(os.path.join(a.paper_dir, "sections", "*.tex"))):
        sec = os.path.basename(p)[:-4]
        srcs = sorted({f for g in SOURCES.get(sec, []) for d in (a.results_dir, os.path.dirname(a.results_dir.rstrip("/")))
                       for f in glob.glob(os.path.join(d, g + ".csv"))})  # results/tables and results/figures
        seen = values(srcs + [key])
        decoy = sum(f"{rng.uniform(0, 100):.1f}" in seen for _ in range(200)) / 2
        lines = [(i, m) for i, line in enumerate(open(p, encoding="utf-8"), 1) if not line.lstrip().startswith("%")
                 for m in NUM.findall(line)]
        bad = [(i, m) for i, m in lines if m not in seen]
        n, miss = n + len(lines), miss + len(bad)
        if lines:
            print(f"== {sec}: {len(lines)} numbers, {len(bad)} not found (chance match {decoy:.0f}%)")
        for i, m in bad:
            print(f"   line {i}: {m}")
    print(f"{n} numbers checked, {miss} not found")


if __name__ == "__main__":
    main()
