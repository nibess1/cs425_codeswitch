#!/usr/bin/env python3
"""Qualitative examples (8 Nov): 16 human stress-set clips drawn at random, with three systems' outputs.

  python scripts/make_examples.py
Draws 8 Mandarin-main and 8 English-main clips from set stress_test (seed 425, not hand-picked) and writes
results/tables/28_qualitative_examples.csv: reference, off-the-shelf small (16-bit), full small (16-bit),
full small (NF4), and the embedded-unit error count of each system on that clip.
"""
import csv
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from csasr.data import manifest  # noqa: E402
from csasr.score import report as R  # noqa: E402

SET, SEED, PER_MAIN = "stress_test", 425, 8
SYSTEMS = [("offshelf_small_16bit", "small__fp16__offshelf_free_human"),
           ("full_small_16bit", "full_small__fp16__full_human"),
           ("full_small_nf4", "full_small__nf4_bnb__full_human")]


def main():
    rows = manifest.load(SET)
    rng = random.Random(SEED)
    pick = []
    for main_ in ("zh", "en"):
        pick += sorted(rng.sample([r for r in rows if r["main"] == main_], PER_MAIN), key=lambda r: r["utt_id"])
    hyps = {k: {r["utt_id"]: r["hyp"] for r in R.read_preds(n)} for k, n in SYSTEMS}
    errs = {}
    for k, n in SYSTEMS:
        ids, st = R.stats(n)[SET]
        errs[k] = {u: (int(x["K_E"]), int(x["N_E"])) for u, x in zip(ids, st)}
    out = manifest.path("results/tables/28_qualitative_examples.csv")
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["utt_id", "speaker", "main", "reference"]
                   + [c for k, _ in SYSTEMS for c in (k, f"{k}_embedded_errors")] + ["n_embedded_units"])
        for r in pick:
            u = r["utt_id"]
            w.writerow([u, r["speaker"], r["main"], r["ref_raw"]]
                       + [c for k, _ in SYSTEMS for c in (hyps[k][u], errs[k][u][0])] + [errs[SYSTEMS[0][0]][u][1]])
    print("wrote", out, len(pick), "clips")


if __name__ == "__main__":
    import argparse  # --help prints the docstring above and exits; nothing else is parsed here
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    main()
