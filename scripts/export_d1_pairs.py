#!/usr/bin/env python3
"""D1: collect every error block (all systems, ASCEND test + val) where embedded units were
replaced by other-script text; deduplicate to unique (reference side, output side) pairs.
-> data/llm_labels/d1_pairs.csv (pair_id, ref_side, out_side, n_occurrences, n_systems,
   cedict_match, labse_sim). NSC is excluded until its licence is confirmed (PROCESS_LOG A4).
"""
import collections
import csv
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from csasr.data import manifest  # noqa: E402
from csasr.score.align import score  # noqa: E402
from csasr.score.confusion import LaBSE, cedict_match  # noqa: E402
from csasr.score.metrics import _side  # noqa: E402
from csasr.text.normalize import normalize  # noqa: E402
from csasr.text.units import label, split_units, unit_lang  # noqa: E402

SETS = ("ascend_test", "ascend_val")


def main():
    pairs = collections.Counter()
    systems = collections.defaultdict(set)
    for p in sorted(glob.glob(manifest.path("preds/*.jsonl"))):
        name = os.path.basename(p)[:-6]
        for l in open(p, encoding="utf-8"):
            r = json.loads(l)
            if r.get("set") not in SETS:
                continue
            lab = label(r["ref"])
            if lab.main not in ("zh", "en") or not lab.mixed:
                continue
            hu = split_units(normalize(r["hyp"]))
            hl = [unit_lang(u) for u in hu]
            emb = "en" if lab.main == "zh" else "zh"
            for b in score(lab.units, hu).blocks:
                if not b.ref_idx or not b.hyp_idx or any(lab.roles[i] != "E" for i in b.ref_idx):
                    continue
                if not any(hl[j] == lab.main for j in b.hyp_idx):  # other-script text in the output
                    continue
                key = (_side(lab.units, b.ref_idx, emb), _side(hu, b.hyp_idx, lab.main if all(
                    hl[j] == lab.main for j in b.hyp_idx) else "en"))
                pairs[key] += 1
                systems[key].add(name)
    keys = sorted(pairs, key=lambda k: -pairs[k])
    sims = LaBSE(device=os.environ.get("CSASR_LABSE_DEVICE", "cpu")).sims(keys)
    os.makedirs(manifest.path("data/llm_labels"), exist_ok=True)
    with open(manifest.path("data/llm_labels/d1_pairs.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["pair_id", "ref_side", "out_side", "n_occurrences", "n_systems", "cedict_match", "labse_sim"])
        for i, (k, s) in enumerate(zip(keys, sims)):
            zh, en = (k[1], k[0]) if all(not c.isascii() for c in k[1].replace(" ", "")) else (k[0], k[1])
            w.writerow([f"d1_{i:05d}", k[0], k[1], pairs[k], len(systems[k]), cedict_match(zh, en), round(s, 3)])
    print(len(keys), "unique pairs;", sum(pairs.values()), "occurrences")


if __name__ == "__main__":
    main()
