#!/usr/bin/env python3
"""Export the files the team needs for the PENDING_HUMAN checks -> review/*.csv

  python scripts/export_for_review.py --system ft_base__fp16__ft
  review/critical_words.csv   every auto-tagged critical word in the test sets (item 2)
  review/translation_blocks.csv  100 candidate language-confusion blocks with LaBSE sim (item 3)
  review/alignments.csv       unit spans of 30 random aligned test utterances (item 5)
  review/partner_leak.csv     20 random NSC test clips to listen to (item 4)
"""
import argparse
import csv
import os
import pickle
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from csasr.data import manifest  # noqa: E402
from csasr.score import report as R  # noqa: E402
from csasr.score.confusion import LaBSE  # noqa: E402
from csasr.score.metrics import _side, _structural_confusion  # noqa: E402
from csasr.score.align import score  # noqa: E402
from csasr.text.normalize import normalize  # noqa: E402
from csasr.text.units import label, split_units, unit_lang  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", default="ft_base__fp16__ft")
    a = ap.parse_args()
    out = manifest.path("review")
    os.makedirs(out, exist_ok=True)
    rng = random.Random(425)
    with open(os.path.join(out, "critical_words.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["set", "utt_id", "reference", "critical_word", "keep(1/0)"])
        for s in manifest.EVAL_TEST:
            for r in manifest.load(s):
                lab = label(r["ref"], already_normalized=True)
                for st, en in lab.critical:
                    w.writerow([s, r["utt_id"], r["ref"], _side(lab.units, range(st, en), lab.langs[st]), ""])
    cands = []
    for r in R.read_preds(a.system):
        lab = label(r["ref"])
        hu = split_units(normalize(r["hyp"]))
        hl = [unit_lang(u) for u in hu]
        for b in score(lab.units, hu).blocks:
            if _structural_confusion(b, lab, hl):
                emb = "en" if lab.main == "zh" else "zh"
                cands.append((r["utt_id"], r["ref"], r["hyp"], _side(lab.units, b.ref_idx, emb), _side(hu, b.hyp_idx, lab.main)))
    rng.shuffle(cands)
    cands = cands[:100]
    sims = LaBSE().sims([(c[3], c[4]) for c in cands]) if cands else []
    with open(os.path.join(out, "translation_blocks.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["utt_id", "reference", "output", "ref_side", "out_side", "labse_sim", "is_translation(1/0)"])
        for c, sm in zip(cands, sims):
            w.writerow([*c, round(sm, 3), ""])
    al = pickle.load(open(manifest.path("data/align/nsc_test.pkl"), "rb")) if os.path.exists(
        manifest.path("data/align/nsc_test.pkl")) else {}
    al_a = pickle.load(open(manifest.path("data/align/ascend_test.pkl"), "rb")) if os.path.exists(
        manifest.path("data/align/ascend_test.pkl")) else {}
    rows = {r["utt_id"]: r for s in ("nsc_test", "ascend_test") for r in manifest.load(s)}
    pick = rng.sample(sorted(set(al) & set(rows)), min(15, len(al))) + rng.sample(sorted(set(al_a) & set(rows)), min(15, len(al_a)))
    with open(os.path.join(out, "alignments.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["utt_id", "audio", "unit", "start_s", "end_s", "true_start_s", "true_end_s"])
        for u in pick:
            v = (al.get(u) or al_a.get(u))
            for unit, (s0, e0, _) in zip(v["units"], v["spans"]):
                w.writerow([u, rows[u]["audio"], unit, s0, e0, "", ""])
    test = manifest.load("nsc_test")
    with open(os.path.join(out, "partner_leak.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["utt_id", "audio", "reference", "partner_speech_audible(1/0)"])
        for r in rng.sample(test, 20):
            w.writerow([r["utt_id"], r["audio"], r["ref_raw"], ""])
    print("wrote", os.listdir(out))


if __name__ == "__main__":
    main()
