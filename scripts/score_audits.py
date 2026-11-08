#!/usr/bin/env python3
"""Human audits of the automated labels (8 Nov): agreement and Cohen's kappa with 95% bootstrap CIs.

  python scripts/score_audits.py
Reads review/translation_audit_human.csv and review/critical_audit_human.csv (filled by team members)
and the answer keys in review/keys/ (adjudicated LLM labels, annotators, rule tagger; not in git).
Writes results/tables/20c_human_audit_agreement.csv. Blank human labels are excluded and counted.
The critical-word items were sampled from rule-tagger / annotator disagreements, so they are harder
than a random sample.
"""
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from csasr.data import manifest  # noqa: E402

B, SEED = 1000, 0


def read(p):
    return list(csv.DictReader(open(manifest.path(p), encoding="utf-8-sig")))


def kappa(a, b):
    a, b = np.asarray(a), np.asarray(b)
    po = float((a == b).mean())
    cats = np.union1d(a, b)
    pe = float(sum((a == c).mean() * (b == c).mean() for c in cats))
    return po, (po - pe) / (1 - pe) if pe < 1 else float("nan")


def boot(a, b):
    a, b = np.asarray(a), np.asarray(b)
    idx = np.random.default_rng(SEED).integers(0, len(a), size=(B, len(a)))
    po, k = zip(*(kappa(a[i], b[i]) for i in idx))
    ci = lambda x: (round(float(np.nanpercentile(x, 2.5)), 3), round(float(np.nanpercentile(x, 97.5)), 3))
    return ci(po), ci(k)


def row(task, comparison, a, b, n_blank):
    po, k = kappa(a, b)
    if np.isnan(k):  # one class only on both sides: agreement is trivial, kappa undefined
        return dict(task=task, comparison=comparison, n_items=len(a), n_blank_excluded=n_blank,
                    raw_agreement=round(po, 3), note="kappa undefined: neither side used the positive class")
    (plo, phi), (klo, khi) = boot(a, b)
    return dict(task=task, comparison=comparison, n_items=len(a), n_blank_excluded=n_blank,
                raw_agreement=round(po, 3), agreement_ci95_low=plo, agreement_ci95_high=phi,
                cohen_kappa=round(k, 3), kappa_ci95_low=klo, kappa_ci95_high=khi, note="")


def main():
    rows = []
    # D1: translation / transliteration / other, human vs adjudicated LLM label
    key = {r["pair_id"]: r for r in read("review/keys/translation_audit_KEY.csv")}
    hum = read("review/translation_audit_human.csv")
    lab = [(r["your_label"].strip().lower(), key[r["pair_id"]]["automatic_label"]) for r in hum]
    filled = [(h, k) for h, k in lab if h]
    n_blank = len(lab) - len(filled)
    h, k = zip(*filled)
    t = "D1 translation audit (30 random ASCEND pairs)"
    rows.append(row(t, "human vs adjudicated LLM label, 3-class", h, k, n_blank))
    for c in ("translation", "transliteration"):
        rows.append(row(t, f"human vs adjudicated LLM label, {c} vs rest",
                        [x == c for x in h], [x == c for x in k], n_blank))
    # D2: critical / not critical, human vs each automated source
    key = {r["item_id"]: r for r in read("review/keys/critical_audit_KEY.csv")}
    hum = read("review/critical_audit_human.csv")
    filled = [r for r in hum if r["your_label"].strip()]
    n_blank = len(hum) - len(filled)
    h = [r["your_label"].strip().lower() for r in filled]
    t = "D2 critical-word audit (50 items sampled from tagger/annotator disagreements)"
    for src, name in (("adjudicated", "adjudicated LLM label"), ("annotator1", "LLM annotator 1"),
                      ("annotator2", "LLM annotator 2"), ("rule_tagger", "rule tagger")):
        rows.append(row(t, f"human vs {name}", h, [key[r["item_id"]][src] for r in filled], n_blank))
    out = manifest.path("results/tables/20c_human_audit_agreement.csv")
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        if r["note"]:
            print(f"{r['comparison']:55s} n={r['n_items']:2d} agree={r['raw_agreement']:.2f} ({r['note']})")
            continue
        print(f"{r['comparison']:55s} n={r['n_items']:2d} agree={r['raw_agreement']:.2f} "
              f"[{r['agreement_ci95_low']:.2f}, {r['agreement_ci95_high']:.2f}] "
              f"kappa={r['cohen_kappa']:.2f} [{r['kappa_ci95_low']:.2f}, {r['kappa_ci95_high']:.2f}]")


if __name__ == "__main__":
    main()
