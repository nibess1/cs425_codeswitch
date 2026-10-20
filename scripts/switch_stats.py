#!/usr/bin/env python3
"""C4: code-switching statistics per corpus -> results/switch_stats.csv.

ASCEND is spontaneous conversation; NSC Part 4 speakers were asked to switch. Statistics are
over the reference transcripts of all splits used in this project (train + val + test).
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from csasr.data import manifest  # noqa: E402
from csasr.text.units import label  # noqa: E402

CORPORA = {"ASCEND (spontaneous)": ["ascend_train", "ascend_val", "ascend_test"],
           "NSC Part 4 Same Room (asked to switch)": ["nsc_train", "nsc_val", "nsc_test"],
           "NSC Part 4 Diff Room (asked to switch)": ["nsc_diff_test"]}


def stats(rows):
    sw, spans, mains, mixed, units = [], [], [], 0, []
    for r in rows:
        lab = label(r["ref"], already_normalized=True)
        langs = [l for l in lab.langs if l != "neu"]
        units.append(len(lab.units))
        mains.append(lab.main)
        if not lab.mixed:
            sw.append(0)
            continue
        mixed += 1
        sw.append(sum(a != b for a, b in zip(langs, langs[1:])))
        cur = 0
        for role in lab.roles + ["end"]:
            if role == "E":
                cur += 1
            elif cur:
                spans.append(cur); cur = 0
    sw = np.array(sw)
    cs = sw[sw > 0]
    m = pd.Series(mains).value_counts(normalize=True)
    return {"n_utterances": len(rows), "code_switched_utterances_pct": round(100 * mixed / len(rows), 1),
            "switches_per_utterance_all": round(float(sw.mean()), 3),
            "switches_per_code_switched_utterance_mean": round(float(cs.mean()), 2) if len(cs) else None,
            "switches_per_code_switched_utterance_median": float(np.median(cs)) if len(cs) else None,
            "embedded_span_units_mean": round(float(np.mean(spans)), 2) if spans else None,
            "embedded_span_units_median": float(np.median(spans)) if spans else None,
            "embedded_spans_single_unit_pct": round(100 * float(np.mean(np.array(spans) == 1)), 1) if spans else None,
            "units_per_utterance_mean": round(float(np.mean(units)), 1),
            "main_chinese_pct": round(100 * m.get("zh", 0), 1), "main_english_pct": round(100 * m.get("en", 0), 1),
            "main_balanced_pct": round(100 * m.get("balanced", 0), 1)}


def main():
    out = []
    for name, sets in CORPORA.items():
        rows = [r for s in sets for r in manifest.load(s)]
        out.append({"corpus": name, "splits": "+".join(sets), **stats(rows)})
        for main_lang in ("zh", "en"):
            sub = [r for r in rows if r["main"] == main_lang]
            out.append({"corpus": f"{name}, {'Chinese' if main_lang == 'zh' else 'English'}-main only",
                        "splits": "+".join(sets), **stats(sub)})
    df = pd.DataFrame(out)
    df.to_csv(manifest.path("results/switch_stats.csv"), index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
