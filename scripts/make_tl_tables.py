#!/usr/bin/env python3
"""Third run (26 Oct): tables for the transliteration-penalised MRT (new files only).

  python scripts/make_tl_tables.py            # all fulltl_* test preds vs the same-size full system
Writes results/tables/25_transliteration_loss.csv (paired diffs, test),
       results/tables/25b_transliteration_loss_val.csv (val numbers behind the selection),
       results/tables/25c_transliteration_loss_absolute.csv (16-bit test values).
Paired diffs are new minus baseline (negative = better). NSC rows add speaker-cluster CIs; ASCEND
has 2 test speakers, so its utterance CIs are optimistic. CWER (b) uses the adjudicated D2 list,
computed in a child process because the label list is read at import time.
"""
import argparse
import glob
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from csasr.data import manifest  # noqa: E402
from csasr.score import report as R  # noqa: E402

TAB = manifest.path("results/tables")
SETS = list(manifest.EVAL_TEST)
METRICS = [("MER", "MER"), ("E_E", "embedded_error"), ("cs_E_emb", "embedded_error_cs_utts"),
           ("CWER", "critical_word_error_rule"), ("translit_rate", "transliteration_rate"),
           ("conf_rate", "translation_rate")]
LABELS = "data/llm_labels/d2_critical_words.txt"


def p(x):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(100 * float(x), 2)


def pairs_default():
    out = []
    for f in sorted(glob.glob(manifest.path("preds/fulltl_*__fp16__fulltl.meta.json"))):
        run = os.path.basename(f).split("__")[0]
        base = "full_small__fp16__full" if "_small_" in run else "full_base__fp16__full"
        out.append((base, f"{run}__fp16__fulltl"))
    return out


def cwer_labels(pairs, sets):
    """Child-process mode: paired CWER with the adjudicated list; prints JSON."""
    res = {}
    for a, b in pairs:
        for s in sets:
            d = R.compare(a, b, s, metrics=("CWER",), speaker_ci=True)["CWER"]
            res[f"{b}|{s}"] = {k: (float(v) if isinstance(v, (int, float, np.floating, np.bool_)) else v)
                               for k, v in d.items()}
    print("JSON" + json.dumps(res))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", nargs="*", default=None, help="baseline:new prediction names")
    ap.add_argument("--sets", nargs="*", default=SETS)
    ap.add_argument("--labels_child", action="store_true")
    ap.add_argument("--suffix", default="")
    a = ap.parse_args()
    pairs = [tuple(x.split(":")) for x in a.pairs] if a.pairs else pairs_default()
    if a.labels_child:
        return cwer_labels(pairs, a.sets)
    assert pairs, "no fulltl test predictions"
    env = dict(os.environ, CSASR_CRITICAL_LABELS=manifest.path(LABELS))
    cmd = [sys.executable, os.path.abspath(__file__), "--labels_child", "--sets", *a.sets,
           "--pairs", *[f"{x}:{y}" for x, y in pairs]]
    out = subprocess.run(cmd, env=env, check=True, capture_output=True, text=True).stdout
    lab = json.loads([l for l in out.splitlines() if l.startswith("JSON")][-1][4:])

    rows = []
    for base, new in pairs:
        for s in a.sets:
            d = R.compare(base, new, s, metrics=tuple(m for m, _ in METRICS), speaker_ci=True)
            row = dict(system_id=new, baseline=base, set=manifest.LABELS.get(s, s) if hasattr(manifest, "LABELS") else s,
                       set_id=s, ci_note="optimistic (2 test speakers)" if s.startswith("ascend") else "")
            for m, col in METRICS:
                x = d[m]
                row[f"{col}_change_points"] = p(x["diff"])
                row[f"{col}_ci95_low"], row[f"{col}_ci95_high"] = p(x["lo"]), p(x["hi"])
                row[f"{col}_significant"] = bool(x["sig"])
                if "spk_lo" in x:
                    row[f"{col}_speaker_ci95_low"], row[f"{col}_speaker_ci95_high"] = p(x["spk_lo"]), p(x["spk_hi"])
            x = lab[f"{new}|{s}"]
            row["critical_word_error_labels_change_points"] = p(x["diff"])
            row["critical_word_error_labels_ci95_low"], row["critical_word_error_labels_ci95_high"] = p(x["lo"]), p(x["hi"])
            row["critical_word_error_labels_significant"] = bool(x["sig"])
            if "spk_lo" in x:
                row["critical_word_error_labels_speaker_ci95_low"] = p(x["spk_lo"])
                row["critical_word_error_labels_speaker_ci95_high"] = p(x["spk_hi"])
            rows.append(row)
    pd.DataFrame(rows).to_csv(os.path.join(TAB, f"25_transliteration_loss{a.suffix}.csv"), index=False)

    names = sorted({x for pr in pairs for x in pr})
    t = R.table(names, a.sets, speaker_ci=True)
    keep = ["system", "set", "n_utt", "N_E", "N_crit"] + [c for c in t.columns if any(
        c == m or c.startswith(m + "_") for m in ("MER", "E_E", "cs_E_emb", "CWER", "translit_rate", "conf_rate"))]
    t = t[keep].copy()
    for c in t.columns[5:]:
        t[c] = t[c].map(p)
    t.to_csv(os.path.join(TAB, f"25c_transliteration_loss_absolute{a.suffix}.csv"), index=False)

    v = manifest.path("results/tl_val.csv")
    if os.path.exists(v):
        dv = pd.read_csv(v)
        for c in dv.columns:
            if c.split("_")[0] in ("MER", "CWER", "E", "cs", "translit", "conf"):
                dv[c] = dv[c].map(p)
        dv.to_csv(os.path.join(TAB, f"25b_transliteration_loss_val{a.suffix}.csv"), index=False)
    print(pd.DataFrame(rows)[["system_id", "set_id", "MER_change_points", "MER_ci95_low", "MER_ci95_high",
                              "embedded_error_change_points", "transliteration_rate_change_points"]].to_string())


if __name__ == "__main__":
    main()
