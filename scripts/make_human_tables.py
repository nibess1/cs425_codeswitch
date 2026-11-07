#!/usr/bin/env python3
"""Human stress set (7 Nov, tag results-human): tables for the team's phone recordings (new files only).

  python scripts/make_human_tables.py
Reads preds/*__*__*_human.jsonl (set stress_test; plan in docs/TESTING_PLAN.md) and the synthetic D5 scores.
Writes results/d5h_stress_human.csv, results/d5h_stress_human_critD2.csv (raw score rows, as for d5_*),
       results/d5h_per_speaker.csv, results/d5h_flagged_clips.csv,
       results/tables/27_stress_set_HUMAN.csv (absolute, every system and precision, all / zh-main / en-main),
       results/tables/27b_human_fragility_R.csv (R and Rec on the human set),
       results/tables/27c_human_vs_synthetic.csv (16-bit, human vs synthetic clean / phone).
CIs are utterance bootstraps; 3 readers, so they are optimistic (as for ASCEND). CWER (b) uses the adjudicated
D2 list, computed in a child process because the label list is read at import time.
"""
import argparse
import glob
import io
import os
import subprocess
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from csasr.data import manifest  # noqa: E402
from csasr.score import bootstrap as BS  # noqa: E402
from csasr.score import report as R  # noqa: E402
from csasr.text.normalize import normalize  # noqa: E402
from csasr.text.units import split_units  # noqa: E402

RES = manifest.path("results")
TAB = manifest.path("results/tables")
SET = "stress_test"
LABELS = "data/llm_labels/d2_critical_words.txt"
REF16 = {"int8_bnb": "fp16", "nf4_bnb": "fp16", "ct2_int8": "ct2_f32",
         "ggml_q8_0": "ggml_f16", "ggml_q4_0": "ggml_f16"}
SLICES = [("all", None), ("zh-main (embedded English)", "zh"), ("en-main (embedded Mandarin)", "en")]
FT16 = ["ft_tiny__fp16__ft_human", "ft_base__fp16__ft_human", "ft_small__fp16__ft_human",
        "full_base__fp16__full_human", "full_small__fp16__full_human"]
PREC_NAME = {"fp16": "16-bit (GPU, HF)", "int8_bnb": "8-bit LLM.int8 (GPU, simulated)",
             "nf4_bnb": "4-bit NF4 (GPU, simulated)", "nf4_qlora": "NF4 + QLoRA adapters (GPU)",
             "ct2_f32": "CTranslate2 float32 (CPU, reference)", "ct2_int8": "CTranslate2 int8 (CPU)",
             "ggml_f16": "whisper.cpp F16 (CPU, reference)", "ggml_q8_0": "whisper.cpp Q8_0 (CPU)",
             "ggml_q4_0": "whisper.cpp Q4_0 (CPU)"}


def p(x):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(100 * float(x), 2)


def systems():
    return sorted(os.path.basename(f)[:-10] for f in glob.glob(manifest.path("preds/*_human.meta.json")))


def score_rows(names):
    """Raw score rows (all + by main language), same columns as results/d5_stress_synthetic.csv plus main."""
    out = []
    for main_name, main in SLICES:
        t = R.table(names, [SET], main=main)
        t.insert(2, "main", main_name)
        out.append(t)
    return pd.concat(out, ignore_index=True)


def per_speaker(names):
    rows = []
    for n in names:
        ids, st = R.stats(n)[SET]
        spk = R.speakers_of(SET)
        for s in sorted(set(spk.values())):
            for main_name, main in SLICES:
                sub = [x for u, x in zip(ids, st) if spk[u] == s and (main is None or x["main"] == main)]
                a = BS.to_arrays(sub)
                rows.append(dict(system=n, speaker=s, main=main_name, n_utt=len(sub),
                                 **{f"{k}_pct": p(BS.point(a, k)) for k in ("MER", "E_E", "E_Ms", "E_Mi", "E_sw", "CWER")},
                                 insertions=int(sum(x["I"] for x in sub))))
    return pd.DataFrame(rows)


def flagged(names):
    """Clips where every fine-tuned 16-bit system writes the same unit(s) instead of the same reference unit(s):
    possible reading slips (the reference is the prompt, not a transcript). For a person to listen to."""
    hyps = {n: {r["utt_id"]: r["hyp"] for r in R.read_preds(n)} for n in names}
    rows = []
    for r in manifest.load(SET):
        ref = split_units(r["ref"])
        hs = [split_units(normalize(hyps[n][r["utt_id"]])) for n in names]
        missing = set(ref).difference(*[set(h) for h in hs]) if hs else set()
        extra = set.intersection(*[set(h) for h in hs]) - set(ref)
        if missing and extra:
            rows.append(dict(utt_id=r["utt_id"], speaker=r["speaker"], ref=r["ref"],
                             ref_units_missing_in_all=" ".join(sorted(missing)),
                             units_all_systems_wrote=" ".join(sorted(extra)),
                             full_small_hyp=hyps.get("full_small__fp16__full_human", {}).get(r["utt_id"], "")))
    return pd.DataFrame(rows)


def fragility(names):
    rows = []
    have = set(names)
    for n in names:
        tag, prec, method = n.split("__")
        if prec not in REF16 or f"{tag}__{REF16[prec]}__{method}" not in have:
            continue
        ref = f"{tag}__{REF16[prec]}__{method}"
        for main_name, main in SLICES:
            fr = R.fragility(ref, n, SET, main=main)
            rows.append(dict(kind="R", system_id=n, model=tag, precision=PREC_NAME[prec], compared_with=ref,
                             main=main_name, d_embedded_points=p(fr["d_emb"]),
                             d_embedded_ci95_low=p(fr["d_emb_ci"][0]), d_embedded_ci95_high=p(fr["d_emb_ci"][1]),
                             d_matrix_points=p(fr["d_mat"]), d_matrix_ci95_low=p(fr["d_mat_ci"][0]),
                             d_matrix_ci95_high=p(fr["d_mat_ci"][1]), value=round(fr["R"], 2),
                             value_ci95_low=None if not fr["reportable"] else round(fr["R_ci"][0], 2),
                             value_ci95_high=None if not fr["reportable"] else round(fr["R_ci"][1], 2),
                             reportable=bool(fr["reportable"])))
    # recovery of the 4-bit embedded loss (same fixes as table 14)
    fixes = []
    for n in names:
        tag, prec, method = n.split("__")
        src = "full" if tag.startswith("full") else "ft"
        if method.startswith("mp") or prec == "nf4_qlora":
            fixes.append((f"{tag}__fp16__{src}_human", f"{tag}__nf4_bnb__{src}_human", n))
        elif method.endswith("_merged_human") and prec in REF16:
            fixes.append((f"{tag}__{REF16[prec]}__{src}_human", f"{tag}__{prec}__{src}_human", n))
    for b16, bq, fx in fixes:
        if b16 not in have or bq not in have:
            continue
        for main_name, main in SLICES:
            rc = R.recovery(b16, bq, fx, SET, main=main)
            ok = rc["Rec_ci"][0] is not None
            rows.append(dict(kind="Rec", system_id=fx, model=fx.split("__")[0], precision=PREC_NAME.get(fx.split("__")[1]),
                             compared_with=f"{b16} -> {bq}", main=main_name, d_embedded_points=p(rc["loss"]),
                             d_embedded_ci95_low=p(rc["loss_ci"][0]), d_embedded_ci95_high=p(rc["loss_ci"][1]),
                             value=round(rc["Rec"], 2), value_ci95_low=round(rc["Rec_ci"][0], 2) if ok else None,
                             value_ci95_high=round(rc["Rec_ci"][1], 2) if ok else None, reportable=ok))
    return pd.DataFrame(rows)


def absolute(raw, raw_b):
    rows = []
    for (_, r), (_, rb) in zip(raw.iterrows(), raw_b.iterrows()):
        assert (r.system, r.main) == (rb.system, rb.main)
        tag, prec, method = r.system.split("__")
        rows.append(dict(system_id=r.system, model=tag, precision=PREC_NAME.get(prec, prec), method=method,
                         main=r.main, n_utterances=r.n_utt, n_units=int(r.N), n_embedded_units=int(r.N_E),
                         MER_pct=p(r.MER), MER_ci95_low=p(r.MER_lo), MER_ci95_high=p(r.MER_hi),
                         embedded_error_pct=p(r.E_E), embedded_ci95_low=p(r.E_E_lo), embedded_ci95_high=p(r.E_E_hi),
                         matrix_at_switch_error_pct=p(r.E_Ms), matrix_other_error_pct=p(r.E_Mi),
                         switch_point_error_pct=p(r.E_sw), switch_point_ci95_low=p(r.E_sw_lo),
                         switch_point_ci95_high=p(r.E_sw_hi),
                         CWER_rule_pct=p(r.CWER), CWER_labels_pct=p(rb.CWER), translit_rate_pct=p(r.translit_rate),
                         translation_rate_pct=p(r.conf_rate), data="HUMAN (3 readers, phone)"))
    return pd.DataFrame(rows)


def vs_synthetic(raw):
    syn = pd.read_csv(os.path.join(RES, "d5_stress_synthetic.csv"))
    rows = []
    for n in [s for s in raw.system.unique() if "__fp16__" in s or "__ggml_" in s or "__ct2_" in s]:
        sn = n.replace("_human", "_stress")
        h = raw[(raw.system == n) & (raw.main == "all")].iloc[0]
        for lab, s in (("synthetic clean", "stress_tts_clean"), ("synthetic phone", "stress_tts_phone")):
            x = syn[(syn.system == sn) & (syn.set == s)]
            if len(x):
                r = x.iloc[0]
                rows.append(dict(system_id=n, audio=lab, n_utterances=r.n_utt, MER_pct=p(r.MER),
                                 embedded_error_pct=p(r.E_E), matrix_at_switch_error_pct=p(r.E_Ms),
                                 matrix_other_error_pct=p(r.E_Mi), switch_point_error_pct=p(r.E_sw),
                                 CWER_rule_pct=p(r.CWER), translit_rate_pct=p(r.translit_rate)))
        rows.append(dict(system_id=n, audio="human", n_utterances=h.n_utt, MER_pct=p(h.MER),
                         embedded_error_pct=p(h.E_E), matrix_at_switch_error_pct=p(h.E_Ms),
                         matrix_other_error_pct=p(h.E_Mi), switch_point_error_pct=p(h.E_sw),
                         CWER_rule_pct=p(h.CWER), translit_rate_pct=p(h.translit_rate)))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels_child", action="store_true")
    a = ap.parse_args()
    names = systems()
    if a.labels_child:  # CWER (b): adjudicated critical-word list
        print("CSV" + score_rows(names).to_csv(index=False).replace("\n", "\\n"))
        return
    assert names, "no *_human predictions"
    raw = score_rows(names)
    env = dict(os.environ, CSASR_CRITICAL_LABELS=manifest.path(LABELS))
    out = subprocess.run([sys.executable, os.path.abspath(__file__), "--labels_child"], env=env, check=True,
                         capture_output=True, text=True).stdout
    raw_b = pd.read_csv(io.StringIO([l for l in out.splitlines() if l.startswith("CSV")][-1][3:].replace("\\n", "\n")))
    raw.to_csv(os.path.join(RES, "d5h_stress_human.csv"), index=False)
    raw_b.to_csv(os.path.join(RES, "d5h_stress_human_critD2.csv"), index=False)
    per_speaker(names).to_csv(os.path.join(RES, "d5h_per_speaker.csv"), index=False)
    flagged([n for n in FT16 if n in names]).to_csv(os.path.join(RES, "d5h_flagged_clips.csv"), index=False)
    absolute(raw, raw_b).to_csv(os.path.join(TAB, "27_stress_set_HUMAN.csv"), index=False)
    fragility(names).to_csv(os.path.join(TAB, "27b_human_fragility_R.csv"), index=False)
    vs_synthetic(raw).to_csv(os.path.join(TAB, "27c_human_vs_synthetic.csv"), index=False)
    print("wrote", len(names), "systems")


if __name__ == "__main__":
    main()
