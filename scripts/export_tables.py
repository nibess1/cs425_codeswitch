#!/usr/bin/env python3
"""Export every result as a clearly labelled, numeric CSV -> results/tables/ (for the team).

Percentages are numbers (e.g. 21.35 means 21.35 %); CI columns are 95 % paired-bootstrap
bounds. Systems that have not been run yet are simply absent.
  python scripts/export_tables.py
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from csasr.data import manifest  # noqa: E402
from csasr.score import report as R  # noqa: E402

OUT = manifest.path("results/tables")
TEST = manifest.EVAL_TEST
SET_NAME = {"ascend_test": "ASCEND test (Chinese-main mostly)", "nsc_test": "NSC Part 4 Same Room test",
            "nsc_diff_test": "NSC Part 4 Different Room test (Zoom, harder)", "ascend_val": "ASCEND val",
            "nsc_val": "NSC Same Room val"}
PREC_NAME = {"fp16": "16-bit (GPU, HF)", "int8_bnb": "8-bit LLM.int8 (GPU, simulated)",
             "nf4_bnb": "4-bit NF4 (GPU, simulated)", "nf4_qlora": "4-bit NF4 + QLoRA adapters (QAT)",
             "ct2_f32": "CTranslate2 float32 (CPU, reference)", "ct2_int8": "CTranslate2 int8 (CPU)",
             "ggml_f16": "whisper.cpp F16 (CPU, reference)", "ggml_q8_0": "whisper.cpp Q8_0 (CPU)",
             "ggml_q5_0": "whisper.cpp Q5_0 (CPU)", "ggml_q4_0": "whisper.cpp Q4_0 (CPU)"}
METHOD_NAME = {"offshelf_free": "off-the-shelf, model predicts language token",
               "offshelf_zh": "off-the-shelf, forced <|zh|>", "offshelf_en": "off-the-shelf, forced <|en|>",
               "offshelf_both": "off-the-shelf, forced <|zh|><|en|>", "ft": "plain fine-tuning",
               "rw": "fine-tuning + role-weighted loss", "semonly": "fine-tuning + semantic loss only",
               "full": "fine-tuning + role-weighted + semantic loss", "sub": "best single method (role-weighted)",
               "rerank_lp": "N-best pooled, Whisper score only", "rerank_mms": "re-ranked with MMS + LID (upper bound)",
               "rerank_ondevice": "re-ranked with LID, on-device variant"}
METRICS = [("MER", "MER_pct", "mixed error rate over all units"),
           ("E_E", "embedded_unit_error_pct", "error on embedded units (E)"),
           ("E_Ms", "matrix_at_switch_error_pct", "error on matrix units next to a switch (Ms)"),
           ("E_Mi", "matrix_other_error_pct", "error on other matrix units (Mi)"),
           ("E_sw", "switch_point_error_pct", "error on all units next to a language change"),
           ("cs_E_mat", "matrix_unit_error_codeswitched_pct", "matrix (Ms+Mi) error on code-switched utterances"),
           ("conf_rate", "translation_rate_pct", "share of embedded units replaced by a translation"),
           ("sMER", "semantic_MER_pct", "MER with partial credit for correct translations (rho0=0.5)"),
           ("CWER", "critical_word_error_pct", "share of critical words not reproduced exactly")]

README = []


def doc(fname, what, cols):
    README.append(f"{fname}\n  {what}\n" + "".join(f"    {c}: {d}\n" for c, d in cols) + "\n")


def pct(x):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(100 * float(x), 2)


def split_name(name):
    tag, prec, method = name.split("__")
    size = tag.split("_")[-1]
    return tag, size, prec, method


def have(name):
    return os.path.exists(manifest.path(f"preds/{name}.meta.json"))


def systems(prec_filter=None):
    out = []
    for p in sorted(glob.glob(manifest.path("preds/*.meta.json"))):
        name = os.path.basename(p)[:-10]
        tag, size, prec, method = split_name(name)
        if method.endswith("_val") or method.startswith(("val", "smoke")):
            continue
        if prec_filter and prec not in prec_filter:
            continue
        out.append(name)
    return out


def accuracy_table(names, fname, what):
    df = R.table(names, TEST)
    rows = []
    for r in df.itertuples():
        tag, size, prec, method = split_name(r.system)
        row = {"system_id": r.system, "model_size": size, "training_or_decoding": METHOD_NAME.get(method, method),
               "precision": PREC_NAME.get(prec, prec), "test_set": SET_NAME.get(r.set, r.set),
               "n_utterances": r.n_utt, "n_reference_units": int(r.N), "n_embedded_units": int(r.N_E),
               "n_critical_words": int(r.N_crit)}
        for k, col, _ in METRICS:
            row[col] = pct(getattr(r, k))
            row[col.replace("_pct", "_ci95_low_pct")] = pct(getattr(r, k + "_lo"))
            row[col.replace("_pct", "_ci95_high_pct")] = pct(getattr(r, k + "_hi"))
        rows.append(row)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, fname), index=False)
    doc(fname, what, [("system_id", "<model>__<precision>__<method> (name of the prediction file)"),
                      ("n_*", "counts in the reference transcripts of that test set")]
        + [(c, d) for _, c, d in METRICS]
        + [("*_ci95_low_pct / *_ci95_high_pct", "95% bootstrap confidence interval (1000 utterance resamples)")])


def main():
    os.makedirs(OUT, exist_ok=True)

    # 1. data
    dc = pd.read_csv(manifest.path("results/data_counts.csv")).rename(columns={
        "set": "dataset_split", "main": "sentence_main_language", "utts": "n_utterances", "hours": "hours",
        "mixed_utts": "n_code_switched_utterances", "units": "n_units", "embedded_units": "n_embedded_units",
        "Ms_units": "n_matrix_at_switch_units", "critical_words": "n_critical_words", "speakers": "n_speakers"})
    dc.to_csv(os.path.join(OUT, "01_data_counts.csv"), index=False)
    doc("01_data_counts.csv", "Size of every split (proposal 3.1 size check).",
        [("sentence_main_language", "all / zh / en / balanced (tie in word counts)")])

    # 2. 16-bit accuracy of every system
    accuracy_table(systems({"fp16"}), "02_accuracy_16bit_all_systems.csv",
                   "Test accuracy of every system at 16-bit (greedy decoding unless re-ranked).")

    # 3. method vs plain fine-tuning (paired)
    rows = []
    for name in systems({"fp16"}):
        tag, size, prec, method = split_name(name)
        if tag == f"ft_{size}" or method.startswith("offshelf") or size not in ("tiny", "base", "small"):
            continue
        ref = f"ft_{size}__fp16__ft"
        if not have(ref):
            continue
        for s in TEST:
            try:
                c = R.compare(ref, name, s, metrics=("MER", "E_E", "conf_rate", "sMER", "CWER"))
            except KeyError:
                continue
            row = {"system_id": name, "model_size": size, "method": METHOD_NAME.get(method, method),
                   "compared_with": "plain fine-tuning, same size", "test_set": SET_NAME[s]}
            for k, col in [("MER", "MER"), ("E_E", "embedded_unit_error"), ("conf_rate", "translation_rate"),
                           ("sMER", "semantic_MER"), ("CWER", "critical_word_error")]:
                d = c[k]
                row[f"{col}_change_points"] = pct(d["diff"])
                row[f"{col}_change_ci95_low"] = pct(d["lo"])
                row[f"{col}_change_ci95_high"] = pct(d["hi"])
                row[f"{col}_change_significant"] = d["sig"]
            rows.append(row)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "03_methods_vs_plain_finetuning.csv"), index=False)
    doc("03_methods_vs_plain_finetuning.csv",
        "Paired bootstrap difference (method minus plain fine-tuning) on the same utterances. Negative = better.",
        [("*_change_points", "difference in percentage points"),
         ("*_change_significant", "True if the 95% CI of the difference excludes 0")])

    # 4. grids on validation
    rows = []
    for g in ("rw", "sem"):
        p = manifest.path(f"results/select_{g}.json")
        if not os.path.exists(p):
            continue
        j = json.load(open(p))
        for run, v in j["val"].items():
            cfg = json.load(open(manifest.path(f"runs/{run}/config.json")))
            rows.append({"grid": "role-weighted loss" if g == "rw" else "semantic loss (on chosen role weights)",
                         "run": run, "alpha": cfg["alpha"], "beta": cfg["beta"], "lambda_sem": cfg["lambda_sem"],
                         "rho0": cfg["rho0"], "is_reference_for_grid": run == j["baseline"],
                         "chosen": run == j["chosen"], "eligible_under_rule": run in j["eligible"],
                         "val_MER_pct": pct(v["MER"]), "val_embedded_unit_error_pct": pct(v["cs_E_emb"]),
                         "val_matrix_unit_error_pct": pct(v["cs_E_mat"]), "val_semantic_MER_pct": pct(v["sMER"]),
                         "val_critical_word_error_pct": pct(v["CWER"]), "val_translation_rate_pct": pct(v["conf_rate"])})
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "04_hyperparameter_grids_validation.csv"), index=False)
    doc("04_hyperparameter_grids_validation.csv", "Grid search on base, pooled ASCEND + NSC validation.",
        [("rule (role-weighted)", "lowest embedded error among runs with MER <= reference MER + 1 point"),
         ("rule (semantic)", "lowest semantic MER among runs with critical-word error <= reference")])

    # 5. accuracy at every precision
    accuracy_table([n for n in systems() if not n.split("__")[2].startswith("rerank")],
                   "05_accuracy_all_precisions.csv",
                   "Test accuracy at every precision (CPU formats are on a fixed subset: all ASCEND test, "
                   "4,000 NSC test, 2,000 Diff Room utterances; see n_utterances).")

    # 6. fragility ratio
    fr = pd.read_csv(manifest.path("results/fragility.csv")) if os.path.exists(manifest.path("results/fragility.csv")) else pd.DataFrame()
    rows = []
    for r in fr.itertuples() if len(fr) else []:
        def num(s):
            if not isinstance(s, str) or "[" not in s:
                return (None, None, None)
            a, b = s.split(" [")
            lo, hi = b.rstrip("]").split(", ")
            return float(a), float(lo), float(hi)
        e, elo, ehi = num(r.d_emb)
        m, mlo, mhi = num(r.d_mat)
        rep = isinstance(r.R, str) and r.R[0] in "-0123456789"
        Rv = num(r.R) if rep else (None, None, None)
        rows.append({"model": r.model, "method": METHOD_NAME.get(r.method, r.method),
                     "quantized_precision": PREC_NAME.get(r.precision, r.precision),
                     "compared_with_16bit": PREC_NAME.get(getattr(r, "ref16", "fp16"), "fp16"),
                     "test_set": SET_NAME.get(r.set, r.set),
                     "sentence_main_language": {"zh": "Chinese-main (embedded English)", "en": "English-main (embedded Mandarin)"}[r.main],
                     "embedded_error_change_points": e, "embedded_change_ci95_low": elo, "embedded_change_ci95_high": ehi,
                     "matrix_error_change_points": m, "matrix_change_ci95_low": mlo, "matrix_change_ci95_high": mhi,
                     "R_fragility_ratio": Rv[0], "R_ci95_low": Rv[1], "R_ci95_high": Rv[2],
                     "R_reported": rep})
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "06_fragility_ratio_R.csv"), index=False)
    doc("06_fragility_ratio_R.csv",
        "R = (embedded error change) / (matrix error change) from 16-bit to quantized, code-switched utterances only.",
        [("R > 1", "quantization hurts embedded units more; < 1 less"),
         ("R_reported", "False when the matrix change's CI includes 0 (R undefined/unstable); use the two changes instead"),
         ("compared_with_16bit", "GPU rows vs HF 16-bit; CPU rows vs the same CPU backend at 16/32-bit")])

    # 7. sensitivity scan
    rows, mp = [], []
    for p in sorted(glob.glob(manifest.path("results/sensitivity_*.json"))):
        j = json.load(open(p))
        model = os.path.basename(p)[len("sensitivity_"):].rsplit("_", 2)[0]
        rank = {g: i + 1 for i, (g, _) in enumerate(sorted([x for x in j["kl"] if x[0] != "ALL"], key=lambda x: -x[1]))}
        for g, s in j["kl"]:
            rows.append({"model": model, "block_quantized_alone": g.replace("model.", ""), "KL_on_embedded_tokens": s,
                         "rank": rank.get(g)})
        for c in j.get("confirm_top5", []):
            rows.append({"model": model, "block_quantized_alone": c["block"].replace("model.", ""),
                         "KL_on_embedded_tokens": c["kl"], "rank": rank.get(c["block"]),
                         "val_embedded_error_pct_only_this_block_4bit": pct(c["val_cs_E_emb"]),
                         "val_MER_pct_only_this_block_4bit": pct(c["val_MER"])})
        if "val_fp16" in j:
            mp.append({"model": model, "configuration": "all 16-bit", "blocks_kept_16bit": 0,
                       "val_embedded_error_pct": pct(j["val_fp16"][0]), "val_MER_pct": pct(j["val_fp16"][1])})
            mp.append({"model": model, "configuration": "all linear layers NF4", "blocks_kept_16bit": 0,
                       "val_embedded_error_pct": pct(j["val_quant"][0]), "val_MER_pct": pct(j["val_quant"][1])})
        for m in j.get("mixed", []):
            mp.append({"model": model, "configuration": "NF4 with most sensitive blocks at 16-bit",
                       "blocks_kept_16bit": m["k"], "which_blocks": "; ".join(b.replace("model.", "") for b in m["keep16"]),
                       "val_embedded_error_pct": pct(m["val_cs_E_emb"]), "val_MER_pct": pct(m["val_MER"]),
                       "chosen": m["k"] == j.get("chosen", {}).get("k")})
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "07_sensitivity_scan_per_block.csv"), index=False)
    pd.DataFrame(mp).to_csv(os.path.join(OUT, "08_mixed_precision_validation.csv"), index=False)
    doc("07_sensitivity_scan_per_block.csv", "A6: KL divergence on embedded-word tokens when only one block is NF4.",
        [("rank", "1 = most sensitive"), ("val_*_only_this_block_4bit", "confirmation decode for the top 5")])
    doc("08_mixed_precision_validation.csv", "NF4 with the top-k most sensitive blocks kept at 16-bit (validation).",
        [("chosen", "k picked on validation embedded error, then run once on test")])

    # 9. LID head
    rows = []
    for p in sorted(glob.glob(manifest.path("ckpt/lid_*/val_metrics.json"))):
        for s, v in json.load(open(p)).items():
            rows.append({"head": p.split(os.sep)[-2], "val_set": SET_NAME.get(s, s),
                         "language_accuracy_on_speech_frames_pct": pct(v.get("speech_lang_acc")),
                         "frame_accuracy_3class_smoothed_pct": pct(v["frame_acc"]),
                         "frame_accuracy_3class_unsmoothed_pct": pct(v["frame_acc_raw"]),
                         "switch_boundary_error_median_ms": round(1000 * v["boundary_median_s"]) if v["boundary_median_s"] is not None else None,
                         "switches_within_100ms_pct": pct(v["boundary_within_100ms"]),
                         "switches_detected_pct": pct(v["switch_detect_rate"])})
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "09_language_id_head.csv"), index=False)
    doc("09_language_id_head.csv", "Frame-level language ID on the frozen fine-tuned base encoder (20 ms frames).",
        [("3class", "en / zh / non-speech"), ("switch_boundary_error", "distance from each true switch to the nearest predicted one")])

    # 10. efficiency (GPU memory footprint)
    rows = []
    for p in sorted(glob.glob(manifest.path("preds/*.meta.json"))):
        m = json.load(open(p))
        if "footprint_mb" in m:
            rows.append({"model": m["model"], "precision": PREC_NAME.get(m["precision"], m["precision"]),
                         "model_memory_footprint_MB": m["footprint_mb"],
                         "n_quantized_linear_layers": m.get("quantized_modules_n", 0)})
        elif "file_mb" in m:
            rows.append({"model": m["model"], "precision": PREC_NAME.get(m["precision"], m["precision"]),
                         "model_file_size_MB": m["file_mb"]})
    pd.DataFrame(rows).drop_duplicates().to_csv(os.path.join(OUT, "10_model_size_and_memory.csv"), index=False)
    doc("10_model_size_and_memory.csv", "Measured GPU memory footprint (bitsandbytes) and CPU model file sizes.", [])

    # 11. RQ1
    p = manifest.path("results/rq1.json")
    if os.path.exists(p):
        sw, reg = [], []
        for sysname, r in json.load(open(p)).items():
            for lang, v in r["same_word"].items():
                if v:
                    sw.append({"system_id": sysname, "word_language": {"en": "English words", "zh": "Chinese characters"}[lang],
                               "n_word_types_seen_in_both_roles": v["n_types"], "n_embedded_tokens": v["n_emb"],
                               "n_matrix_tokens": v["n_mat"], "error_when_embedded_pct": pct(v["err_emb"]),
                               "error_when_matrix_pct": pct(v["err_mat"]), "difference_points": pct(v["diff"]),
                               "difference_ci95_low": pct(v["diff_ci"][0]), "difference_ci95_high": pct(v["diff_ci"][1])})
            g = r["regression"]
            for k, v in g["coef"].items():
                reg.append({"system_id": sysname, "term": k, "coefficient_log_odds": round(v, 4),
                            "ci95_low": round(g["ci"].get(k, [np.nan])[0], 4), "ci95_high": round(g["ci"].get(k, [np.nan, np.nan])[1], 4),
                            "n_units": g["n_units"], "n_speakers": g["n_speakers"]})
        pd.DataFrame(sw).to_csv(os.path.join(OUT, "11_rq1_same_word_embedded_vs_matrix.csv"), index=False)
        pd.DataFrame(reg).to_csv(os.path.join(OUT, "12_rq1_logistic_regression.csv"), index=False)
        doc("11_rq1_same_word_embedded_vs_matrix.csv", "Same word type: error when embedded vs when part of the main language.", [])
        doc("12_rq1_logistic_regression.csv", "A5: logit P(unit error) on role, language, frequency, length, corpus.",
            [("emb", "embedded-unit effect (b1) with language etc. held fixed"), ("en", "English unit"),
             ("emb_en", "interaction"), ("logf", "log training frequency"), ("dur", "spoken length (s)"),
             ("nsc", "NSC corpus"), ("ci95", "bootstrap over speakers")])

    with open(os.path.join(OUT, "README.txt"), "w") as f:
        f.write("CS425 results tables (exported " + pd.Timestamp.now().strftime("%Y-%m-%d %H:%M") + ")\n"
                "All *_pct columns are percentages (21.35 = 21.35%). Test sets unless the name says val.\n"
                "MER = (substitutions + deletions + insertions) / reference units; one unit = one Chinese\n"
                "character, English word or digit string. Role rates charge insertions to error blocks, so\n"
                "they can exceed 100%. Speed on the MacBook is in 13_speed_macbook.csv.\n\n" + "".join(README))
    print("wrote", sorted(os.listdir(OUT)))


if __name__ == "__main__":
    main()
