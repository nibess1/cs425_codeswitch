#!/usr/bin/env python3
"""Second-run tables 13-24 (called from export_tables.py; writes into results/tables/).

13 speed (MacBook), 14 recovery Rec, 15 gated re-ranking, 16 speaker-cluster CIs, 17 synthetic
stress set, 18 partner-leakage sensitivity, 19 alignment checks, 20 label agreement,
21 Diff Room training ablation, 22 switching statistics, 23 tau sensitivity, 24 CWER rule vs labels.
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
from csasr.score.metrics import corpus_metrics  # noqa: E402

RES = manifest.path("results")
TEST = manifest.EVAL_TEST


def num(s):
    """'1.2 [0.3, 4.5]' -> (1.2, 0.3, 4.5); anything else -> (None, None, None)."""
    if not isinstance(s, str) or "[" not in s:
        try:
            return float(str(s).split(" ")[0]), None, None
        except ValueError:
            return None, None, None
    a, b = s.split(" [")
    lo, hi = b.split("]")[0].split(", ")
    return float(a), float(lo), float(hi)


def p(x):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(100 * float(x), 2)


def kappa(a, b):
    a, b = list(a), list(b)
    labs = sorted(set(a) | set(b))
    po = np.mean([x == y for x, y in zip(a, b)])
    pe = sum((a.count(l) / len(a)) * (b.count(l) / len(b)) for l in labs)
    return round(float(po), 3), round(float((po - pe) / (1 - pe)), 3)


def run(out, doc, set_name, prec_name, method_name):
    W = lambda df, f: df.to_csv(os.path.join(out, f), index=False)  # noqa: E731

    # 13 speed
    sp = json.load(open(os.path.join(RES, "speed_macbook.json")))
    mode = {"plain": "ASR only", "lid": "ASR + LID head (separate module)",
            "rerank_ondevice": "ASR beam 4 + LID head + on-device re-ranking"}
    rows = [{"model": r["tag"].replace("qat_", "") + (" (QAT merged)" if r["tag"].startswith("qat_") else ""),
             "export_format": prec_name.get(r["format"], r["format"]), "pipeline": mode[r["mode"]],
             "cpu_threads": r["threads"], "audio_seconds": r["audio_s"], "processing_seconds": r["seconds"],
             "real_time_factor": r["rtf"], "peak_memory_RSS_MB": r["peak_rss_mb"], "model_file_MB": r["file_mb"],
             "n_clips": r["n_clips"]} for r in sp]
    W(pd.DataFrame(rows).sort_values(["model", "pipeline", "export_format"]), "13_speed_macbook.csv")
    doc("13_speed_macbook.csv", "CPU speed on the MacBook (plugged in, Low Power Mode off), 100 fixed test clips,"
        " each configuration in a fresh process after a warm-up run.",
        [("real_time_factor", "processing time / audio length (lower is faster; < 1 = faster than real time)"),
         ("peak_memory_RSS_MB", "peak resident memory of the process"),
         ("note", "the first-batch ft_tiny / ft_small / ft_base rows were re-measured (first batch kept in "
                  "speed_first_batch_*.json, not used)")])

    # 14 recovery
    rc = pd.read_csv(os.path.join(RES, "recovery.csv"))
    rows = []
    for r in rc.itertuples():
        l, llo, lhi = num(r.loss)
        v, vlo, vhi = num(r.Rec)
        rows.append({"fix": r.fix, "model": r.model, "quantized_precision": prec_name.get(r.precision, r.precision),
                     "test_set": set_name.get(r.set, r.set),
                     "sentence_main_language": {"zh": "Chinese-main", "en": "English-main"}[r.main],
                     "embedded_error_loss_from_quantization_points": l, "loss_ci95_low": llo, "loss_ci95_high": lhi,
                     "Rec_recovery": v, "Rec_ci95_low": vlo, "Rec_ci95_high": vhi,
                     "Rec_reportable": vlo is not None})
    W(pd.DataFrame(rows), "14_recovery_Rec.csv")
    doc("14_recovery_Rec.csv", "Rec = (Eq_emb - Eq,fix_emb) / (Eq_emb - E16_emb), code-switched utterances.",
        [("Rec", "1 = fix removes the whole quantization loss, 0 = no help, < 0 = worse"),
         ("Rec_reportable", "False when the quantization loss CI includes 0 (Rec is then not meaningful)"),
         ("fix", "mixed precision (bitsandbytes NF4, k blocks at 16-bit chosen on val), QLoRA QAT with 16-bit"
                 " adapters, or QAT merged then re-quantized (deployed rows use the same-backend 16-bit reference)")])

    # 15 gated re-ranking
    rows = []
    for f in sorted(glob.glob(os.path.join(RES, "gated_*_gatedv2.json"))):
        j = json.load(open(f))
        tag = os.path.basename(f)[len("gated_"):-len("_gatedv2.json")]
        model = tag.replace("fullown", "full")
        for var, th in j["chosen"].items():
            name = f"{tag}__fp16__{var}_gatedv2"
            ref = f"{model}__fp16__full"
            for s in TEST:
                row = {"system_id": name, "model": model, "lid_head": os.path.basename(os.path.dirname(j["lid_head"])),
                       "note": "gatedv2 = rerun keyed by (set, utt_id); the 20 Oct *_gated files had an ASCEND id collision",
                       "reranker": var.replace("_filt", "").replace("rerank_", ""),
                       "theta_ms_chosen_on_val": th,
                       "val_MER_by_theta": "; ".join(f"{k}:{100 * v:.2f}" for k, v in j["thetas"][var].items()),
                       "test_set": set_name.get(s, s), "share_utterances_reranked_pct": p(j["share_reranked"][var].get(s))}
                try:
                    c = R.compare(ref, name, s, metrics=("MER", "E_E", "CWER"))
                    t = R.table([name], [s]).iloc[0]
                except (KeyError, IndexError, FileNotFoundError):
                    continue
                row.update(MER_pct=p(t.MER), embedded_error_pct=p(t.E_E))
                for k, col in (("MER", "MER"), ("E_E", "embedded_error"), ("CWER", "critical_word_error")):
                    row[f"{col}_change_vs_greedy_points"] = p(c[k]["diff"])
                    row[f"{col}_change_ci95_low"] = p(c[k]["lo"])
                    row[f"{col}_change_ci95_high"] = p(c[k]["hi"])
                    row[f"{col}_change_significant"] = c[k]["sig"]
                rows.append(row)
    W(pd.DataFrame(rows), "15_gated_reranking.csv")
    doc("15_gated_reranking.csv", "C1/C2: re-ranking applied only when the LID head finds a minority-language run"
        " >= theta ms; theta in {0,200,400,800,inf} chosen on val (lowest MER, ties to larger).",
        [("change_vs_greedy", "paired difference vs the same model with greedy decoding (negative = better)"),
         ("lid_head", "lid_ft_base = shared head; lid_full_* = the model's own head (C2)")])

    # 16 speaker-cluster CIs for the main systems
    main = [n for n in ["small__fp16__offshelf_free", "ft_tiny__fp16__ft", "ft_base__fp16__ft", "ft_small__fp16__ft",
                        "full_base__fp16__full", "full_small__fp16__full"] if os.path.exists(manifest.path(f"preds/{n}.meta.json"))]
    t = R.table(main, ["nsc_test", "nsc_diff_test"], speaker_ci=True)
    rows = []
    for r in t.itertuples():
        row = {"system_id": r.system, "test_set": set_name.get(r.set, r.set), "n_utterances": r.n_utt,
               "n_speakers": r.n_speakers}
        for k, col in (("MER", "MER"), ("E_E", "embedded_error"), ("cs_E_emb", "embedded_error_cs_utts"),
                       ("CWER", "critical_word_error")):
            row[f"{col}_pct"] = p(getattr(r, k))
            row[f"{col}_utt_ci95_low"], row[f"{col}_utt_ci95_high"] = p(getattr(r, k + "_lo")), p(getattr(r, k + "_hi"))
            row[f"{col}_speaker_ci95_low"], row[f"{col}_speaker_ci95_high"] = p(getattr(r, k + "_spk_lo")), p(getattr(r, k + "_spk_hi"))
        rows.append(row)
    W(pd.DataFrame(rows), "16_speaker_cluster_ci.csv")
    doc("16_speaker_cluster_ci.csv", "A3: NSC CIs from resampling whole speakers vs resampling utterances.",
        [("speaker_ci95", "use these for NSC claims; utterance CIs are too narrow when speakers repeat")])

    # 17 synthetic stress set
    rows = []
    for f, mode_ in (("d5_stress_synthetic.csv", "(a) rule tagger"), ("d5_stress_synthetic_critD2.csv", "(b) adjudicated labels")):
        d = pd.read_csv(os.path.join(RES, f))
        for r in d.itertuples():
            tag, prec, method = r.system.split("__")
            rows.append({"system_id": r.system, "model": tag, "precision": prec_name.get(prec, prec),
                         "channel": {"stress_tts_clean": "clean TTS", "stress_tts_phone": "TTS + simulated phone channel"}[r.set],
                         "critical_word_definition": mode_, "n_utterances": r.n_utt, "n_embedded_units": int(r.N_E),
                         "n_critical_words": int(r.N_crit), "MER_pct": p(r.MER), "embedded_error_pct": p(r.E_E),
                         "embedded_ci95_low": p(r.E_E_lo), "embedded_ci95_high": p(r.E_E_hi),
                         "CWER_pct": p(r.CWER), "CWER_ci95_low": p(r.CWER_lo), "CWER_ci95_high": p(r.CWER_hi),
                         "translit_rate_pct": p(r.translit_rate), "data": "SYNTHETIC"})
    W(pd.DataFrame(rows), "17_stress_set_SYNTHETIC.csv")
    doc("17_stress_set_SYNTHETIC.csv", "D5: SYNTHETIC stress set - 120 sentences x 4 CosyVoice 2 zero-shot voices"
        " (FLEURS prompts, 2 en + 2 zh speakers). NOT human speech; the human recordings are still pending.",
        [("channel", "phone = 300-3400 Hz band-limit, 8 kHz round trip, synthetic RIR (RT60 0.3 s), 20 dB white noise"),
         ("CI", "utterance bootstrap; the 4 voices are not independent speakers")])

    # 18 leakage sensitivity
    s = pd.read_parquet(manifest.path("data/leak/signals.parquet"))
    flag = set(s[s.leak_label != "clean"].utt_id)
    rows = []
    for n in main:
        st = R.stats(n, use_labse=False)
        for setn in ("nsc_test", "nsc_diff_test"):
            ids, x = st[setn]
            a = corpus_metrics(x)
            b = corpus_metrics([v for u, v in zip(ids, x) if u not in flag])
            rows.append({"system_id": n, "test_set": set_name.get(setn, setn), "n_all": len(x),
                         "n_flagged_excluded": sum(u in flag for u in ids),
                         "MER_all_pct": p(a["MER"]), "MER_flagged_excluded_pct": p(b["MER"]),
                         "embedded_error_all_pct": p(a["E_E"]), "embedded_error_flagged_excluded_pct": p(b["E_E"])})
    lab = s[s.set != "nsc_val"].groupby("set").leak_label.value_counts(normalize=True).mul(100).round(2).unstack(fill_value=0)
    W(pd.DataFrame(rows), "18_partner_leakage_sensitivity.csv")
    lab.reset_index().rename(columns={"set": "test_set"}).to_csv(os.path.join(out, "18b_partner_leakage_rates.csv"), index=False)
    doc("18_partner_leakage_sensitivity.csv", "D3: MER with and without utterances flagged for audible partner speech.", [])
    doc("18b_partner_leakage_rates.csv", "D3: % of test utterances per leakage label (rule piloted on val, borderline"
        " test clips judged by annotator 1).", [])

    # 19 alignment
    a = json.load(open(os.path.join(RES, "align_checks.json")))
    rows = [dict(check="(a) MMS at known switch times, FLEURS splices", **a["a_fleurs_splices_known_switches"]),
            dict(check="(b) MMS vs Whisper cross-attention at switches", **a["b_mms_vs_whisper_xattn_at_switches"])]
    for st, v in a["c_mms_vs_textgrid_edges"].items():
        for k, vv in v.items():
            rows.append(dict(check=f"(c) {set_name.get(st, st)}: {k.replace('_', ' ')}", **vv))
    for h, v in a.get("lid_boundary_median_ms_for_reference", {}).items():
        for k, ms in v.items():
            rows.append(dict(check=f"LID head {h} boundary error ({k}), for reference", median_ms=ms))
    W(pd.DataFrame(rows), "19_alignment_checks.csv")
    doc("19_alignment_checks.csv", "D4: forced-alignment quality without human labels.",
        [("median_ms / mean_ms", "absolute timing difference"), ("within_*", "% of switches within 50/100 ms")])

    # 20 label agreement
    L = manifest.path("data/llm_labels")
    rows = []
    d1 = [pd.read_csv(f"{L}/d1_labels_annotator{i}.csv").set_index("pair_id").label for i in (1, 2)]
    d1a = pd.read_csv(f"{L}/d1_labels_adjudicated.csv").set_index("pair_id").label
    ix = d1[0].index
    for nm, fn in (("3-class", lambda x: x), ("translation vs rest", lambda x: x == "translation"),
                   ("transliteration vs rest", lambda x: x == "transliteration"), ("other vs rest", lambda x: x == "other")):
        po, k = kappa(d1[0][ix].map(fn), d1[1][ix].map(fn))
        rows.append(dict(task="D1 translation/transliteration (1,768 ASCEND pairs)", comparison=f"annotator 1 vs 2, {nm}",
                         n_items=len(ix), raw_agreement=po, cohen_kappa=k))
    d2 = [pd.read_csv(f"{L}/d2_labels_annotator{i}.csv").set_index("item_id").label for i in (1, 2)]
    cand = pd.read_csv(f"{L}/d2_candidates.csv").set_index("item_id")
    rule = cand.rule_tagger_critical.map(lambda b: "critical" if b else "not_critical")
    ix = d2[0].index
    for nm, x, y in (("annotator 1 vs 2", d2[0], d2[1]), ("annotator 1 vs rule tagger", d2[0], rule),
                     ("annotator 2 vs rule tagger", d2[1], rule)):
        po, k = kappa(x[ix], y[ix])
        rows.append(dict(task="D2 critical words (335 candidate types)", comparison=nm, n_items=len(ix),
                         raw_agreement=po, cohen_kappa=k))
    j = pd.read_csv(f"{L}/d3_test_judgements.csv")
    po, k = kappa(j.rule_label, j.judge)
    rows.append(dict(task="D3 partner leakage (158 borderline test clips)", comparison="signal rule vs annotator 1",
                     n_items=len(j), raw_agreement=po, cohen_kappa=k))
    W(pd.DataFrame(rows), "20_label_agreement.csv")
    counts = pd.concat([d1a.value_counts().rename("D1_adjudicated"),
                        pd.read_csv(f"{L}/d2_labels_adjudicated.csv").label.value_counts().rename("D2_adjudicated")], axis=1)
    counts.reset_index().rename(columns={"index": "label"}).to_csv(os.path.join(out, "20b_adjudicated_label_counts.csv"), index=False)
    doc("20_label_agreement.csv", "D1-D3: agreement of the automated annotators (LLM annotators were fresh agents"
        " that saw only the rubric and the items).", [("cohen_kappa", "chance-corrected agreement")])

    # 21 Diff Room ablation
    e = pd.read_csv(os.path.join(RES, "e_diffroom_ablation.csv"))
    cond = {"esame_base": "A: 40 h Same Room (= ft_base)", "ediff_base": "B: 40 h Diff Room",
            "emix_base": "C: 20 h Same + 20 h Diff Room"}
    st_name = dict(set_name, nsce_diff_test="NSC Diff Room held-out test (E split, sessions unseen)")
    rows = []
    for r in e.itertuples():
        tag = r.system.split("__")[0]
        row = {"training_data": cond[tag], "system_id": r.system, "test_set": st_name.get(r.set, r.set),
               "n_utterances": r.n_utt, "MER_pct": p(r.MER), "MER_ci95_low": p(r.MER_lo), "MER_ci95_high": p(r.MER_hi),
               "embedded_error_pct": p(r.E_E), "CWER_pct": p(r.CWER)}
        if tag != "esame_base":
            c = R.compare("esame_base__fp16__eabl", r.system, r.set, metrics=("MER", "E_E"),
                          speaker_ci=not r.set.startswith("ascend"))
            row.update(MER_change_vs_A_points=p(c["MER"]["diff"]), MER_change_ci95_low=p(c["MER"]["lo"]),
                       MER_change_ci95_high=p(c["MER"]["hi"]), MER_change_significant=c["MER"]["sig"],
                       MER_change_speaker_ci95_low=p(c["MER"].get("spk_lo")),
                       MER_change_speaker_ci95_high=p(c["MER"].get("spk_hi")))
        rows.append(row)
    W(pd.DataFrame(rows), "21_diff_room_training_ablation.csv")
    doc("21_diff_room_training_ablation.csv", "E: does training on Diff Room audio help on Diff Room speech?"
        " Same ft_base recipe, no tuning; negative change = better than A.", [])

    # 22 switching statistics, 23 tau sensitivity
    pd.read_csv(os.path.join(RES, "switch_stats.csv")).to_csv(os.path.join(out, "22_switching_statistics.csv"), index=False)
    doc("22_switching_statistics.csv", "C4: how often and how long speakers switch, per corpus.", [])
    pd.read_csv(os.path.join(RES, "d1_tau_sensitivity.csv")).to_csv(os.path.join(out, "23_tau_sensitivity.csv"), index=False)
    doc("23_tau_sensitivity.csv", "D1: translation detector (CC-CEDICT or LaBSE >= tau) vs adjudicated labels.",
        [("pair_*", "each output pair counted once"), ("occ_*", "weighted by occurrences across all systems"),
         ("flagged_occ_rel_to_tau0.65", "approximate multiplier on conf_rate relative to the chosen tau")])

    # 24 CWER: rule tagger vs adjudicated labels (ASCEND test)
    # the label list is read at import time, so mode (b) runs in a child process
    import subprocess
    tmp = os.path.join(RES, "_cwer_labels_tmp.csv")
    env = dict(os.environ, CSASR_CRITICAL_LABELS=manifest.path("data/llm_labels/d2_critical_words.txt"))
    subprocess.run([sys.executable, manifest.path("scripts/score.py"), *main, "--set", "ascend_test", "--csv", tmp],
                   env=env, check=True, stdout=subprocess.DEVNULL)
    tb = pd.read_csv(tmp).rename(columns={"system": "system"})
    os.remove(tmp)
    ta = R.table(main, ["ascend_test"])
    tb = tb.set_index("system").loc[ta.system].reset_index()
    rows = []
    for (x, y) in zip(ta.itertuples(), tb.itertuples()):
        rows.append({"system_id": x.system, "test_set": "ASCEND test",
                     "n_critical_rule": int(x.N_crit), "CWER_rule_pct": p(x.CWER), "CWER_rule_ci95": f"[{p(x.CWER_lo)}, {p(x.CWER_hi)}]",
                     "n_critical_labels": int(y.N_crit), "CWER_labels_pct": p(y.CWER), "CWER_labels_ci95": f"[{p(y.CWER_lo)}, {p(y.CWER_hi)}]"})
    W(pd.DataFrame(rows), "24_cwer_rule_vs_labels.csv")
    doc("24_cwer_rule_vs_labels.csv", "D2: CWER with (a) the rule tagger vs (b) the adjudicated critical-word list"
        " (249 types + every digit string).", [])
