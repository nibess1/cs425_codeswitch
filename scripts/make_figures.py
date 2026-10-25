#!/usr/bin/env python3
"""F2: presentation figures -> results/figures/figN_*.{png,svg,csv} (CSV = exactly the plotted data).

Reads only results/tables/*.csv and results/speed_macbook.json (run export_tables.py first).
  python scripts/make_figures.py
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from csasr.data import manifest  # noqa: E402

T = manifest.path("results/tables")
OUT = manifest.path("results/figures")
SETS = {"ASCEND test (Chinese-main mostly)": "ASCEND", "NSC Part 4 Same Room test": "NSC Same Room",
        "NSC Part 4 Different Room test (Zoom, harder)": "NSC Diff Room"}
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "svg.fonttype": "none"})


def tab(n):
    return pd.read_csv(os.path.join(T, n))


def save(fig, df, name):
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), dpi=200)
    df.to_csv(os.path.join(OUT, f"{name}.csv"), index=False)
    plt.close(fig)
    print("wrote", name)


def label(sid):
    tag, prec, method = sid.split("__")
    size = tag.split("_")[-1]
    kind = {"offshelf_free": "off-the-shelf", "ft": "fine-tuned", "full": "full system"}.get(method, method)
    return f"{size} {kind}"


def bars(ax, df, cats, groups, val, lo, hi, gcol, ccol):
    w = 0.8 / len(groups)
    for i, g in enumerate(groups):
        d = df[df[gcol] == g].set_index(ccol).reindex(cats)
        x = np.arange(len(cats)) + (i - (len(groups) - 1) / 2) * w
        ax.bar(x, d[val], w, label=g, yerr=[d[val] - d[lo], d[hi] - d[val]], capsize=2, error_kw={"lw": 0.8})
    ax.set_xticks(np.arange(len(cats)), cats)


def forest(ax, df, ylab, est, lo, hi, ref=0.0, xlabel=""):
    y = np.arange(len(df))[::-1]
    sig = (df[lo] > ref) | (df[hi] < ref)
    ax.errorbar(df[est], y, xerr=[df[est] - df[lo], df[hi] - df[est]], fmt="none", ecolor="grey", lw=1)
    ax.scatter(df[est][sig], y[sig.values], color="C3", zorder=3, s=18, label="CI excludes reference")
    ax.scatter(df[est][~sig], y[~sig.values], facecolor="white", edgecolor="k", zorder=3, s=18, label="n.s.")
    ax.axvline(ref, color="k", lw=0.8, ls="--")
    ax.set_yticks(y, df[ylab], fontsize=7)
    ax.set_xlabel(xlabel)


def main():
    os.makedirs(OUT, exist_ok=True)
    a16 = tab("02_accuracy_16bit_all_systems.csv")
    a16["set"] = a16.test_set.map(SETS)
    core = ["tiny__fp16__offshelf_free", "base__fp16__offshelf_free", "small__fp16__offshelf_free",
            "ft_tiny__fp16__ft", "ft_base__fp16__ft", "ft_small__fp16__ft", "full_base__fp16__full", "full_small__fp16__full"]
    d = a16[a16.system_id.isin(core)].copy()
    d["system"] = d.system_id.map(label)
    cats = [label(s) for s in core if s in set(d.system_id)]

    # 1 MER and embedded error
    cols = ["system", "system_id", "set", "MER_pct", "MER_ci95_low_pct", "MER_ci95_high_pct", "embedded_unit_error_pct",
            "embedded_unit_error_ci95_low_pct", "embedded_unit_error_ci95_high_pct"]
    fig, axs = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    bars(axs[0], d, cats, list(SETS.values()), "MER_pct", "MER_ci95_low_pct", "MER_ci95_high_pct", "set", "system")
    bars(axs[1], d, cats, list(SETS.values()), "embedded_unit_error_pct", "embedded_unit_error_ci95_low_pct",
         "embedded_unit_error_ci95_high_pct", "set", "system")
    axs[0].set_ylabel("MER (%)"); axs[1].set_ylabel("Embedded-unit error (%)")
    axs[0].legend(frameon=False); axs[1].tick_params(axis="x", rotation=30)
    axs[0].set_title("Off-the-shelf vs fine-tuned vs full system (16-bit, 95% CI)")
    save(fig, d[cols], "fig1_mer_and_embedded_error")

    # 2 role errors
    roles = [("embedded_unit_error", "E (embedded)"), ("matrix_at_switch_error", "Ms (matrix at switch)"),
             ("matrix_other_error", "Mi (other matrix)")]
    long = pd.concat([d.assign(role=r, val=d[f"{c}_pct"], lo=d[f"{c}_ci95_low_pct"], hi=d[f"{c}_ci95_high_pct"])
                      for c, r in roles])[["system", "system_id", "set", "role", "val", "lo", "hi"]]
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.5), sharey=False)
    for ax, s in zip(axs, SETS.values()):
        bars(ax, long[long.set == s], cats, [r for _, r in roles], "val", "lo", "hi", "role", "system")
        ax.set_title(s); ax.tick_params(axis="x", rotation=60); ax.set_ylabel("error (%)")
    axs[0].legend(frameon=False, fontsize=8)
    save(fig, long.rename(columns={"val": "error_pct", "lo": "ci95_low", "hi": "ci95_high"}), "fig2_role_errors")

    # 3 RQ2 forest
    m = tab("03_methods_vs_plain_finetuning.csv")
    m = m[~m.system_id.str.contains("rerank")].copy()
    m["row"] = m.system_id.str.split("__").str[0] + " | " + m.test_set.map(SETS)
    fig, axs = plt.subplots(1, 2, figsize=(12, 0.28 * len(m) + 1.5), sharey=True)
    forest(axs[0], m, "row", "MER_change_points", "MER_change_ci95_low", "MER_change_ci95_high", xlabel="MER change (points)")
    forest(axs[1], m, "row", "embedded_unit_error_change_points", "embedded_unit_error_change_ci95_low",
           "embedded_unit_error_change_ci95_high", xlabel="Embedded-unit error change (points)")
    axs[0].set_title("Methods vs plain fine-tuning (negative = better)"); axs[1].legend(frameon=False, fontsize=7)
    save(fig, m, "fig3_rq2_methods_forest")

    # 4 R forest
    r = tab("06_fragility_ratio_R.csv")
    r = r[r.R_reported].copy()
    r["row"] = (r.model + " " + r.quantized_precision.str.replace(r" \(.*\)", "", regex=True) + " | "
                + r.test_set.map(lambda s: SETS.get(s, s)) + " " + r.sentence_main_language.str.split(" ").str[0])
    fig, ax = plt.subplots(figsize=(8, 0.28 * len(r) + 1.5))
    forest(ax, r, "row", "R_fragility_ratio", "R_ci95_low", "R_ci95_high", ref=1.0,
           xlabel="R = embedded change / matrix change (1 = equally fragile)")
    ax.set_title("Fragility ratio R (only where the matrix change CI excludes 0)"); ax.legend(frameon=False, fontsize=7)
    save(fig, r, "fig4_fragility_R_forest")

    # 5 Rec
    rc = tab("14_recovery_Rec.csv")
    rc = rc[rc.Rec_reportable].copy()
    rc["row"] = (rc.model + " " + rc.fix.str.replace("mixed precision", "MP").str.replace("at 16-bit", "")
                 + " " + rc.quantized_precision.str.replace(r" \(.*\)", "", regex=True) + " | "
                 + rc.test_set.map(lambda s: SETS.get(s, s)))
    fig, ax = plt.subplots(figsize=(9, 0.3 * len(rc) + 1.5))
    if len(rc):
        forest(ax, rc, "row", "Rec_recovery", "Rec_ci95_low", "Rec_ci95_high", ref=0.0,
               xlabel="Rec (1 = full recovery of the 4-bit embedded-error loss, 0 = none)")
        ax.axvline(1, color="green", lw=0.8, ls=":")
    ax.set_title("Recovery by each quantization fix (slices where the loss CI excludes 0)")
    save(fig, rc, "fig5_recovery_Rec")

    # 6 accuracy vs speed (MacBook RTF vs MER on the CPU subset of NSC Same Room test)
    sp = pd.DataFrame(json.load(open(manifest.path("results/speed_macbook.json"))))
    sp = sp[sp["mode"] == "plain"]
    a = tab("05_accuracy_all_precisions.csv")
    a = a[a.test_set == "NSC Part 4 Same Room test"].set_index("system_id")
    def sid(tag, fmt):
        if tag.startswith("offshelf_"):
            return f"{tag.split('_')[1]}__{fmt}__offshelf_free"
        if tag.startswith("qat_"):
            t = tag[4:]
            return f"{t}__{fmt}__qat_{'full' if t.startswith('full') else 'ft'}_merged"
        return f"{tag}__{fmt}__{'full' if tag.startswith('full') else 'ft'}"
    sp["system_id"] = [sid(t, f) for t, f in zip(sp.tag, sp.format)]
    sp = sp[sp.system_id.isin(a.index)].copy()
    sp["MER_pct"] = sp.system_id.map(a.MER_pct)
    sp["embedded_unit_error_pct"] = sp.system_id.map(a.embedded_unit_error_pct)
    sp["plotted"] = ~sp.tag.str.startswith("offshelf")  # off-the-shelf MER (50-300%) is off this scale; kept in the CSV
    marks = {"ct2_int8": "s", "ggml_q8_0": "^", "ggml_q5_0": "v", "ggml_q4_0": "o", "ggml_f16": "D"}
    from matplotlib.lines import Line2D
    fig, ax = plt.subplots(figsize=(9, 6))
    q = sp[sp.plotted]
    tags = sorted(q.tag.unique())
    for i, tag in enumerate(tags):
        for fmt, mk in marks.items():
            g = q[(q.tag == tag) & (q.format == fmt)]
            ax.scatter(g.rtf, g.MER_pct, s=g.file_mb * 2, marker=mk, alpha=0.75, color=plt.cm.tab10(i % 10),
                       edgecolor="k", lw=0.4)
    h1 = [Line2D([], [], ls="", marker="o", color=plt.cm.tab10(i % 10), label=t.replace("qat_", "QAT "))
          for i, t in enumerate(tags)]
    h2 = [Line2D([], [], ls="", marker=m, color="grey",
                 label=f.replace("ggml_", "whisper.cpp ").replace("ct2_int8", "CTranslate2 int8")) for f, m in marks.items()]
    ax.add_artist(ax.legend(handles=h1, frameon=False, fontsize=7, loc="upper left", title="model", title_fontsize=8))
    ax.legend(handles=h2, frameon=False, fontsize=7, loc="upper right", title="format", title_fontsize=8)
    ax.set_xscale("log"); ax.set_xlabel("Real-time factor on MacBook, 4 threads (log; lower = faster)")
    ax.set_ylabel("MER (%) on NSC Same Room test (CPU subset)")
    ax.set_title("Accuracy vs speed, fine-tuned models (marker area = model file MB)")
    save(fig, sp[["system_id", "tag", "format", "rtf", "file_mb", "peak_rss_mb", "MER_pct", "embedded_unit_error_pct",
                  "plotted"]],
         "fig6_accuracy_vs_speed_macbook")

    # 7 re-ranking: greedy vs always vs gated
    rows = []
    for size in ("base", "small"):
        variants = [(f"full_{size}__fp16__full", "greedy"),
                    (f"full_{size}__fp16__rerank_ondevice_filt", "always re-rank (on-device)"),
                    (f"full_{size}__fp16__rerank_ondevice_filt_gated", "LID-gated (shared head)"),
                    (f"fullown_{size}__fp16__rerank_ondevice_filt_gated", "LID-gated (own head)"),
                    (f"full_{size}__fp16__rerank_mms_filt", "always re-rank (MMS)")]
        for s, v in variants:
            for x in a16[a16.system_id == s].itertuples():
                rows.append(dict(model=f"full_{size}", variant=v, system_id=s, set=x.set, MER_pct=x.MER_pct,
                                 lo=x.MER_ci95_low_pct, hi=x.MER_ci95_high_pct, embedded_unit_error_pct=x.embedded_unit_error_pct))
    rr = pd.DataFrame(rows)
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
    vs = list(dict.fromkeys(rr.variant))
    for ax, (mdl, g) in zip(axs, rr.groupby("model")):
        bars(ax, g, list(SETS.values()), vs, "MER_pct", "lo", "hi", "variant", "set")
        ax.set_title(mdl); ax.set_ylabel("MER (%)")
    axs[0].legend(frameon=False, fontsize=7)
    save(fig, rr.rename(columns={"lo": "MER_ci95_low", "hi": "MER_ci95_high"}), "fig7_reranking_gated")


if __name__ == "__main__":
    main()
