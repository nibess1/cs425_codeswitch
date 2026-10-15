#!/usr/bin/env python3
"""Build every results table from preds/ (via csasr.score.report) -> results/*.md, *.csv.

Safe to re-run at any time; systems that have not been run yet are skipped.
  python scripts/make_results.py
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
from csasr.score.flags import flag_pr  # noqa: E402

RES = manifest.path("results")
TEST = manifest.EVAL_TEST
SIZES = ["tiny", "base", "small"]
SIM_PREC = ["fp16", "int8_bnb", "nf4_bnb"]
DEP_PREC = ["ct2_f32", "ct2_int8", "ggml_f16", "ggml_q8_0", "ggml_q5_0", "ggml_q4_0"]
# 16-bit reference for R / Rec: the same decoder implementation, so only the weights differ
# (HF and CPU decoders differ in repetition behaviour on off-the-shelf models)
REF16 = {"int8_bnb": "fp16", "nf4_bnb": "fp16", "ct2_int8": "ct2_f32",
         "ggml_q8_0": "ggml_f16", "ggml_q5_0": "ggml_f16", "ggml_q4_0": "ggml_f16"}
# (set, main language) slices for R: Chinese-main (embedded English) and English-main (embedded Mandarin)
R_SLICES = [("ascend_test", "zh"), ("nsc_test", "en"), ("nsc_test", "zh"), ("nsc_diff_test", "en")]


def have(name):
    return os.path.exists(manifest.path(f"preds/{name}.jsonl")) and os.path.exists(manifest.path(f"preds/{name}.meta.json"))


def f(x, lo=None, hi=None):
    return R.fmt(x, lo, hi)


def md(df):
    return df.to_markdown(index=False) if len(df) else "_not run yet_"


def write(name, title, body):
    with open(os.path.join(RES, name), "w", encoding="utf-8") as fh:
        fh.write(f"# {title}\n\n{body}\n")


def sig(d):
    return f"{100 * d['diff']:+.1f} [{100 * d['lo']:+.1f}, {100 * d['hi']:+.1f}]{' *' if d['sig'] else ''}"


# ------------------------------------------------------------------ baselines + FT
def baselines():
    names = [f"{m}__fp16__offshelf_{l}" for m in SIZES for l in ["free", "zh", "en", "both"]]
    names += [f"ft_{m}__fp16__ft" for m in SIZES]
    names = [n for n in names if have(n)]
    if not names:
        return
    df = R.table(names, TEST)
    df.to_csv(os.path.join(RES, "baselines.csv"), index=False)
    out = []
    for s in TEST:
        d = df[df.set == s]
        rows = [dict(system=r.system.replace("__fp16__", " "), MER=f(r.MER, r.MER_lo, r.MER_hi),
                     E_emb=f(r.E_E, r.E_E_lo, r.E_E_hi), E_Ms=f(r.E_Ms), E_Mi=f(r.E_Mi), E_switch=f(r.E_sw),
                     conf_rate=f(r.conf_rate), sMER=f(r.sMER), CWER=f(r.CWER), N_emb=int(r.N_E)) for r in d.itertuples()]
        out.append(f"## {s}\n\n" + md(pd.DataFrame(rows)))
    write("baselines.md", "Off-the-shelf and plainly fine-tuned Whisper (16-bit, greedy, test sets)",
          "MER and role-level errors in % with 95% bootstrap CIs. E_emb / E_Ms / E_Mi: Pass-2 error of embedded,"
          " matrix-at-switch and other matrix units (insertions charged by error block, so rates can exceed 100)."
          " conf_rate: share of embedded units inside a translation (language-confusion) block.\n\n" + "\n\n".join(out))


# ------------------------------------------------------------------ RQ1
def rq1():
    p = os.path.join(RES, "rq1.json")
    if not os.path.exists(p):
        return
    res = json.load(open(p))
    parts = []
    for sysname, r in res.items():
        sw = r["same_word"]
        rows = []
        for lang, v in sw.items():
            if v:
                rows.append(dict(word_language=lang, word_types=v["n_types"], embedded_tokens=v["n_emb"],
                                 matrix_tokens=v["n_mat"], err_embedded=f(v["err_emb"]), err_matrix=f(v["err_mat"]),
                                 diff=f(v["diff"], *v["diff_ci"])))
        reg = r["regression"]
        coef = pd.DataFrame([dict(term=k, coef=round(v, 3), ci=f"[{reg['ci'].get(k, [np.nan])[0]:.3f}, "
                                                              f"{reg['ci'].get(k, [np.nan, np.nan])[1]:.3f}]")
                             for k, v in reg["coef"].items()])
        rt = pd.DataFrame(r["role_unit_error"])
        rt["mean"] = rt["mean"].map(lambda x: f"{100 * x:.1f}")
        parts.append(f"## {sysname}\n\n### Unit error (substitution/deletion) by role, code-switched utterances\n\n"
                     + md(rt.rename(columns={"mean": "err_%", "count": "units"}))
                     + "\n\n### Same word, embedded vs matrix\n\n" + md(pd.DataFrame(rows))
                     + f"\n\n### Logistic regression (A5), {reg['n_units']} units, {reg['n_speakers']} speakers,"
                       f" speaker-bootstrap CIs (B={reg['n_boot']}); covariates: {', '.join(reg['covariates'])}\n\n"
                     + md(coef))
    write("rq1.md", "RQ1: role or language?", "\n\n".join(parts))


# ------------------------------------------------------------------ methods on base
def methods():
    rows = []
    for log in sorted(glob.glob(manifest.path("runs/*/log.jsonl"))):
        run = log.split(os.sep)[-2]
        if not run.startswith(("ft_", "rw_", "sem_", "full_", "abl_")):
            continue
        ev = [json.loads(l) for l in open(log) if '"event": "eval"' in l]
        if not ev:
            continue
        best = min(ev, key=lambda e: e["mean_MER"])
        cfg = json.load(open(manifest.path(f"runs/{run}/config.json")))
        rows.append(dict(run=run, model=cfg["model"], alpha=cfg["alpha"], beta=cfg["beta"], lambda_sem=cfg["lambda_sem"],
                         rho0=cfg["rho0"], best_step=best["step"], val_mean_MER=f(best["mean_MER"]),
                         **{f"{s}_MER": f(best[s]["MER"]) for s in manifest.EVAL_VAL if s in best},
                         **{f"{s}_E_emb": f(best[s]["E_E"]) for s in manifest.EVAL_VAL if s in best}))
    val = pd.DataFrame(rows)
    # test comparisons against plain fine-tuning of the same size
    comps = []
    for p in sorted(glob.glob(manifest.path("preds/*__fp16__*.meta.json"))):
        name = os.path.basename(p)[:-10]
        tag, _, method = name.split("__")
        if method in ("ft", "ft_val") or method.startswith(("offshelf", "val")) or method.endswith("_val"):
            continue
        size = tag.split("_")[-1] if "_" in tag else None  # <method>_<size> tags of fine-tuned systems
        if size not in SIZES or tag == f"ft_{size}":
            continue
        ref = f"ft_{size}__fp16__ft"
        if not have(ref) or not have(name):
            continue
        for s in TEST:
            try:
                c = R.compare(ref, name, s, metrics=("MER", "E_E", "cs_E_emb", "conf_rate", "sMER", "CWER"))
            except KeyError:
                continue
            comps.append(dict(system=name.replace("__fp16__", " "), vs=f"ft_{size}", set=s,
                              **{k: sig(v) for k, v in c.items()}))
    body = ("## Validation (checkpoint selection, best step by mean val MER)\n\n" + md(val)
            + "\n\n## Test: paired bootstrap difference vs plain fine-tuning of the same size"
              " (percentage points; * = 95% CI excludes 0; negative = better)\n\n" + md(pd.DataFrame(comps)))
    rr = sorted(glob.glob(os.path.join(RES, "rerank_*.json")))
    for p in rr:
        j = json.load(open(p))
        body += f"\n\n## Re-ranking weights ({os.path.basename(p)[7:-5]}), chosen on validation\n\n" + json.dumps(j["chosen"])
    flags = []
    for p in sorted(glob.glob(manifest.path("preds/*__rerank_*.jsonl"))):
        rows_ = [json.loads(l) for l in open(p, encoding="utf-8")]
        for s in TEST:
            rs = [r for r in rows_ if r["set"] == s]
            if rs:
                m = flag_pr(rs)
                flags.append(dict(system=os.path.basename(p)[:-6], set=s, precision=f(m["precision"]),
                                  recall=f(m["recall"]), flags=m["tp"] + m["fp"], critical_errors=m["ref_err"]))
    body += "\n\n## Critical-term flags (precision / recall against real critical-word errors)\n\n" + md(pd.DataFrame(flags))
    write("methods.md", "Training and decoding methods", body)


# ------------------------------------------------------------------ quantization
def quant():
    rows, frs = [], []
    tags = sorted({os.path.basename(p).split("__")[0] for p in glob.glob(manifest.path("preds/*.meta.json"))})
    for tag in tags:
        methods_ = sorted({os.path.basename(p)[:-10].split("__")[2] for p in glob.glob(manifest.path(f"preds/{tag}__*.meta.json"))})
        for method in methods_:
            if method.endswith("_val") or method.startswith(("val", "smoke")) or method.startswith("rerank"):
                continue
            base = f"{tag}__fp16__{method}"
            if not have(base):
                continue
            for prec in SIM_PREC + DEP_PREC:
                name = f"{tag}__{prec}__{method}"
                if not have(name):
                    continue
                meta = json.load(open(manifest.path(f"preds/{name}.meta.json")))
                t = R.table([name], TEST)
                for r in t.itertuples():
                    rows.append(dict(model=tag, method=method, precision=prec, set=r.set, MER=f(r.MER, r.MER_lo, r.MER_hi),
                                     cs_E_emb=f(r.cs_E_emb, r.cs_E_emb_lo, r.cs_E_emb_hi),
                                     cs_E_mat=f(r.cs_E_mat, r.cs_E_mat_lo, r.cs_E_mat_hi), CWER=f(r.CWER),
                                     size_mb=meta.get("footprint_mb", meta.get("file_mb"))))
                if prec not in REF16:
                    continue
                ref16 = f"{tag}__{REF16[prec]}__{method}"
                if not have(ref16):
                    continue
                for s, main in R_SLICES:
                    try:
                        fr = R.fragility(ref16, name, s, main=main)
                    except KeyError:
                        continue
                    frs.append(dict(model=tag, method=method, precision=prec, ref16=REF16[prec], set=s, main=main,
                                    d_emb=f(fr["d_emb"], *fr["d_emb_ci"]), d_mat=f(fr["d_mat"], *fr["d_mat_ci"]),
                                    R=(f"{fr['R']:.2f} [{fr['R_ci'][0]:.2f}, {fr['R_ci'][1]:.2f}]" if fr["reportable"]
                                       else "not reported (matrix change within noise)")))
    pd.DataFrame(rows).to_csv(os.path.join(RES, "quant.csv"), index=False)
    pd.DataFrame(frs).to_csv(os.path.join(RES, "fragility.csv"), index=False)
    body = ("## Accuracy by precision (test sets; cs_ = code-switched utterances only)\n\n" + md(pd.DataFrame(rows))
            + "\n\n## Fragility ratio R = (Eq_emb - E16_emb) / (Eq_mat - E16_mat)\n\n"
              "d_emb / d_mat: change in embedded / matrix error (points) vs the same model at 16-bit, 95% CIs. "
              "R is reported only when the CI of d_mat excludes 0.\n\n" + md(pd.DataFrame(frs)))
    # recovery: mixed precision and QAT
    recs = []
    for p in sorted(glob.glob(os.path.join(RES, "sensitivity_*.json"))):
        j = json.load(open(p))
        if "chosen" not in j:
            continue
        tag = os.path.basename(p)[len("sensitivity_"):].rsplit("_", 2)[0]
        prec = "nf4_bnb" if "nf4" in p else "int8_bnb"
        k = j["chosen"]["k"]
        fix = f"{tag}__{prec}__mp{k}"
        for s, main in R_SLICES:
            if have(fix) and have(f"{tag}__fp16__ft") and have(f"{tag}__{prec}__ft"):
                try:
                    rc = R.recovery(f"{tag}__fp16__ft", f"{tag}__{prec}__ft", fix, s, main=main)
                except KeyError:
                    continue
                recs.append(dict(fix=f"mixed precision top-{k} at 16-bit", model=tag, precision=prec, set=s, main=main,
                                 loss=f(rc["loss"], *rc["loss_ci"]),
                                 Rec=(f"{rc['Rec']:.2f} [{rc['Rec_ci'][0]:.2f}, {rc['Rec_ci'][1]:.2f}]"
                                      if rc["Rec_ci"][0] is not None else f"{rc['Rec']:.2f} (loss within noise)")))
    for qat in sorted(glob.glob(manifest.path("preds/*__nf4_qlora__*.meta.json"))):
        name = os.path.basename(qat)[:-10]
        tag, _, method = name.split("__")
        src = method.replace("qat_", "")
        for s, main in R_SLICES:
            b16, bq = f"{tag}__fp16__{src}", f"{tag}__nf4_bnb__{src}"
            if have(b16) and have(bq):
                try:
                    rc = R.recovery(b16, bq, name, s, main=main)
                except KeyError:
                    continue
                recs.append(dict(fix="QLoRA QAT (adapters 16-bit)", model=tag, precision="nf4_bnb", set=s, main=main,
                                 loss=f(rc["loss"], *rc["loss_ci"]),
                                 Rec=(f"{rc['Rec']:.2f} [{rc['Rec_ci'][0]:.2f}, {rc['Rec_ci'][1]:.2f}]"
                                      if rc["Rec_ci"][0] is not None else f"{rc['Rec']:.2f} (loss within noise)")))
    body += ("\n\n## Recovery Rec = (Eq_emb - Eq,fix_emb) / (Eq_emb - E16_emb)\n\n"
             "loss: embedded-error increase from quantization (points). Rec is reported with a CI only when the loss"
             " CI excludes 0.\n\n" + md(pd.DataFrame(recs)))
    sens = []
    for p in sorted(glob.glob(os.path.join(RES, "sensitivity_*.json"))):
        j = json.load(open(p))
        top = sorted([x for x in j["kl"] if x[0] != "ALL"], key=lambda x: -x[1])[:5]
        sens.append(dict(file=os.path.basename(p), top5_by_KL=", ".join(f"{g.replace('model.', '')} ({s:.4f})" for g, s in top),
                         all_quantized_KL=dict(j["kl"]).get("ALL"), chosen_k=j.get("chosen", {}).get("k")))
    body += "\n\n## Sensitivity scan (A6)\n\n" + md(pd.DataFrame(sens))
    write("quant.md", "Quantization (RQ3)", body)


# ------------------------------------------------------------------ efficiency
def efficiency():
    rows = []
    for p in sorted(glob.glob(manifest.path("preds/*.meta.json"))):
        m = json.load(open(p))
        if "footprint_mb" in m and not str(m.get("method", "")).startswith(("val", "smoke")) \
                and not str(m.get("method", "")).endswith("_val"):
            rows.append(dict(model=m["model"], precision=m["precision"], method=m["method"],
                             gpu_footprint_mb=m["footprint_mb"], quantized_linear_modules=m.get("quantized_modules_n", 0),
                             peak_gpu_mb_decoding=m.get("peak_gpu_mb")))
    gpu = pd.DataFrame(rows).drop_duplicates(["model", "precision"]) if rows else pd.DataFrame()
    sp = os.path.join(RES, "speed_macbook.json")
    speed = pd.DataFrame(json.load(open(sp))) if os.path.exists(sp) else pd.DataFrame()
    write("efficiency.md", "Efficiency",
          "## Simulated (bitsandbytes on GPU): measured memory footprint\n\nOnly linear layers are quantized; conv"
          " front-end, embeddings, layer norms and proj_out stay 16-bit.\n\n" + md(gpu)
          + "\n\n## Deployed (CPU, MacBook M2 Pro, 4 threads): real-time factor, peak RSS, file size\n\n" + md(speed))


def lid():
    rows = []
    for p in sorted(glob.glob(manifest.path("ckpt/lid_*/val_metrics.json"))):
        j = json.load(open(p))
        for s, v in j.items():
            rows.append(dict(head=p.split(os.sep)[-2], set=s, speech_lang_acc=f(v.get("speech_lang_acc")),
                             frame_acc=f(v["frame_acc"]),
                             frame_acc_unsmoothed=f(v["frame_acc_raw"]),
                             boundary_median_ms=round(1000 * v["boundary_median_s"]) if v["boundary_median_s"] is not None else None,
                             within_100ms=f(v["boundary_within_100ms"]), switch_detected=f(v["switch_detect_rate"])))
    write("lid.md", "Language-ID head (validation, after smoothing)",
          "Frame labels come from MMS forced alignment (PENDING_HUMAN: 30-utterance hand check).\n\n" + md(pd.DataFrame(rows)))


def main():
    os.makedirs(RES, exist_ok=True)
    for fn in (baselines, rq1, methods, quant, efficiency, lid):
        try:
            fn()
            print(f"{fn.__name__}: ok", flush=True)
        except Exception as e:  # keep building the other tables
            print(f"{fn.__name__}: FAILED {type(e).__name__}: {e}", flush=True)


if __name__ == "__main__":
    main()
