#!/usr/bin/env python3
"""RQ1: where does Whisper fail on code-switched speech, role or language? (§4.2, A5)

(a) same-word comparison: an English word embedded in Chinese-main sentences vs. the same
    word as a matrix unit in English-main sentences (and Chinese characters likewise);
(b) logistic regression  logit P(err_i) = b0 + b1 emb + b2 en + b3 emb*en + b4 log f + b5 d + b6 nsc,
    with a speaker-cluster bootstrap CI for every coefficient.

  python scripts/rq1.py small__fp16__offshelf_free ft_small__fp16__ft
"""
import collections
import json
import os
import pickle
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from csasr.data import manifest  # noqa: E402
from csasr.score import report as R  # noqa: E402
from csasr.score.metrics import analyze  # noqa: E402
from csasr.text.units import split_units  # noqa: E402

SETS = ["ascend_test", "nsc_test"]


def train_freq():
    c = collections.Counter()
    for s in ["ascend_train", "nsc_train"]:
        for r in manifest.load(s):
            c.update(split_units(r["ref"]))
    return c


def unit_table(system, freq, durs):
    rows = []
    preds = {}
    for r in R.read_preds(system):
        preds[(r["set"], r["utt_id"])] = r
    for s in SETS:
        man = {r["utt_id"]: r for r in manifest.load(s)}
        items = [(u, p) for (ss, u), p in preds.items() if ss == s]
        st = analyze([(p["ref"], p["hyp"]) for _, p in items], keep_units=True)
        for (uid, _), x in zip(items, st):
            if not x["mixed"] or x["main"] not in ("zh", "en"):
                continue
            u = x["_units"]
            spans = durs.get(s, {}).get(uid)
            for i, (unit, lang, role, err) in enumerate(zip(u["units"], u["langs"], u["roles"], u["err"])):
                if role not in ("E", "Ms", "Mi"):
                    continue
                d = (spans[i][1] - spans[i][0]) if spans and i < len(spans) else np.nan
                rows.append(dict(set=s, utt=uid, speaker=man[uid]["speaker"], unit=unit, lang=lang, role=role,
                                 emb=int(role == "E"), en=int(lang == "en"), nsc=int(s.startswith("nsc")),
                                 logf=np.log1p(freq.get(unit, 0)), dur=d, err=int(err), main=x["main"]))
    return pd.DataFrame(rows)


def same_word(df):
    """Error of the same word type when embedded vs. when matrix, averaged over word types
    seen in both roles (weighted by the smaller count)."""
    out = {}
    for lang in ("en", "zh"):
        d = df[df.lang == lang]
        g = d.groupby(["unit", "emb"]).err.agg(["mean", "count"]).unstack("emb").dropna()
        if g.empty:
            out[lang] = None
            continue
        w = np.minimum(g[("count", 0)], g[("count", 1)])
        e_emb = float((g[("mean", 1)] * w).sum() / w.sum())
        e_mat = float((g[("mean", 0)] * w).sum() / w.sum())
        rng = np.random.default_rng(0)
        types = np.arange(len(g))
        diffs = []
        for _ in range(1000):
            k = rng.choice(types, len(types))
            ww = w.values[k]
            diffs.append((g[("mean", 1)].values[k] * ww).sum() / ww.sum() - (g[("mean", 0)].values[k] * ww).sum() / ww.sum())
        out[lang] = dict(n_types=int(len(g)), n_emb=int(g[("count", 1)].sum()), n_mat=int(g[("count", 0)].sum()),
                         err_emb=e_emb, err_mat=e_mat, diff=e_emb - e_mat,
                         diff_ci=[float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))])
    return out


def regression(df, B=200):
    import statsmodels.api as sm
    cols = ["emb", "en", "emb_en", "logf", "nsc"]
    d = df.copy()
    if d.dur.notna().mean() > 0.5:  # spoken length needs the forced alignment of the test set
        cols.insert(4, "dur")
        d = d.dropna(subset=["dur"])
    d["emb_en"] = d.emb * d.en

    def fit(x):
        return sm.Logit(x.err.astype(float), sm.add_constant(x[cols].astype(float), has_constant="add")).fit(disp=0)

    coefs = fit(d).params.to_dict()
    spk = d.speaker.unique()
    rng = np.random.default_rng(0)
    groups = {s: g for s, g in d.groupby("speaker")}
    boots = []
    for _ in range(B):
        try:
            boots.append(fit(pd.concat([groups[s] for s in rng.choice(spk, len(spk))])).params)
        except Exception:
            continue
    bt = pd.DataFrame(boots)
    ci = {k: [float(bt[k].quantile(0.025)), float(bt[k].quantile(0.975))] for k in bt.columns}
    return dict(n_units=int(len(d)), n_speakers=int(len(spk)), covariates=cols, coef=coefs, ci=ci, n_boot=len(bt))


def main():
    freq = train_freq()
    durs = {}
    for s in SETS:
        p = manifest.path(f"data/align/{s}.pkl")
        if os.path.exists(p):
            al = pickle.load(open(p, "rb"))
            durs[s] = {u: [(a, b) for a, b, _ in v["spans"]] for u, v in al.items()}
    res = {}
    for system in sys.argv[1:]:
        df = unit_table(system, freq, durs)
        role_tab = df.groupby(["set", "main", "role"]).err.agg(["mean", "count"]).reset_index()
        res[system] = dict(role_unit_error=role_tab.to_dict("records"), same_word=same_word(df),
                           regression=regression(df))
        print(system, json.dumps(res[system]["same_word"]), json.dumps(res[system]["regression"]["coef"]))
    out = manifest.path("results/rq1.json")
    old = json.load(open(out)) if os.path.exists(out) else {}
    old.update(res)
    json.dump(old, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
