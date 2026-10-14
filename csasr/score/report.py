"""Scoring API on prediction files: cached per-utterance stats, tables, comparisons."""
import json
import os
import pickle

import numpy as np
import pandas as pd

from ..data.manifest import path
from . import bootstrap as BS
from .confusion import DEFAULT_TAU, LaBSE
from .metrics import analyze, corpus_metrics

_LABSE = None
STATS_VERSION = 5  # bump whenever csasr/text or csasr/score changes


def labse():
    global _LABSE
    if _LABSE is None:
        _LABSE = LaBSE(device=os.environ.get("CSASR_LABSE_DEVICE", "cpu"))
    return _LABSE


def read_preds(name_or_path: str) -> list[dict]:
    p = name_or_path if name_or_path.endswith(".jsonl") else path(f"preds/{name_or_path}.jsonl")
    return [json.loads(l) for l in open(p, encoding="utf-8")]


def stats(name_or_path: str, use_labse=True, tau=DEFAULT_TAU, rho0=0.5, refresh=False) -> dict:
    """{set_name: (utt_ids, list_of_stat_dicts)} with an on-disk cache next to the preds."""
    p = name_or_path if name_or_path.endswith(".jsonl") else path(f"preds/{name_or_path}.jsonl")
    cache = p.replace(".jsonl", f".stats{'' if use_labse else '_nolabse'}.pkl")
    key = (STATS_VERSION, tau, rho0, os.path.getmtime(p))
    if not refresh and os.path.exists(cache):
        c = pickle.load(open(cache, "rb"))
        if c.get("key") == key:
            return c["data"]
    rows = read_preds(p)
    by_set = {}
    for r in rows:
        by_set.setdefault(r["set"], []).append(r)
    data = {}
    lb = labse() if use_labse else None
    for s, rs in by_set.items():
        st = analyze([(r["ref"], r["hyp"]) for r in rs], labse=lb, tau=tau, rho0=rho0)
        data[s] = ([r["utt_id"] for r in rs], st)
    if lb is not None:
        lb.save()
    pickle.dump({"key": key, "data": data}, open(cache, "wb"))
    return data


def _align(ids_a, st_a, ids_b, st_b):
    pos = {u: i for i, u in enumerate(ids_b)}
    common = [u for u in ids_a if u in pos]
    ia = {u: i for i, u in enumerate(ids_a)}
    return [st_a[ia[u]] for u in common], [st_b[pos[u]] for u in common]


def table(names: list[str], sets: list[str] | None = None, B=BS.B_DEFAULT, main=None, **kw) -> pd.DataFrame:
    """One row per (system, set): point estimates and 95% CIs."""
    rows = []
    for n in names:
        for s, (ids, st) in stats(n, **kw).items():
            if sets and s not in sets:
                continue
            if main:
                st = [x for x in st if x["main"] == main]
            if not st:
                continue
            a = BS.to_arrays(st)
            idx = BS.boot_idx(len(st), B)
            m = corpus_metrics(st)
            row = dict(system=n, set=s, n_utt=len(st), N=m["N"], N_E=m["N_E"], N_crit=m["N_crit"])
            for k in ["MER", "sMER", "CWER", "E_E", "E_Ms", "E_Mi", "E_sw", "cs_E_emb", "cs_E_mat", "conf_rate"]:
                row[k] = BS.point(a, k)
                lo, hi = BS.metric_ci(a, k, idx)
                row[k + "_lo"], row[k + "_hi"] = lo, hi
            rows.append(row)
    return pd.DataFrame(rows)


def compare(a: str, b: str, set_name: str, metrics=("MER", "E_E", "cs_E_emb", "CWER", "sMER"), B=BS.B_DEFAULT,
            main=None, **kw) -> dict:
    """Paired bootstrap of metric(b) - metric(a) on the utterances both systems decoded."""
    ia, sa = stats(a, **kw)[set_name]
    ib, sb = stats(b, **kw)[set_name]
    sa, sb = _align(ia, sa, ib, sb)
    if main:
        keep = [i for i, x in enumerate(sa) if x["main"] == main]
        sa, sb = [sa[i] for i in keep], [sb[i] for i in keep]
    A, Bb = BS.to_arrays(sa), BS.to_arrays(sb)
    idx = BS.boot_idx(len(sa), B)
    return {m: BS.paired_diff(A, Bb, m, idx) for m in metrics}


def fragility(sys16: str, sysq: str, set_name: str, main=None, B=BS.B_DEFAULT, **kw) -> dict:
    i16, s16 = stats(sys16, **kw)[set_name]
    iq, sq = stats(sysq, **kw)[set_name]
    s16, sq = _align(i16, s16, iq, sq)
    if main:
        keep = [i for i, x in enumerate(s16) if x["main"] == main]
        s16, sq = [s16[i] for i in keep], [sq[i] for i in keep]
    return BS.fragility(BS.to_arrays(s16), BS.to_arrays(sq), BS.boot_idx(len(s16), B))


def recovery(sys16: str, sysq: str, sysfix: str, set_name: str, main=None, B=BS.B_DEFAULT, **kw) -> dict:
    i16, s16 = stats(sys16, **kw)[set_name]
    iq, sq = stats(sysq, **kw)[set_name]
    ifx, sfx = stats(sysfix, **kw)[set_name]
    common = set(i16) & set(iq) & set(ifx)
    o = [u for u in i16 if u in common]  # align all three on the order of sys16
    m16, mq, mf = dict(zip(i16, s16)), dict(zip(iq, sq)), dict(zip(ifx, sfx))
    s16, sq, sfx = [m16[u] for u in o], [mq[u] for u in o], [mf[u] for u in o]
    if main:
        keep = [i for i, x in enumerate(s16) if x["main"] == main]
        s16, sq, sfx = [[s[i] for i in keep] for s in (s16, sq, sfx)]
    return BS.recovery(BS.to_arrays(s16), BS.to_arrays(sq), BS.to_arrays(sfx), BS.boot_idx(len(s16), B))


def fmt(x, lo=None, hi=None, pct=True):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "–"
    f = (lambda v: f"{100 * v:.1f}") if pct else (lambda v: f"{v:.2f}")
    if lo is None or (isinstance(lo, float) and np.isnan(lo)):
        return f(x)
    return f"{f(x)} [{f(lo)}, {f(hi)}]"
