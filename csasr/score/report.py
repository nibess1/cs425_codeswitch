"""Scoring API on prediction files: cached per-utterance stats, tables, comparisons."""
import hashlib
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
STATS_VERSION = 6  # bump whenever csasr/text or csasr/score changes


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
    crit = os.environ.get("CSASR_CRITICAL_LABELS", "")
    crit_tag = "_critD2" if crit else ""  # D2 adjudicated critical words vs the rule tagger
    cache = p.replace(".jsonl", f".stats{'' if use_labse else '_nolabse'}{crit_tag}.pkl")
    crit_hash = hashlib.md5(open(crit, "rb").read()).hexdigest() if crit else ""  # label list is part of the key
    key = (STATS_VERSION, tau, rho0, os.path.getmtime(p)) + ((crit_hash,) if crit else ())
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


_SPK = {}


def speakers_of(set_name: str) -> dict:
    """utt_id -> speaker for a manifest set (cached)."""
    if set_name not in _SPK:
        from ..data import manifest
        _SPK[set_name] = {r["utt_id"]: r["speaker"] for r in manifest.load(set_name)}
    return _SPK[set_name]


def table(names: list[str], sets: list[str] | None = None, B=BS.B_DEFAULT, main=None, speaker_ci=False,
          **kw) -> pd.DataFrame:
    """One row per (system, set): point estimates and 95% CIs."""
    rows = []
    for n in names:
        for s, (ids, st) in stats(n, **kw).items():
            if sets and s not in sets:
                continue
            if main:
                ids = [u for u, x in zip(ids, st) if x["main"] == main]
                st = [x for x in st if x["main"] == main]
            if not st:
                continue
            a = BS.to_arrays(st)
            idx = BS.boot_idx(len(st), B)
            m = corpus_metrics(st)
            row = dict(system=n, set=s, n_utt=len(st), N=m["N"], N_E=m["N_E"], N_crit=m["N_crit"])
            for k in ["MER", "sMER", "CWER", "E_E", "E_Ms", "E_Mi", "E_sw", "cs_E_emb", "cs_E_mat", "conf_rate",
                      "translit_rate"]:
                row[k] = BS.point(a, k)
                lo, hi = BS.metric_ci(a, k, idx)
                row[k + "_lo"], row[k + "_hi"] = lo, hi
            if speaker_ci and not s.startswith("ascend"):
                spk_map = speakers_of(s)
                sa = BS.speaker_arrays(st, [spk_map.get(u, u) for u in ids])
                sidx = BS.boot_idx(sa["n_speakers"], B)
                row["n_speakers"] = sa["n_speakers"]
                for k in ["MER", "E_E", "cs_E_emb", "CWER"]:
                    lo, hi = BS.metric_ci(sa, k, sidx)
                    row[k + "_spk_lo"], row[k + "_spk_hi"] = lo, hi
            rows.append(row)
    return pd.DataFrame(rows)


def compare(a: str, b: str, set_name: str, metrics=("MER", "E_E", "cs_E_emb", "CWER", "sMER"), B=BS.B_DEFAULT,
            main=None, speaker_ci=False, **kw) -> dict:
    """Paired bootstrap of metric(b) - metric(a) on the utterances both systems decoded."""
    ia, sa = stats(a, **kw)[set_name]
    ib, sb = stats(b, **kw)[set_name]
    sa, sb = _align(ia, sa, ib, sb)
    if main:
        keep = [i for i, x in enumerate(sa) if x["main"] == main]
        sa, sb = [sa[i] for i in keep], [sb[i] for i in keep]
    A, Bb = BS.to_arrays(sa), BS.to_arrays(sb)
    idx = BS.boot_idx(len(sa), B)
    out = {m: BS.paired_diff(A, Bb, m, idx) for m in metrics}
    if speaker_ci and not set_name.startswith("ascend"):
        spk_map = speakers_of(set_name)
        pos = {u: i for i, u in enumerate(ib)}
        common = [u for u in ia if u in pos]
        if main:
            ma = dict(zip(ia, stats(a, **kw)[set_name][1]))
            common = [u for u in common if ma[u]["main"] == main]
        spks = [spk_map.get(u, u) for u in common]
        SA, SB = BS.speaker_arrays(sa, spks), BS.speaker_arrays(sb, spks)
        sidx = BS.boot_idx(SA["n_speakers"], B)
        for m in metrics:
            d = BS.paired_diff(SA, SB, m, sidx)
            out[m].update(spk_lo=d["lo"], spk_hi=d["hi"], spk_sig=d["sig"], n_speakers=SA["n_speakers"])
    return out


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
