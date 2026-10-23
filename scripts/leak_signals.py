#!/usr/bin/env python3
"""D3: partner-leakage signals for NSC utterances (val = pilot, Same Room test, Diff Room test).

Each NSC wav is one speaker's phone; the partner's voice can leak into it. Signals per clip:
  s1_overlap   share of the clip covered by the partner's transcribed speech (pool)
  s2_env_corr  Pearson r of 20 ms log-energy envelopes, this channel vs the partner's channel
  s3_spk_diff  max over 1.5 s windows (hop 0.75 s) of cos(window, partner) - cos(window, self),
               ECAPA (speechbrain/spkrec-ecapa-voxceleb) vs enrollment means built from up to
               8 clean utterances (no partner overlap, 2-10 s) of each speaker
  s4_ins_match units inserted by fine-tuned small (alignment I ops) that also occur in the
               partner's transcript within +-2 s of the clip; s4_ins total inserted units
-> data/leak/signals.parquet
  CUDA_VISIBLE_DEVICES=7 python scripts/leak_signals.py
"""
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
import soundfile as sf
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from csasr.data import manifest  # noqa: E402
from csasr.score import report as R  # noqa: E402
from csasr.score.align import I, align  # noqa: E402
from csasr.text.normalize import normalize  # noqa: E402
from csasr.text.units import split_units  # noqa: E402

SETS = {"nsc_val": "ft_small__fp16__ft_val", "nsc_test": "ft_small__fp16__ft", "nsc_diff_test": "ft_small__fp16__ft"}
SR = 16000
COMMON = {"lah", "leh", "lor", "ah", "oh", "uh", "um", "eh", "ya", "yeah", "ok", "the", "a", "i", "you", "and", "so",
          "is", "to", "it", "that", "嗯", "啊", "哦", "呃", "的", "了", "是", "我", "你"}


def load16(path, cache):
    if path not in cache:
        if len(cache) > 6:
            cache.clear()
        w, sr = sf.read(path, dtype="float32")
        if w.ndim > 1:
            w = w.mean(1)
        if sr != SR:
            import librosa
            w = librosa.resample(w, orig_sr=sr, target_sr=SR)
        cache[path] = w
    return cache[path]


def envelope(x):
    n = len(x) // 320
    if n < 3:
        return None
    f = x[: n * 320].reshape(n, 320)
    return np.log10((f ** 2).mean(1) + 1e-8)


def main():
    from speechbrain.inference.speaker import EncoderClassifier
    enc = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb",
                                         savedir=os.path.expanduser("~/hf-cache/ecapa"), run_opts={"device": "cuda"})

    def embed(segs):
        out = []
        for b in range(0, len(segs), 32):
            chunk = segs[b: b + 32]
            L = max(len(s) for s in chunk)
            x = torch.zeros(len(chunk), L)
            lens = torch.tensor([len(s) / L for s in chunk])
            for i, s in enumerate(chunk):
                x[i, : len(s)] = torch.from_numpy(s)
            with torch.no_grad():
                e = enc.encode_batch(x.cuda(), lens.cuda()).squeeze(1)
            out.append(torch.nn.functional.normalize(e, dim=-1).cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, 192))

    pool = pd.read_parquet(manifest.path("data/nsc/pool.parquet"))
    pool["key"] = pool.room + "_" + pool.session
    chan = {(k, s): g for (k, s), g in pool.groupby(["key", "speaker"])}
    partner_of = {}
    for k, g in pool.groupby("key"):
        spk = sorted(g.speaker.unique())
        for s in spk:
            others = [o for o in spk if o != s]
            partner_of[(k, s)] = others[0] if others else None
    rows = []
    for set_name, sysname in SETS.items():
        man = manifest.load(set_name)
        preds = {r["utt_id"]: r["hyp"] for r in R.read_preds(sysname) if r["set"] == set_name}
        for r in man:
            rows.append(dict(set=set_name, utt_id=r["utt_id"], hyp=preds.get(r["utt_id"], ""), ref=r["ref_raw"]))
    df = pd.DataFrame(rows).merge(pool[["utt_id", "key", "speaker", "wav", "start", "end", "partner_overlap"]], on="utt_id")
    print(len(df), "clips", flush=True)

    # enrollment: up to 8 clean utterances per speaker channel (excluding evaluated clips)
    evaluated = set(df.utt_id)
    enroll_emb = {}
    cache = {}
    need = sorted({(k, s) for k, s in zip(df.key, df.speaker)} | {(k, partner_of[(k, s)]) for k, s in zip(df.key, df.speaker)
                                                                 if partner_of[(k, s)]})
    for k, s in need:
        g = chan.get((k, s))
        if g is None:
            continue
        cand = g[(g.drop_reason == "") & (g.partner_overlap == 0) & (g.dur.between(2, 10)) & ~g.utt_id.isin(evaluated)]
        cand = cand.sample(min(8, len(cand)), random_state=425) if len(cand) else cand
        if not len(cand):
            continue
        w = load16(cand.wav.iloc[0], cache)
        segs = [w[int(a * SR): int(b * SR)] for a, b in zip(cand.start, cand.end)]
        e = embed(segs).mean(0)
        enroll_emb[(k, s)] = e / (np.linalg.norm(e) + 1e-9)
    print(len(enroll_emb), "enrolled channels", flush=True)

    out = []
    for (k, s), g in df.groupby(["key", "speaker"]):
        p = partner_of.get((k, s))
        pg = chan.get((k, p)) if p else None
        own = load16(g.wav.iloc[0], cache)
        par = load16(pg.wav.iloc[0], cache) if pg is not None else None
        p_ivs = list(zip(pg.start, pg.end, pg.ref)) if pg is not None else []
        wins, owners = [], []
        for r in g.itertuples():
            seg = own[int(r.start * SR): int(r.end * SR)]
            # s2: envelope correlation with the partner's own channel over the same window
            env_r = np.nan
            if par is not None:
                a, b = envelope(seg), envelope(par[int(r.start * SR): int(r.end * SR)])
                if a is not None and b is not None and len(a) == len(b) and a.std() > 0 and b.std() > 0:
                    env_r = float(np.corrcoef(a, b)[0, 1])
            # s4: inserted units matching the partner's transcript within +-2 s
            ref_u = split_units(normalize(r.ref))
            hyp_u = split_units(normalize(r.hyp))
            ins = [hyp_u[j] for op, _, j in align(ref_u, hyp_u) if op == I]
            near = set()
            for a0, b0, t in p_ivs:
                if b0 >= r.start - 2 and a0 <= r.end + 2:
                    near |= set(split_units(t))
            match = [u for u in ins if u in near and u not in COMMON]
            # s3 windows
            L, H = int(1.5 * SR), int(0.75 * SR)
            if len(seg) <= L:
                ws = [seg]
            else:
                ws = [seg[i: i + L] for i in range(0, len(seg) - L + 1, H)]
            for wv in ws:
                wins.append(wv); owners.append(r.utt_id)
            out.append(dict(set=r.set, utt_id=r.utt_id, speaker=s, partner=p, dur=r.end - r.start,
                            s1_overlap=r.partner_overlap, s2_env_corr=env_r, s4_ins=len(ins), s4_ins_match=len(match),
                            ins_match_units=" ".join(match), hyp=r.hyp, ref=r.ref))
        E = embed(wins) if wins else np.zeros((0, 192))
        self_e, par_e = enroll_emb.get((k, s)), enroll_emb.get((k, p)) if p else None
        best = defaultdict(lambda: np.nan)
        if self_e is not None and par_e is not None and len(E):
            d = E @ par_e - E @ self_e
            for u, v in zip(owners, d):
                best[u] = max(best[u], float(v)) if not np.isnan(best[u]) else float(v)
        for o in out[-len(g):]:
            o["s3_spk_diff"] = best[o["utt_id"]]
    res = pd.DataFrame(out)
    os.makedirs(manifest.path("data/leak"), exist_ok=True)
    res.to_parquet(manifest.path("data/leak/signals.parquet"))
    print(res.groupby("set")[["s1_overlap", "s2_env_corr", "s3_spk_diff", "s4_ins_match"]].describe().T.to_string())


if __name__ == "__main__":
    main()
