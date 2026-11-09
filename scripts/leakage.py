#!/usr/bin/env python3
"""D3: partner-leakage signals and labels for NSC utterances (val = pilot, Same Room test, Diff Room test).

Each NSC wav is one speaker's phone; the partner's voice can leak into it. Signals per clip:
  s1_overlap   share of the clip covered by the partner's transcribed speech (pool)
  s2_env_corr  Pearson r of 20 ms log-energy envelopes, this channel vs the partner's channel
  s3_spk_diff  max over 1.5 s windows (hop 0.75 s) of cos(window, partner) - cos(window, self),
               ECAPA (speechbrain/spkrec-ecapa-voxceleb) vs enrollment means built from up to
               8 clean utterances (no partner overlap, 2-10 s) of each speaker
  s4_ins_match units inserted by fine-tuned small (alignment I ops) that also occur in the
               partner's transcript within +-2 s of the clip; s4_ins total inserted units
-> data/leak/signals.parquet

Then, on that file (CPU): s5 adds s5_out_match (output units not in the reference but in the partner's
transcript within +-2 s); rule labels every clip (clean / faint / partner_words_audible) and stores the
pilot judgements (annotator 1, NSC val); final applies annotator 1's judgements of the ambiguous test
clips -> leak_label; sensitivity prints MER/E with flagged clips removed and writes review/leak_audit.csv.

  CUDA_VISIBLE_DEVICES=7 python scripts/leakage.py signals
  python scripts/leakage.py s5 && python scripts/leakage.py rule && python scripts/leakage.py final
  python scripts/leakage.py sensitivity
"""
import argparse
import csv
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
from csasr.score.metrics import corpus_metrics  # noqa: E402
from csasr.score.align import C, I, align  # noqa: E402
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


def signals():
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


S5_COMMON = {"lah", "leh", "lor", "ah", "oh", "uh", "um", "eh", "ya", "yeah", "ok", "the", "a", "i", "you", "and", "so",
             "is", "to", "it", "that", "like", "then", "but", "what", "this", "one", "no", "not", "we", "they", "he",
             "she", "my", "your", "in", "of", "for", "do", "be", "are", "was", "have", "got", "can", "know", "just",
             "very", "also", "right", "huh", "wah", "sia", "hor", "meh", "mah",
             "嗯", "啊", "哦", "呃", "的", "了", "是", "我", "你", "他", "她", "这", "那", "个", "就", "也", "不", "在", "有",
             "都", "吗", "呢", "吧", "啦"}
SIGNALS = "data/leak/signals.parquet"


def s5():
    s = pd.read_parquet(manifest.path(SIGNALS))
    pool = pd.read_parquet(manifest.path("data/nsc/pool.parquet"))
    pool["key"] = pool.room + "_" + pool.session
    P = pool.set_index("utt_id")
    by_chan = {k: g for k, g in pool.groupby(["key", "speaker"])}
    s5, s5f, ptxt = [], [], []
    for r in s.itertuples():
        me = P.loc[r.utt_id]
        g = by_chan.get((me.key, r.partner))
        near = g[(g.end >= me.start - 2) & (g.start <= me.end + 2)] if g is not None else []
        pu = set(u for t in (near.ref if len(near) else []) for u in split_units(t))
        ref_u = set(split_units(normalize(r.ref)))
        hyp_u = split_units(normalize(r.hyp))
        m = [u for u in hyp_u if u not in ref_u and u in pu and u not in S5_COMMON]
        s5.append(len(m))
        s5f.append(len(m) / max(1, len(hyp_u)))
        ptxt.append(" / ".join(near.ref_raw.astype(str))[:200] if len(near) else "")
    s["s5_out_match"] = s5
    s["s5_frac"] = s5f
    s["partner_text"] = ptxt
    s.to_parquet(manifest.path(SIGNALS))
    v = s[s.set == "nsc_val"]
    for t in [1, 2, 3, 4]:
        print("val s5 >=", t, (v.s5_out_match >= t).sum())
    print(pd.crosstab(v.s5_out_match.clip(0, 4), pd.cut(v.s3_spk_diff, [-1, 0, 0.1, 0.2, 1])))


def rule():
    s = pd.read_parquet(manifest.path(SIGNALS))

    def recall(r):
        ru = split_units(normalize(r.ref))
        hu = split_units(normalize(r.hyp))
        return sum(op == C for op, _, _ in align(ru, hu)) / max(1, len(ru))

    def label(r):
        if (r.s5_out_match >= 2 and r.s3_spk_diff > -0.15) or (r.s5_out_match >= 1 and r.s3_spk_diff > 0.1) \
                or (r.s3_spk_diff > 0.2 and r.ref_recall < 0.3):
            return "partner_words_audible"
        if r.s3_spk_diff > 0.2:
            return "faint"
        return "clean"

    s["ref_recall"] = [recall(r) for r in s.itertuples()]
    s["rule_label"] = [label(r) for r in s.itertuples()]
    s.to_parquet(manifest.path(SIGNALS))
    # pilot judgements (annotator 1, PROCESS_LOG): index into the 103 val cases sorted by utt_id
    v = s[(s.set == "nsc_val") & ((s.s5_out_match >= 1) | (s.s3_spk_diff > 0.2))].sort_values("utt_id").reset_index(drop=True)
    P = {1, 12, 13, 14, 31, 32, 33, 34, 43, 52, 53, 54, 55, 57, 58, 59, 63, 65, 66, 67, 68, 69, 70, 71, 72, 74, 87, 89, 94, 98}
    F = {0, 11, 20, 21, 22, 23, 24, 25, 28, 29, 30, 35, 37, 38, 39, 42, 47, 51, 56, 60, 61, 62, 64, 76, 80, 82, 86, 88, 92,
         97, 100, 101, 102}
    v["judge"] = ["partner_words_audible" if i in P else ("faint" if i in F else "clean") for i in range(len(v))]
    print("pilot cases", len(v), "rule vs my judgement agreement", round((v.judge == v.rule_label).mean(), 3))
    print(pd.crosstab(v.judge, v.rule_label))
    print(v[v.judge != v.rule_label][["utt_id", "s3_spk_diff", "s5_out_match", "ref_recall", "judge", "rule_label"]].to_string())
    v[["utt_id", "judge"]].to_csv(manifest.path("data/llm_labels/d3_pilot_judgements.csv"), index=False)
    for st in ["nsc_val", "nsc_test", "nsc_diff_test"]:
        print(st, s[s.set == st].rule_label.value_counts(normalize=True).round(4).to_dict())


def final():
    s = pd.read_parquet(manifest.path(SIGNALS))
    amb = pd.read_parquet(manifest.path("data/leak/ambiguous_test.parquet"))
    P = {20, 55, 61, 82, 83, 87, 88, 89, 90, 91, 92, 94, 95, 96, 98, 99, 100, 117, 127, 128, 138, 149}
    F = {2, 3, 13, 28, 42, 50, 58, 59, 60, 68, 69, 81, 84, 93, 97, 110, 129, 135, 136, 144}
    amb["judge"] = ["partner_words_audible" if i in P else ("faint" if i in F else "clean") for i in range(len(amb))]
    print("ambiguous: rule vs my judgement agreement", round((amb.judge == amb.rule_label).mean(), 3))
    print(pd.crosstab(amb.judge, amb.rule_label))
    j = dict(zip(amb.utt_id, amb.judge))
    s["leak_label"] = [j.get(u, r) if st != "nsc_val" else r for u, r, st in zip(s.utt_id, s.rule_label, s.set)]
    s["leak_label_source"] = ["annotator1" if u in j else "rule" for u in s.utt_id]
    s.to_parquet(manifest.path(SIGNALS))
    amb[["utt_id", "set", "rule_label", "judge"]].to_csv(manifest.path("data/llm_labels/d3_test_judgements.csv"), index=False)
    for st in ["nsc_test", "nsc_diff_test"]:
        t = s[s.set == st]
        vc = t.leak_label.value_counts()
        print(st, len(t), {k: f"{v} ({100*v/len(t):.2f}%)" for k, v in vc.items()})


def sensitivity():
    s = pd.read_parquet(manifest.path(SIGNALS))
    flag = set(s[s.leak_label != "clean"].utt_id)
    for sysn in ["small__fp16__offshelf_free", "ft_small__fp16__ft", "full_small__fp16__full", "ft_base__fp16__ft",
                 "full_base__fp16__full"]:
        st = R.stats(sysn, use_labse=False)
        row = []
        for setn in ["nsc_test", "nsc_diff_test"]:
            ids, x = st[setn]
            a = corpus_metrics(x)
            b = corpus_metrics([v for u, v in zip(ids, x) if u not in flag])
            row.append(f"{setn}: MER {100*a['MER']:.2f}->{100*b['MER']:.2f} emb {100*a['E_E']:.2f}->{100*b['E_E']:.2f}")
        print(sysn, " | ".join(row))
    t = s[s.set != "nsc_val"]
    aud = pd.concat([t[t.leak_label != "clean"].sample(10, random_state=425),
                     t[t.leak_label == "clean"].sample(10, random_state=425)]).sample(frac=1, random_state=425)
    man = {r["utt_id"]: r for st in ("nsc_test", "nsc_diff_test") for r in manifest.load(st)}
    os.makedirs(manifest.path("review"), exist_ok=True)
    with open(manifest.path("review/leak_audit.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["utt_id", "audio", "reference", "fine_tuned_small_output", "partner_transcript_within_2s",
                    "automatic_label", "signal_speaker_verif", "signal_partner_words",
                    "your_label (clean/faint/partner_words_audible)"])
        for r in aud.itertuples():
            w.writerow([r.utt_id, man[r.utt_id]["audio"], r.ref, r.hyp, r.partner_text, r.leak_label,
                        round(r.s3_spk_diff, 3), r.s5_out_match, ""])
    print("review/leak_audit.csv: 20 rows")


def main():
    steps = {"signals": signals, "s5": s5, "rule": rule, "final": final, "sensitivity": sensitivity}
    ap = argparse.ArgumentParser(description="D3 partner leakage (see the top of this file)")
    ap.add_argument("step", choices=list(steps))
    steps[ap.parse_args().step]()


if __name__ == "__main__":
    main()
