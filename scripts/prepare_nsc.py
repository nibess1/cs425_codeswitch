#!/usr/bin/env python3
"""Prepare NSC Part 4 (Mandarin speakers): utterances, filters, speaker split, clips.

  python scripts/prepare_nsc.py --raw data/nsc_part4/raw --out data/nsc

Writes data/nsc/pool.parquet (every utterance, no audio), data/nsc/manifests/*.jsonl
(selected splits) and data/nsc/clips/<split>/<utt_id>.flac.
"""
import argparse
import json
import os
import random
import sys
import warnings
from multiprocessing import Pool

import librosa
import numpy as np
import pandas as pd
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from csasr.data.nsc import SILENCE, index, overlap, read_textgrid  # noqa: E402
from csasr.text.normalize import normalize  # noqa: E402
from csasr.text.units import label, training_target  # noqa: E402

MIN_DUR, MAX_DUR = 0.3, 30.0
PARTNER_OVERLAP_MAX = 0.25  # proxy for audible untranscribed partner speech (PENDING_HUMAN)


def parse_file(args):
    rec, partner_tg = args
    ivs = read_textgrid(rec["textgrid"])
    wav_dur = sf.info(rec["wav"]).duration  # a few transcripts run past the end of the audio
    partner = []
    if partner_tg:
        partner = sorted((a, b) for a, b, t in read_textgrid(partner_tg) if t not in SILENCE)
    rows = []
    for k, (a, b, txt) in enumerate(ivs):
        if txt in SILENCE:
            continue
        dur = b - a
        drop = None
        if "<UNK>" in txt or "[UNK]" in txt:
            drop = "unk"
        elif dur < MIN_DUR or dur > MAX_DUR:
            drop = "duration"
        elif b > wav_dur:
            drop = "beyond_audio"
        norm = normalize(txt)
        if not norm and not drop:
            drop = "empty"
        ov = overlap(a, b, partner) / dur if partner and dur > 0 else 0.0
        if not drop and ov > PARTNER_OVERLAP_MAX:
            drop = "partner_overlap"
        lab = label(norm, already_normalized=True) if norm else None
        rows.append(dict(
            utt_id=f"nsc_{rec['room'][0]}_{rec['session']}_{rec['speaker']}_{k:05d}",
            session=rec["session"], speaker=rec["speaker"], room=rec["room"], wav=rec["wav"],
            start=a, end=b, dur=dur, ref_raw=txt, ref=norm,
            target=training_target(txt) if norm else "",
            main=lab.main if lab else "", mixed=bool(lab and lab.mixed),
            n_units=len(lab.units) if lab else 0,
            n_emb=sum(r == "E" for r in lab.roles) if lab else 0,
            partner_overlap=round(ov, 3), drop_reason=drop or "",
        ))
    return rows


def cut_clips(args):
    wav, items, clip_dir = args
    audio, sr = sf.read(wav, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(1)
    out = []
    for it in items:
        path = os.path.join(clip_dir, it["split"], it["utt_id"] + ".flac")
        if not os.path.exists(path):
            seg = audio[int(it["start"] * sr): int(it["end"] * sr)]
            if sr != 16000:  # a few Diff Room channels are 44.1 kHz
                seg = librosa.resample(seg, orig_sr=sr, target_sr=16000)
            sf.write(path, seg, 16000, format="FLAC", subtype="PCM_16")
        out.append((it["utt_id"], path))
    return out


def take_hours(df, hours, rng_seed):
    """Random subset of rows whose durations sum to at most `hours`."""
    df = df.sample(frac=1.0, random_state=rng_seed)
    return df[df.dur.cumsum() <= hours * 3600]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/nsc_part4/raw")
    ap.add_argument("--sheet", default="data/nsc_part4/speaker_metadata.xlsx")
    ap.add_argument("--out", default="data/nsc")
    ap.add_argument("--train_hours", type=float, default=40.0)
    ap.add_argument("--train_mixed_hours", type=float, default=30.0)
    ap.add_argument("--val_hours", type=float, default=3.0)
    ap.add_argument("--test_hours", type=float, default=8.0)
    ap.add_argument("--diff_hours", type=float, default=4.0)
    ap.add_argument("--seed", type=int, default=425)
    ap.add_argument("-j", type=int, default=32)
    a = ap.parse_args()
    os.makedirs(os.path.join(a.out, "manifests"), exist_ok=True)

    recs = index(a.raw)
    by_sess = {}
    for r in recs:
        by_sess.setdefault((r["room"], r["session"]), []).append(r)
    jobs = []
    for r in recs:
        mates = [x for x in by_sess[(r["room"], r["session"])] if x["speaker"] != r["speaker"]]
        jobs.append((r, mates[0]["textgrid"] if mates else None))
    print(f"{len(recs)} speaker channels in {len(by_sess)} sessions", flush=True)

    with Pool(a.j) as p:
        rows = [x for chunk in p.imap_unordered(parse_file, jobs, chunksize=4) for x in chunk]
    pool = pd.DataFrame(rows)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sheet = pd.read_excel(a.sheet, dtype=str)
    sheet = sheet[["Speaker ID", "Age", "Gender", "First Language", "Dominant Language", "Partner ID"]]
    sheet = sheet.drop_duplicates("Speaker ID").rename(columns={
        "Speaker ID": "speaker", "Age": "age", "Gender": "gender", "First Language": "first_lang",
        "Dominant Language": "dominant_lang", "Partner ID": "partner_id"})
    pool = pool.merge(sheet, on="speaker", how="left")

    # speaker split over Same Room sessions (partners stay together); Diff Room is test-only
    rng = random.Random(a.seed)
    same_sess = sorted(pool[pool.room == "same"].session.unique())
    rng.shuffle(same_sess)
    n = len(same_sess)
    cut1, cut2 = int(0.7 * n), int(0.8 * n)
    split_of = {s: "train" for s in same_sess[:cut1]}
    split_of.update({s: "val" for s in same_sess[cut1:cut2]})
    split_of.update({s: "test" for s in same_sess[cut2:]})
    pool["split"] = np.where(pool.room == "same", pool.session.map(split_of), "diff_test")
    spk_split = pool[pool.room == "same"].groupby("speaker").split.agg(set)
    assert all(len(s) == 1 for s in spk_split), "speaker in two Same Room splits"
    # a few speakers also appear in Diff Room; drop them there if they are in train/val
    seen = set(pool[(pool.room == "same") & pool.split.isin(["train", "val"])].speaker)
    pool.loc[(pool.room == "diff") & pool.speaker.isin(seen) & (pool.drop_reason == ""), "drop_reason"] = "speaker_seen"
    pool.to_parquet(os.path.join(a.out, "pool.parquet"))

    ok = pool[pool.drop_reason == ""]
    tr = ok[ok.split == "train"]
    mixed = take_hours(tr[tr.mixed], a.train_mixed_hours, a.seed)
    rest_h = a.train_hours - mixed.dur.sum() / 3600
    other = take_hours(tr[~tr.mixed], rest_h, a.seed)
    sel = {
        "train": pd.concat([mixed, other]),
        "val": take_hours(ok[ok.split == "val"], a.val_hours, a.seed),
        "test": take_hours(ok[ok.split == "test"], a.test_hours, a.seed),
        "diff_test": take_hours(ok[ok.split == "diff_test"], a.diff_hours, a.seed),
    }
    clip_dir = os.path.join(a.out, "clips")
    for s in sel:
        os.makedirs(os.path.join(clip_dir, s), exist_ok=True)
    allsel = pd.concat([d.assign(split=s) for s, d in sel.items()])
    jobs = [(w, g.to_dict("records"), clip_dir) for w, g in allsel.groupby("wav")]
    paths = {}
    with Pool(min(a.j, 16)) as p:
        for chunk in p.imap_unordered(cut_clips, jobs):
            paths.update(chunk)
    corpus = {"train": "nsc_same", "val": "nsc_same", "test": "nsc_same", "diff_test": "nsc_diff"}
    for s, d in sel.items():
        with open(os.path.join(a.out, "manifests", f"{s}.jsonl"), "w") as f:
            for r in d.sort_values("utt_id").to_dict("records"):
                f.write(json.dumps(dict(
                    utt_id=r["utt_id"], corpus=corpus[s], split="test" if s == "diff_test" else s,
                    audio=paths[r["utt_id"]], duration=round(r["dur"], 3), speaker=r["speaker"],
                    session=r["session"], ref_raw=r["ref_raw"], ref=r["ref"], target=r["target"],
                    main=r["main"], mixed=r["mixed"]), ensure_ascii=False) + "\n")
        print(f"{s}: {len(d)} utts, {d.dur.sum() / 3600:.1f} h, mixed {d.mixed.mean():.1%}, "
              f"speakers {d.speaker.nunique()}", flush=True)
    print(pool.groupby(["room", "drop_reason"]).size().to_string())


if __name__ == "__main__":
    main()
