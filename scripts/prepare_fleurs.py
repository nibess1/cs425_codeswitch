#!/usr/bin/env python3
"""FLEURS en_us + cmn_hans_cn (train) -> manifests, alignment, and spliced LID clips (§4.3(c)).

Splices: word-sized pieces (cut with the MMS forced alignment) from random utterances are
concatenated; at each boundary the language switches with p = 0.5 and the next piece always
comes from a different utterance, so same-language speaker changes are as common as language
changes. Each piece is speed-perturbed (0.9 / 1.0 / 1.1). Frame labels follow the pieces.

  CUDA_VISIBLE_DEVICES=4 python scripts/prepare_fleurs.py --n_splices 6000
"""
import argparse
import csv
import json
import os
import pickle
import random
import sys

import jieba
import librosa
import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from csasr.data import manifest  # noqa: E402
from csasr.lid.align import FRAME, LAB, Aligner, align_rows  # noqa: E402
from csasr.text.normalize import normalize  # noqa: E402
from csasr.text.units import label  # noqa: E402

LANGS = {"en_us": "en", "cmn_hans_cn": "zh"}


def build_manifests(root):
    out = {}
    for d, l in LANGS.items():
        rows = []
        for r in csv.reader(open(os.path.join(root, d, "train.tsv"), encoding="utf-8"), delimiter="\t",
                            quoting=csv.QUOTE_NONE):
            path = os.path.join(root, d, "train", r[1])
            if not os.path.exists(path):
                continue
            ref = normalize(r[2])
            if not ref:
                continue
            info = sf.info(path)
            rows.append(dict(utt_id=f"fleurs_{l}_{r[0]}_{r[1][:-4]}", corpus=f"fleurs_{l}", split="train",
                             audio=os.path.relpath(path, manifest.ROOT), duration=round(info.duration, 3),
                             speaker=r[1], ref_raw=r[2], ref=ref, main=l, mixed=False, sr=info.samplerate))
        p = manifest.path(f"data/fleurs/manifests/{l}_train.jsonl")
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        out[l] = rows
        print(f"fleurs {l}: {len(rows)} utts, {sum(r['duration'] for r in rows) / 3600:.1f} h, "
              f"sample rates {sorted({r['sr'] for r in rows})}", flush=True)
    return out


def load16(p):
    w, sr = sf.read(manifest.path(p), dtype="float32")
    if w.ndim > 1:
        w = w.mean(1)
    return librosa.resample(w, orig_sr=sr, target_sr=16000) if sr != 16000 else w


def pieces_of(row, units, spans):
    """Word-sized pieces: (start, end, lang) per English word / jieba Chinese word."""
    lab = label(row["ref"], already_normalized=True)
    out, i = [], 0
    while i < len(units):
        l = lab.langs[i]
        if l == "zh":
            j = i
            while j < len(units) and lab.langs[j] == "zh":
                j += 1
            k = i
            for w in jieba.lcut("".join(units[i:j])):
                out.append((spans[k][0], spans[k + len(w) - 1][1], "zh")); k += len(w)
            i = j
        else:
            if l == "en":
                out.append((spans[i][0], spans[i][1], "en"))
            i += 1
    return [(s, e, l) for s, e, l in out if 0.08 <= e - s <= 1.5]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/fleurs")
    ap.add_argument("--n_splices", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=425)
    a = ap.parse_args()
    root = manifest.path(a.root)
    rows = build_manifests(root)
    al = Aligner()
    pool = {"en": [], "zh": []}
    for l, rs in rows.items():
        for r, units, spans in align_rows(al, rs, load16):
            if spans:
                pool[l].append((r, pieces_of(r, units, spans)))
    print({l: (len(v), sum(len(p) for _, p in v)) for l, v in pool.items()}, flush=True)

    rng = random.Random(a.seed)
    out_dir = manifest.path("data/fleurs/splices")
    os.makedirs(out_dir, exist_ok=True)
    labels, man = {}, []
    cache = {}

    def audio(r):
        if r["utt_id"] not in cache:
            if len(cache) > 400:
                cache.clear()
            cache[r["utt_id"]] = load16(r["audio"])
        return cache[r["utt_id"]]

    for n in range(a.n_splices):
        target = rng.uniform(3.0, 15.0)
        lang = rng.choice(["en", "zh"])
        wav, lab, t, last = [], [], 0.0, None
        while t < target:
            cands = pool[lang]
            r, ps = rng.choice(cands)
            while (last is not None and r["utt_id"] == last) or not ps:
                r, ps = rng.choice(cands)
            last = r["utt_id"]
            k = rng.randrange(len(ps))
            m = rng.choice([1, 1, 2, 3])  # 1-3 consecutive words from the same utterance
            s, e = ps[k][0], ps[min(len(ps) - 1, k + m - 1)][1]
            seg = audio(r)[int(s * 16000): int(e * 16000)]
            speed = rng.choice([0.9, 1.0, 1.1])
            if speed != 1.0:
                seg = librosa.resample(seg, orig_sr=int(16000 * speed), target_sr=16000)
            gap = np.zeros(int(rng.uniform(0.0, 0.15) * 16000), dtype=np.float32)
            wav += [gap, seg.astype(np.float32)]
            lab += [np.full(int(round(len(gap) / 16000 / FRAME)), LAB["sil"], np.int8),
                    np.full(int(round(len(seg) / 16000 / FRAME)), LAB[lang], np.int8)]
            t += (len(gap) + len(seg)) / 16000
            if rng.random() < 0.5:
                lang = "zh" if lang == "en" else "en"
        w = np.concatenate(wav)
        uid = f"splice_{n:05d}"
        path = os.path.join(out_dir, uid + ".flac")
        sf.write(path, w, 16000, format="FLAC")
        fl = np.concatenate(lab)
        labels[uid] = dict(frames=fl[: int(np.ceil(len(w) / 16000 / FRAME))])
        man.append(dict(utt_id=uid, corpus="fleurs_splice", split="train", audio=os.path.relpath(path, manifest.ROOT),
                        duration=round(len(w) / 16000, 3), ref="", ref_raw="", main="", mixed=True))
    with open(manifest.path("data/fleurs/manifests/splice_train.jsonl"), "w") as f:
        for r in man:
            f.write(json.dumps(r) + "\n")
    os.makedirs(manifest.path("data/align"), exist_ok=True)
    pickle.dump(labels, open(manifest.path("data/align/fleurs_splice_train.pkl"), "wb"))
    print(f"{len(man)} splices, {sum(r['duration'] for r in man) / 3600:.1f} h", flush=True)


if __name__ == "__main__":
    main()
