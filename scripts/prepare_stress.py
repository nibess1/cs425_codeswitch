#!/usr/bin/env python3
"""Critical-term stress set -> manifests (proposal §3.1).

--source human (default): the team's phone recordings (stress/assignments.tsv says who read which
sentence), converted to WAV as data/stress/audio/<reader>/<id>.wav (e.g. data/stress/audio/phoebe/zh001.wav)
-> data/stress/clips/ and data/stress/manifests/test.jsonl.

--source tts (D5, SYNTHETIC data): phone-channel version of the CosyVoice clips made by tts_stress.py.
phone = band-limit 300-3400 Hz (4th-order Butterworth), down to 8 kHz and back to 16 kHz,
convolve with a synthetic room impulse response (exponential decay, RT60 0.3 s), add white
noise at 20 dB SNR. -> data/stress/tts/phone/v<k>/<id>.wav and
data/stress/manifests/{tts_clean,tts_phone}.jsonl; also writes stress/assignments.tsv (30 sentences
per member) if it does not exist yet (the team later re-assigned them: 3 readers x 40).

  python scripts/prepare_stress.py [--source tts]
"""
import argparse
import csv
import glob
import json
import os
import random
import sys

import librosa
import numpy as np
import soundfile as sf
from scipy.signal import butter, fftconvolve, resample_poly, sosfilt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from csasr.data import manifest  # noqa: E402
from csasr.text.normalize import normalize  # noqa: E402
from csasr.text.units import label, training_target  # noqa: E402

MEMBERS = ["rui_cong", "phoebe", "nibras", "bridgette"]


def human():
    sents = {r["id"]: r for r in csv.DictReader(open(manifest.path("stress/sentences.tsv"), encoding="utf-8"),
                                                delimiter="\t")}
    wavs = sorted(glob.glob(manifest.path("data/stress/audio/*/*.wav")))
    if not wavs:
        print("no recordings in data/stress/audio/<reader>/<id>.wav")
        return
    out_dir = manifest.path("data/stress/clips")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(manifest.path("data/stress/manifests"), exist_ok=True)
    with open(manifest.path("data/stress/manifests/test.jsonl"), "w", encoding="utf-8") as f:
        for w in wavs:
            spk, sid = w.split(os.sep)[-2], os.path.basename(w)[:-4]
            if sid not in sents:
                continue
            x, sr = sf.read(w, dtype="float32")
            x = x.mean(1) if x.ndim > 1 else x
            if sr != 16000:
                x = librosa.resample(x, orig_sr=sr, target_sr=16000)
            p = os.path.join(out_dir, f"stress_{spk}_{sid}.flac")
            sf.write(p, x, 16000, format="FLAC")
            raw = sents[sid]["text"]
            lab = label(raw)
            f.write(json.dumps(dict(utt_id=f"stress_{spk}_{sid}", corpus="stress", split="test",
                                    audio=os.path.relpath(p, manifest.ROOT), duration=round(len(x) / 16000, 3),
                                    speaker=spk, ref_raw=raw, ref=normalize(raw), target=training_target(raw),
                                    main=lab.main, mixed=lab.mixed), ensure_ascii=False) + "\n")
    print("stress manifest written")



def phone(x, rng, sr=16000):
    sos = butter(4, [300, 3400], btype="band", fs=sr, output="sos")
    y = sosfilt(sos, x)
    y = resample_poly(resample_poly(y, 1, 2), 2, 1)[: len(x)]
    t = np.arange(int(0.3 * sr)) / sr
    rir = rng.standard_normal(len(t)) * np.exp(-6.9 * t / 0.3)
    rir[0] = 1.0
    y = fftconvolve(y, rir / np.abs(rir).sum() * 4)[: len(x)]
    p = np.mean(y ** 2) + 1e-12
    y = y + rng.standard_normal(len(y)) * np.sqrt(p / 10 ** (20 / 10))
    return (y / (np.abs(y).max() + 1e-9) * 0.9).astype(np.float32)


def tts():
    root = manifest.path("data/stress/tts")
    voices = json.load(open(os.path.join(root, "voices.json")))
    sents = {r["id"]: r for r in csv.DictReader(open(manifest.path("stress/sentences.tsv"), encoding="utf-8"), delimiter="\t")}
    rng = np.random.default_rng(425)
    os.makedirs(manifest.path("data/stress/manifests"), exist_ok=True)
    rows = {"tts_clean": [], "tts_phone": []}
    for p in sorted(glob.glob(os.path.join(root, "clean", "v*", "*.wav"))):
        v, sid = p.split(os.sep)[-2], os.path.basename(p)[:-4]
        x, sr = sf.read(p, dtype="float32")
        q = p.replace(os.sep + "clean" + os.sep, os.sep + "phone" + os.sep)
        os.makedirs(os.path.dirname(q), exist_ok=True)
        if not os.path.exists(q):
            sf.write(q, phone(x, rng), 16000)
        raw = sents[sid]["text"]
        lab = label(raw)
        vi = voices[int(v[1:])]
        for kind, path in (("tts_clean", p), ("tts_phone", q)):
            rows[kind].append(dict(utt_id=f"{kind}_{v}_{sid}", corpus=f"stress_{kind}", split="test",
                                   audio=os.path.relpath(path, manifest.ROOT), duration=round(len(x) / 16000, 3),
                                   speaker=f"tts_{v}_{vi['lang']}_{vi['gender']}", ref_raw=raw, ref=normalize(raw),
                                   target=training_target(raw), main=lab.main, mixed=lab.mixed, sentence_id=sid))
    for kind, rs in rows.items():
        with open(manifest.path(f"data/stress/manifests/{kind}.jsonl"), "w", encoding="utf-8") as f:
            for r in rs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(kind, len(rs), "clips", round(sum(r["duration"] for r in rs) / 60, 1), "min")
    if os.path.exists(manifest.path("stress/assignments.tsv")):  # never overwrite the team's assignment
        return
    ids = sorted(sents)
    random.Random(425).shuffle(ids)
    with open(manifest.path("stress/assignments.tsv"), "w", encoding="utf-8") as f:
        f.write("member\tsentence_id\ttext\n")
        for m, member in enumerate(MEMBERS):
            for sid in sorted(ids[30 * m: 30 * (m + 1)]):
                f.write(f"{member}\t{sid}\t{sents[sid]['text']}\n")
    print("stress/assignments.tsv: 4 members x 30 sentences")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["human", "tts"], default="human")
    human() if ap.parse_args().source == "human" else tts()


if __name__ == "__main__":
    main()
