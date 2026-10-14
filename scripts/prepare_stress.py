#!/usr/bin/env python3
"""Critical-term stress set -> manifest (proposal §3.1). PENDING_HUMAN: recordings.

Each member records every line of stress/sentences.tsv on a phone (16 kHz or higher, mono),
saved as data/stress/audio/<member>/<id>.wav (e.g. data/stress/audio/nibras/zh001.wav).
  python scripts/prepare_stress.py
"""
import csv
import glob
import json
import os
import sys

import librosa
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from csasr.data import manifest  # noqa: E402
from csasr.text.normalize import normalize  # noqa: E402
from csasr.text.units import label, training_target  # noqa: E402


def main():
    sents = {r["id"]: r for r in csv.DictReader(open(manifest.path("stress/sentences.tsv"), encoding="utf-8"),
                                                delimiter="\t")}
    wavs = sorted(glob.glob(manifest.path("data/stress/audio/*/*.wav")))
    if not wavs:
        print("no recordings yet: see PENDING_HUMAN.md")
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


if __name__ == "__main__":
    main()
