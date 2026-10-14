#!/usr/bin/env python3
"""ASCEND (CAiRE/ASCEND parquet) -> 16 kHz FLAC clips + JSONL manifests, official splits.

  python scripts/prepare_ascend.py --hf data/ascend/hf --out data/ascend
"""
import argparse
import glob
import io
import json
import os
import sys

import librosa
import numpy as np
import pandas as pd
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from csasr.text.normalize import normalize  # noqa: E402
from csasr.text.units import label, training_target  # noqa: E402

SPLITS = {"train": "train", "validation": "val", "test": "test"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf", default="data/ascend/hf")
    ap.add_argument("--out", default="data/ascend")
    a = ap.parse_args()
    os.makedirs(os.path.join(a.out, "manifests"), exist_ok=True)
    speakers = {}
    for hf_split, split in SPLITS.items():
        files = sorted(glob.glob(os.path.join(a.hf, "main", f"{hf_split}-*.parquet")))
        df = pd.concat([pd.read_parquet(f) for f in files])
        clip_dir = os.path.join(a.out, "clips", split)
        os.makedirs(clip_dir, exist_ok=True)
        n_drop = 0
        with open(os.path.join(a.out, "manifests", f"{split}.jsonl"), "w") as f:
            for r in df.itertuples():
                ref = normalize(r.transcription)
                if not ref:
                    n_drop += 1
                    continue
                path = os.path.join(clip_dir, f"ascend_{r.id}.flac")
                if not os.path.exists(path):
                    wav, sr = sf.read(io.BytesIO(r.audio["bytes"]), dtype="float32")
                    if wav.ndim > 1:
                        wav = wav.mean(1)
                    if sr != 16000:
                        wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)
                    sf.write(path, wav, 16000, format="FLAC")
                lab = label(ref, already_normalized=True)
                speakers.setdefault(split, set()).add(int(r.original_speaker_id))
                f.write(json.dumps(dict(
                    utt_id=f"ascend_{r.id}", corpus="ascend", split=split, audio=path,
                    duration=round(float(r.duration), 3), speaker=str(r.original_speaker_id),
                    session=str(r.session_id), ref_raw=r.transcription, ref=ref,
                    target=training_target(r.transcription), main=lab.main, mixed=lab.mixed,
                    ascend_language=r.language, topic=r.topic), ensure_ascii=False) + "\n")
        print(f"{split}: {len(df) - n_drop} utts ({n_drop} empty after normalization dropped), "
              f"{df.duration.sum() / 3600:.2f} h", flush=True)
    s = speakers
    print("speaker overlap train/test:", len(s["train"] & s["test"]), "of", len(s["test"]),
          "| train/val:", len(s["train"] & s["val"]), "of", len(s["val"]))


if __name__ == "__main__":
    main()
