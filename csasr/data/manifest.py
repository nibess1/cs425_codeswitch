"""Manifest IO and audio loading shared by training and decoding."""
import json
import os

import numpy as np
import soundfile as sf

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# name -> manifest path (relative to repo root)
SETS = {
    "ascend_train": "data/ascend/manifests/train.jsonl",
    "ascend_val": "data/ascend/manifests/val.jsonl",
    "ascend_test": "data/ascend/manifests/test.jsonl",
    "nsc_train": "data/nsc/manifests/train.jsonl",
    "nsc_val": "data/nsc/manifests/val.jsonl",
    "nsc_test": "data/nsc/manifests/test.jsonl",
    "nsc_diff_test": "data/nsc/manifests/diff_test.jsonl",
    "stress_test": "data/stress/manifests/test.jsonl",
    # D5 synthetic stress set (CosyVoice 2, 4 voices): clean and phone-channel versions
    "stress_tts_clean": "data/stress/manifests/tts_clean.jsonl",
    "stress_tts_phone": "data/stress/manifests/tts_phone.jsonl",
    # section E (Diff Room recording-condition ablation)
    "nsce_diff_train": "data/nsc_e/manifests/diff_train.jsonl",
    "nsce_diff_val": "data/nsc_e/manifests/diff_val.jsonl",
    "nsce_diff_test": "data/nsc_e/manifests/diff_test.jsonl",
    "nsce_mix_train": "data/nsc_e/manifests/mix_train.jsonl",
}
EVAL_TEST = ["ascend_test", "nsc_test", "nsc_diff_test"]
EVAL_VAL = ["ascend_val", "nsc_val"]


def path(p: str) -> str:
    return p if os.path.isabs(p) else os.path.join(ROOT, p)


def load(name_or_path: str, limit: int | None = None) -> list[dict]:
    p = path(SETS.get(name_or_path, name_or_path))
    rows = [json.loads(l) for l in open(p, encoding="utf-8")]
    return rows[:limit] if limit else rows


def load_audio(p: str) -> np.ndarray:
    wav, sr = sf.read(path(p), dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(1)
    assert sr == 16000, (p, sr)
    return wav
