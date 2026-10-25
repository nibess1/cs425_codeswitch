#!/usr/bin/env python3
"""D5 (run in the `cosyvoice` env): synthesize the 120 stress sentences with CosyVoice 2
(zero-shot, 4 voices prompted by FLEURS clips: 2 English + 2 Mandarin speakers, CC-BY 4.0).
-> data/stress/tts/clean/v<k>/<id>.wav (16 kHz) and data/stress/tts/voices.json

  CUDA_VISIBLE_DEVICES=7 ~/micromamba/envs/cosyvoice/bin/python scripts/tts_stress.py
"""
import csv
import json
import os
import random
import sys

import torch
import torchaudio

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COSY = os.path.join(ROOT, "third_party", "CosyVoice")
sys.path += [COSY, os.path.join(COSY, "third_party", "Matcha-TTS")]
from cosyvoice.cli.cosyvoice import CosyVoice2  # noqa: E402
from cosyvoice.utils.file_utils import load_wav  # noqa: E402


def pick_prompts(seed=425):
    """Two English + two Mandarin FLEURS clips of 4-9 s (different speakers / genders when possible)."""
    out = []
    for lang in ("en", "zh"):
        rows = [json.loads(l) for l in open(os.path.join(ROOT, f"data/fleurs/manifests/{lang}_train.jsonl"))]
        tsv = {}
        d = "en_us" if lang == "en" else "cmn_hans_cn"
        for r in csv.reader(open(os.path.join(ROOT, f"data/fleurs/{d}/train.tsv"), encoding="utf-8"), delimiter="\t",
                            quoting=csv.QUOTE_NONE):
            tsv[r[1]] = r[6] if len(r) > 6 else "?"  # gender column
        rows = [r for r in rows if 4 <= r["duration"] <= 9]
        rng = random.Random(seed)
        rng.shuffle(rows)
        got = {}
        for r in rows:
            g = tsv.get(os.path.basename(r["audio"]), "?")
            if g not in got:
                got[g] = r
            if len(got) == 2:
                break
        out += [dict(lang=lang, gender=g, audio=r["audio"], text=r["ref_raw"]) for g, r in got.items()]
    return out


def main():
    model = CosyVoice2(os.path.join(COSY, "pretrained_models/CosyVoice2-0.5B"), load_jit=False, load_trt=False, fp16=False)
    voices = pick_prompts()
    out = os.path.join(ROOT, "data/stress/tts")
    os.makedirs(out, exist_ok=True)
    json.dump(voices, open(os.path.join(out, "voices.json"), "w"), indent=1, ensure_ascii=False)
    sents = list(csv.DictReader(open(os.path.join(ROOT, "stress/sentences.tsv"), encoding="utf-8"), delimiter="\t"))
    for k, v in enumerate(voices):
        prompt = os.path.join(ROOT, v["audio"])  # this CosyVoice 2 version loads the prompt itself (24 kHz)
        d = os.path.join(out, "clean", f"v{k}")
        os.makedirs(d, exist_ok=True)
        for s in sents:
            p = os.path.join(d, f"{s['id']}.wav")
            if os.path.exists(p):
                continue
            chunks = [j["tts_speech"] for j in model.inference_zero_shot(s["text"], v["text"], prompt, stream=False)]
            wav = torch.cat(chunks, dim=1)
            wav = torchaudio.functional.resample(wav, model.sample_rate, 16000)
            torchaudio.save(p, wav, 16000)
        print(f"voice {k} ({v['lang']}, {v['gender']}): {len(sents)} sentences", flush=True)


if __name__ == "__main__":
    main()
