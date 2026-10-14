#!/usr/bin/env python3
"""CPU speed on the MacBook (proposal §4.4): real-time factor, peak memory, file size.

Runs every configuration in a fresh process with a fixed thread count (4) on a fixed clip
set, and reports RTF = processing time / audio length and peak RSS.

  python scripts/speed_bench.py --clips speed/clips --models speed/models --out results/speed.json
    models/<tag>/ct2_int8/            CTranslate2 model dir
    models/<tag>/ggml-model-q*.bin    whisper.cpp models
    models/<tag>/lid_head.npz         LID head weights (optional; adds the head + re-ranking costs)
"""
import argparse
import glob
import json
import os
import re
import resource
import subprocess
import sys
import time

THREADS = 4


def child_ct2(model_dir, clips, mode):
    import ctranslate2
    import numpy as np
    import soundfile as sf
    from transformers import WhisperFeatureExtractor, WhisperTokenizer
    fe = WhisperFeatureExtractor.from_pretrained("openai/whisper-small")
    tok = WhisperTokenizer.from_pretrained("openai/whisper-small")
    t = tok.convert_tokens_to_ids
    sot, trans, notime = t("<|startoftranscript|>"), t("<|transcribe|>"), t("<|notimestamps|>")
    lang = {"zh": t("<|zh|>"), "en": t("<|en|>")}
    m = ctranslate2.models.Whisper(model_dir, device="cpu", compute_type="int8", intra_threads=THREADS)
    head = None
    if mode != "plain":
        h = np.load(os.path.join(os.path.dirname(model_dir.rstrip("/")), "lid_head.npz"))
        head = {k: h[k] for k in h.files}
    wavs = [sf.read(c, dtype="float32")[0] for c in clips]
    t0 = time.time()
    for w in wavs:
        f = fe(w, sampling_rate=16000, return_tensors="np").input_features.astype(np.float32)
        x = ctranslate2.StorageView.from_array(f)
        det = dict(m.detect_language(x)[0])
        l = "zh" if det.get("<|zh|>", 0) >= det.get("<|en|>", 0) else "en"
        if mode == "plain":
            m.generate(x, [[sot, lang[l], trans, notime]], beam_size=1, max_length=132)
            continue
        # LID head on the encoder output (separate module), then on-device re-ranking:
        # beam-4 x 4 hypotheses under zh / en / predicted prefix + cross-attention alignment
        enc = np.array(m.encode(x, to_cpu=True))[0]
        z = (enc - enc.mean(-1, keepdims=True)) / np.sqrt(enc.var(-1, keepdims=True) + 1e-5)
        z = z * head["ln_w"] + head["ln_b"]
        hdn = z @ head["w1"].T + head["b1"]
        hdn = 0.5 * hdn * (1 + np.tanh(0.79788456 * (hdn + 0.044715 * hdn ** 3)))
        _ = hdn @ head["w2"].T + head["b2"]
        if mode == "lid":
            m.generate(x, [[sot, lang[l], trans, notime]], beam_size=1, max_length=132)
            continue
        hyps = []
        for p in ("zh", "en", l):
            r = m.generate(x, [[sot, lang[p], trans, notime]], beam_size=4, num_hypotheses=4, max_length=132)
            hyps += r[0].sequences_ids
        n = int(np.ceil(len(w) / 16000 / 0.02))
        for ids in hyps:
            if ids:
                m.align(x, [sot, lang[l], trans, notime], [ids], [min(n, 1500)])
    dt = time.time() - t0
    return dt


def run_child(kind, target, clips, mode="plain"):
    """Run one configuration in a fresh process; returns (seconds, peak RSS MB)."""
    if kind == "ggml":
        cli = os.environ["WHISPER_CLI"]
        cmd = ["/usr/bin/time", "-l", cli, "-m", target, "-t", str(THREADS), "-l", "auto", "-nt", "-bs", "1",
               "-bo", "1", "-mc", "0", "-np"]
        for c in clips:
            cmd += ["-f", c]
        t0 = time.time()
        p = subprocess.run(cmd, capture_output=True, text=True)
        dt = time.time() - t0
        m = re.search(r"(\d+)\s+maximum resident set size", p.stderr)
        return dt, int(m.group(1)) / 2**20 if m else None
    code = (f"import sys,json,resource; sys.path.insert(0,{os.path.dirname(os.path.abspath(__file__))!r});"
            f"from speed_bench import child_ct2; dt=child_ct2({target!r},{clips!r},{mode!r});"
            f"print(json.dumps([dt, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20]))")
    p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    if p.returncode:
        raise RuntimeError(p.stderr[-2000:])
    dt, rss = json.loads(p.stdout.strip().splitlines()[-1])
    return dt, rss


def size_mb(p):
    if os.path.isdir(p):
        return sum(os.path.getsize(os.path.join(p, f)) for f in os.listdir(p)) / 2**20
    return os.path.getsize(p) / 2**20


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", required=True)
    ap.add_argument("--models", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import soundfile as sf
    clips = sorted(glob.glob(os.path.join(a.clips, "*.wav")))
    audio_s = sum(sf.info(c).duration for c in clips)
    res = json.load(open(a.out)) if os.path.exists(a.out) else []
    done = {(r["tag"], r["format"], r["mode"]) for r in res}
    for tag in sorted(os.listdir(a.models)):
        d = os.path.join(a.models, tag)
        confs = []
        if os.path.isdir(os.path.join(d, "ct2_int8")):
            confs.append(("ct2", os.path.join(d, "ct2_int8"), "ct2_int8", "plain"))
            if os.path.exists(os.path.join(d, "lid_head.npz")):
                confs += [("ct2", os.path.join(d, "ct2_int8"), "ct2_int8", "lid"),
                          ("ct2", os.path.join(d, "ct2_int8"), "ct2_int8", "rerank_ondevice")]
        for b in sorted(glob.glob(os.path.join(d, "ggml-model*.bin"))):
            q = os.path.basename(b)[len("ggml-model"):-4].strip("-") or "f16"
            confs.append(("ggml", b, f"ggml_{q}", "plain"))
        for kind, target, fmt, mode in confs:
            if (tag, fmt, mode) in done:
                continue
            run_child(kind, target, clips[:2], mode)  # warm-up (file cache)
            dt, rss = run_child(kind, target, clips, mode)
            r = dict(tag=tag, format=fmt, mode=mode, threads=THREADS, audio_s=round(audio_s, 1),
                     seconds=round(dt, 2), rtf=round(dt / audio_s, 4), peak_rss_mb=round(rss, 1) if rss else None,
                     file_mb=round(size_mb(target), 1), n_clips=len(clips))
            print(json.dumps(r), flush=True)
            res.append(r)
            json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
