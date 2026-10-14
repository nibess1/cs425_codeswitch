"""Deployed CPU formats (proposal §4.4): CTranslate2 int8 and whisper.cpp Q8_0 / Q5_0 / Q4_0.

Both use quantization schemes different from bitsandbytes, so accuracy is re-measured for
every format. CTranslate2 decoding mirrors the HF path exactly (language token restricted
to {zh, en}, greedy, 128-token cap). whisper.cpp has no restricted language detection, so it
runs with `-l auto`, no context conditioning, greedy.
"""
import os
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np
import soundfile as sf

from ..data.manifest import ROOT, load_audio, path

THIRD = os.path.join(ROOT, "third_party")
WCPP = os.environ.get("CSASR_WHISPERCPP", os.path.join(THIRD, "whisper.cpp"))
OPENAI = os.path.join(THIRD, "openai-whisper")


def hf_dir(model: str) -> str:
    """Local HF checkpoint dir for tiny/base/small or a ckpt path."""
    if os.path.isdir(path(model)):
        return path(model)
    from huggingface_hub import snapshot_download
    return snapshot_download(f"openai/whisper-{model}", allow_patterns=["*.json", "*.txt", "model.safetensors"])


def to_ct2(model: str, out_dir: str, quant: str = "int8") -> str:
    out = path(out_dir)
    if os.path.exists(os.path.join(out, "model.bin")):
        return out
    src = hf_dir(model)
    tmp = tempfile.mkdtemp()
    # the converter needs tokenizer + preprocessor files next to the weights
    for f in os.listdir(src):
        os.symlink(os.path.join(src, f), os.path.join(tmp, f))
    from transformers import WhisperFeatureExtractor, WhisperTokenizer
    if not os.path.exists(os.path.join(tmp, "tokenizer.json")):
        WhisperTokenizer.from_pretrained("openai/whisper-small").save_pretrained(tmp)
    if not os.path.exists(os.path.join(tmp, "preprocessor_config.json")):
        WhisperFeatureExtractor.from_pretrained("openai/whisper-small").save_pretrained(tmp)
    subprocess.run([os.path.join(os.path.dirname(sys.executable), "ct2-transformers-converter"), "--model", tmp, "--output_dir", out, "--quantization", quant,
                    "--force"], check=True, capture_output=True)
    shutil.rmtree(tmp, ignore_errors=True)
    return out


def to_ggml(model: str, out_dir: str, quants=("q8_0", "q5_0", "q4_0")) -> dict:
    out = path(out_dir)
    os.makedirs(out, exist_ok=True)
    f16 = os.path.join(out, "ggml-model.bin")
    if not os.path.exists(f16):
        subprocess.run([sys.executable, os.path.join(WCPP, "models", "convert-h5-to-ggml.py"), hf_dir(model), OPENAI, out],
                       check=True, capture_output=True)
    q = os.path.join(WCPP, "build", "bin", "whisper-quantize")
    res = {"f16": f16}
    for t in quants:
        p = os.path.join(out, f"ggml-model-{t}.bin")
        if not os.path.exists(p):
            subprocess.run([q, f16, p, t], check=True, capture_output=True)
        res[t] = p
    return res


class CT2Decoder:
    def __init__(self, ct2_dir, threads=4, compute_type="int8"):
        import ctranslate2
        from transformers import WhisperFeatureExtractor, WhisperTokenizer
        self.m = ctranslate2.models.Whisper(ct2_dir, device="cpu", compute_type=compute_type, intra_threads=threads)
        self.tok = WhisperTokenizer.from_pretrained("openai/whisper-small")
        self.fe = WhisperFeatureExtractor.from_pretrained("openai/whisper-small")
        t = self.tok.convert_tokens_to_ids
        self.sot, self.trans, self.notime = t("<|startoftranscript|>"), t("<|transcribe|>"), t("<|notimestamps|>")
        self.lang = {"zh": t("<|zh|>"), "en": t("<|en|>")}

    def encode_features(self, wavs):
        import ctranslate2
        f = np.stack([self.fe(w, sampling_rate=16000, return_tensors="np").input_features[0] for w in wavs])
        return ctranslate2.StorageView.from_array(f.astype(np.float32))

    def transcribe_batch(self, wavs):
        feats = self.encode_features(wavs)
        det = self.m.detect_language(feats)
        langs = []
        for d in det:
            p = dict(d)
            langs.append("zh" if p.get("<|zh|>", 0) >= p.get("<|en|>", 0) else "en")
        prompts = [[self.sot, self.lang[l], self.trans, self.notime] for l in langs]
        res = self.m.generate(feats, prompts, beam_size=1, max_length=128 + 4, return_scores=False,
                              suppress_blank=True)
        return [self.tok.decode(r.sequences_ids[0], skip_special_tokens=True).strip() for r in res], langs


def ct2_transcribe(rows, ct2_dir, threads=4, batch=8):
    dec = CT2Decoder(ct2_dir, threads)
    out = []
    t0 = time.time()
    for b in range(0, len(rows), batch):
        wavs = [load_audio(r["audio"]) for r in rows[b: b + batch]]
        texts, langs = dec.transcribe_batch(wavs)
        out += list(zip(texts, langs))
    return out, time.time() - t0


def ggml_transcribe(rows, model_bin, threads=4, chunk=200):
    """Greedy whisper.cpp decoding of many clips per process (model loaded once per chunk)."""
    cli = os.path.join(WCPP, "build", "bin", "whisper-cli")
    out = []
    t_total = 0.0
    for c in range(0, len(rows), chunk):
        part = rows[c: c + chunk]
        tmp = tempfile.mkdtemp()
        files = []
        for i, r in enumerate(part):
            f = os.path.join(tmp, f"{i:05d}.wav")
            sf.write(f, load_audio(r["audio"]), 16000, subtype="PCM_16")
            files.append(f)
        cmd = [cli, "-m", model_bin, "-t", str(threads), "-l", "auto", "-nt", "-otxt", "-bs", "1", "-bo", "1",
               "-mc", "0", "-np"]
        for f in files:
            cmd += ["-f", f]
        t0 = time.time()
        subprocess.run(cmd, check=True, capture_output=True)
        t_total += time.time() - t0
        for f in files:
            p = f + ".txt"
            out.append((open(p, encoding="utf-8", errors="replace").read().strip().replace("\n", " ")
                        if os.path.exists(p) else "", ""))
        shutil.rmtree(tmp, ignore_errors=True)
    return out, t_total
