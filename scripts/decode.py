#!/usr/bin/env python3
"""Decode eval sets with a (possibly quantized) Whisper model -> preds/<name>.jsonl.

  CUDA_VISIBLE_DEVICES=4 python scripts/decode.py --model small --precision fp16 --lang free \
      --method offshelf_free --sets ascend_test nsc_test nsc_diff_test
"""
import argparse
import json
import os
import subprocess
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from csasr.data import manifest  # noqa: E402
from csasr.decode.whisper_hf import Decoder, load_model, load_processor, quantized_modules  # noqa: E402


def git_hash():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=manifest.ROOT, text=True).strip()
    except Exception:
        return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="tiny|base|small or a checkpoint dir")
    ap.add_argument("--model_tag", default=None, help="name used in outputs (default: --model)")
    ap.add_argument("--processor", default=None, help="tokenizer/feature extractor source (default: --model)")
    ap.add_argument("--precision", default="fp16")
    ap.add_argument("--lang", default="free", choices=["free", "zh", "en", "both"])
    ap.add_argument("--method", required=True)
    ap.add_argument("--sets", nargs="+", default=manifest.EVAL_TEST)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--beams", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--skip_modules", nargs="*", default=None, help="modules kept 16-bit (mixed precision)")
    ap.add_argument("--adapter", default=None, help="QLoRA adapter dir (base --model loaded in NF4)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    if a.precision == "int8_bnb":
        # LLM.int8 has a large fixed cost per matmul call, so fewer, larger decoding steps help a lot
        a.batch = max(a.batch, 160)
    tag = a.model_tag or a.model
    name = a.out or f"{tag}__{a.precision}__{a.method}"
    out_path = manifest.path(f"preds/{name}.jsonl")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.cuda.reset_peak_memory_stats()
    if a.adapter:
        from csasr.quant.qat import load_qlora_for_eval
        model = load_qlora_for_eval(a.model, a.adapter)
    else:
        model = load_model(manifest.path(a.model) if os.path.isdir(manifest.path(a.model)) else a.model,
                           a.precision, skip_modules=a.skip_modules)
    tok, fe = load_processor(a.processor or "small")  # all multilingual sizes share these
    meta = dict(model=tag, precision=a.precision, method=a.method, lang=a.lang, beams=a.beams,
                footprint_mb=round(model.get_memory_footprint() / 2**20, 1),
                gpu_alloc_after_load_mb=round(torch.cuda.memory_allocated() / 2**20, 1),
                quantized_modules=quantized_modules(model) if ("bnb" in a.precision or a.adapter) else [],
                adapter=a.adapter,
                skip_modules=a.skip_modules or [], git=git_hash(), sets=a.sets)
    dec = Decoder(model, tok, fe)
    t0 = time.time()
    with open(out_path, "w", encoding="utf-8") as f:
        for s in a.sets:
            rows = manifest.load(s, a.limit)
            ts = time.time()
            res = dec.transcribe(rows, lang_mode=a.lang, batch_size=a.batch, num_beams=a.beams)
            audio_s = sum(r["duration"] for r in rows)
            meta.setdefault("rtf", {})[s] = round((time.time() - ts) / audio_s, 5)
            for r, (hyp, lang) in zip(rows, res):
                f.write(json.dumps(dict(utt_id=r["utt_id"], corpus=r["corpus"], split=r["split"], set=s,
                                        model=tag, precision=a.precision, method=a.method, ref=r["ref_raw"],
                                        hyp=hyp, lang_token=lang), ensure_ascii=False) + "\n")
            print(f"{name} {s}: {len(rows)} utts in {time.time() - ts:.0f}s", flush=True)
    meta["peak_gpu_mb"] = round(torch.cuda.max_memory_allocated() / 2**20, 1)
    meta["wall_s"] = round(time.time() - t0, 1)
    meta["quantized_modules_n"] = len(meta["quantized_modules"])
    json.dump(meta, open(out_path.replace(".jsonl", ".meta.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
