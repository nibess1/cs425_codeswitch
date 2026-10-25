#!/usr/bin/env python3
"""Deployed-format accuracy on CPU: export a checkpoint to CTranslate2 int8 and whisper.cpp
Q8_0/Q5_0/Q4_0, decode the test sets, write preds/<tag>__<format>__<method>.jsonl.

  python scripts/deploy_eval.py --model ckpt/ft_small/best --tag ft_small --method ft --workers 16
"""
import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from csasr.data import manifest  # noqa: E402
from csasr.quant import export as X  # noqa: E402


def _ct2_job(args):
    rows, d, threads = args
    return X.ct2_transcribe(rows, d, threads)


def _ggml_job(args):
    rows, b, threads = args
    return X.ggml_transcribe(rows, b, threads)


def subset(rows, name, spec):
    n = dict(x.split(":") for x in spec.split(",")).get(name, "0")
    n = int(n)
    if not n or n >= len(rows):
        return rows
    import random
    keep = set(random.Random(425).sample(range(len(rows)), n))
    return [r for i, r in enumerate(rows) if i in keep]


def run(fmt, rows, target, workers, threads):
    parts = [rows[i::workers] for i in range(workers)]
    fn = _ct2_job if fmt.startswith("ct2") else _ggml_job
    with ProcessPoolExecutor(workers) as ex:
        res = list(ex.map(fn, [(p, target, threads) for p in parts]))
    out = [None] * len(rows)
    for w, (pairs, _) in enumerate(res):
        for k, pr in enumerate(pairs):
            out[w + k * workers] = pr
    return out, sum(t for _, t in res)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--method", required=True)
    ap.add_argument("--formats", nargs="+", default=["ct2_int8", "ggml_q8_0", "ggml_q5_0", "ggml_q4_0"],
                    help="also: ct2_f32, ggml_f16 (same-backend 16-bit references for R)")
    ap.add_argument("--sets", nargs="+", default=manifest.EVAL_TEST)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--export_dir", default=None, help="reuse ckpt/export/<this> (default <tag>_<method>)")
    ap.add_argument("--subset", default="ascend_test:0,nsc_test:4000,nsc_diff_test:2000",
                    help="set:n (0 = all) seeded random subset per set; R compares the same utterances")
    a = ap.parse_args()
    exp = manifest.path(f"ckpt/export/{a.export_dir or a.tag + '_' + a.method}")
    targets = {}
    if "ct2_int8" in a.formats:
        targets["ct2_int8"] = X.to_ct2(a.model, os.path.join(exp, "ct2_int8"))
    if "ct2_f32" in a.formats:
        targets["ct2_f32"] = X.to_ct2(a.model, os.path.join(exp, "ct2_f32"), quant="float16")
    g = [f.split("_", 1)[1] for f in a.formats if f.startswith("ggml_")]
    if g:
        bins = X.to_ggml(a.model, os.path.join(exp, "ggml"), [q for q in g if q != "f16"])
        targets.update({f"ggml_{q}": bins[q] for q in g})
    for fmt, tgt in targets.items():
        name = f"{a.tag}__{fmt}__{a.method}"
        if os.path.exists(manifest.path(f"preds/{name}.meta.json")):
            continue
        meta = dict(model=a.tag, precision=fmt, method=a.method, file_mb={}, cpu_seconds={})
        size = (sum(os.path.getsize(os.path.join(tgt, f)) for f in os.listdir(tgt)) if os.path.isdir(tgt)
                else os.path.getsize(tgt))
        meta["file_mb"] = round(size / 2**20, 1)
        with open(manifest.path(f"preds/{name}.jsonl"), "w", encoding="utf-8") as f:
            for s in a.sets:
                rows = subset(manifest.load(s), s, a.subset)
                res, cpu_s = run(fmt, rows, tgt, a.workers, a.threads)
                meta["cpu_seconds"][s] = round(cpu_s, 1)
                for r, (hyp, lang) in zip(rows, res):
                    f.write(json.dumps(dict(utt_id=r["utt_id"], corpus=r["corpus"], split=r["split"], set=s,
                                            model=a.tag, precision=fmt, method=a.method, ref=r["ref_raw"], hyp=hyp,
                                            lang_token=lang), ensure_ascii=False) + "\n")
                print(f"{name} {s}: {len(rows)} utts", flush=True)
        json.dump(meta, open(manifest.path(f"preds/{name}.meta.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
