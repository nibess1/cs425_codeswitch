#!/usr/bin/env python3
"""B2: deployed mixed precision. whisper.cpp Q4_0 with the blocks chosen by the A6 sensitivity
scan (results/sensitivity_<tag>_nf4_bnb.json, chosen on validation) kept at F16, via the
patched whisper-quantize (env WHISPER_QUANT_KEEP). Evaluated on the fixed CPU subset.

  python scripts/deploy_mixed.py --tag ft_tiny --method ft --workers 16
"""
import argparse
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from csasr.data import manifest  # noqa: E402
from csasr.quant import export as X  # noqa: E402
from deploy_eval import run, subset  # noqa: E402


def ggml_regex(block: str) -> str:
    """model.encoder.layers.3 -> encoder.blocks.3\\..* (whisper.cpp tensor names)."""
    side, _, idx = block.replace("model.", "").split(".")
    return f"{side}.blocks.{idx}\\..*"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--method", required=True)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--subset", default="ascend_test:0,nsc_test:4000,nsc_diff_test:2000")
    a = ap.parse_args()
    sens = json.load(open(manifest.path(f"results/sensitivity_{a.tag}_nf4_bnb.json")))
    k, keep = sens["chosen"]["k"], sens["chosen"]["keep16"]
    d = manifest.path(f"ckpt/export/{a.tag}_{a.method}/ggml")
    f16 = os.path.join(d, "ggml-model.bin")
    out_bin = os.path.join(d, f"ggml-model-q4_0-mp{k}.bin")
    if not os.path.exists(out_bin):
        env = dict(os.environ, WHISPER_QUANT_KEEP=",".join(ggml_regex(b) for b in keep))
        r = subprocess.run([os.path.join(X.WCPP, "build", "bin", "whisper-quantize"), f16, out_bin, "q4_0"],
                           env=env, capture_output=True, text=True)
        if r.returncode:
            raise RuntimeError(r.stderr[-2000:])
    fmt = f"ggml_q4_0_mp{k}"
    name = f"{a.tag}__{fmt}__{a.method}"
    if os.path.exists(manifest.path(f"preds/{name}.meta.json")):
        return
    meta = dict(model=a.tag, precision=fmt, method=a.method, kept_f16=keep, file_mb=round(os.path.getsize(out_bin) / 2**20, 1),
                cpu_seconds={})
    with open(manifest.path(f"preds/{name}.jsonl"), "w", encoding="utf-8") as f:
        for s in manifest.EVAL_TEST:
            rows = subset(manifest.load(s), s, a.subset)
            res, cpu_s = run("ggml_q4_0", rows, out_bin, a.workers, a.threads)
            meta["cpu_seconds"][s] = round(cpu_s, 1)
            for r, (hyp, lang) in zip(rows, res):
                f.write(json.dumps(dict(utt_id=r["utt_id"], corpus=r["corpus"], split=r["split"], set=s, model=a.tag,
                                        precision=fmt, method=a.method, ref=r["ref_raw"], hyp=hyp, lang_token=lang),
                                   ensure_ascii=False) + "\n")
            print(f"{name} {s}: {len(rows)} utts", flush=True)
    json.dump(meta, open(manifest.path(f"preds/{name}.meta.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
