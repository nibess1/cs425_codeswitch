#!/usr/bin/env python3
"""Sensitivity scan + mixed-precision selection (proposal §4.4, A6).

1. Rank blocks by KL on embedded-word tokens (only that block quantized, teacher forced, val).
2. Confirm the top 5 with embedded-unit error on val (decode with only that block quantized).
3. Keep the top-k blocks at 16-bit (k <= 1/4 of all blocks); choose k on val embedded error.

  CUDA_VISIBLE_DEVICES=4 python scripts/sensitivity.py --model ckpt/ft_small/best --tag ft_small
"""
import argparse
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from csasr.data import manifest  # noqa: E402
from csasr.decode.whisper_hf import load_processor  # noqa: E402
from csasr.quant.sensitivity import scan  # noqa: E402
from csasr.score import report as R  # noqa: E402

PY = sys.executable


def decode(model, tag, precision, method, sets, skip):
    name = f"{tag}__{precision}__{method}"
    if not os.path.exists(manifest.path(f"preds/{name}.meta.json")):
        cmd = [PY, manifest.path("scripts/decode.py"), "--model", model, "--model_tag", tag, "--precision", precision,
               "--lang", "free", "--method", method, "--sets", *sets, "--batch", "48"]
        if skip:
            cmd += ["--skip_modules", *skip]
        subprocess.run(cmd, check=True)
    return name


def val_emb_error(name):
    # code-switched (non-balanced) utterances of both val sets pooled
    st = R.stats(name, use_labse=False)
    pooled = [x for s in manifest.EVAL_VAL for x in st[s][1]]
    from csasr.score.metrics import corpus_metrics
    m = corpus_metrics(pooled)
    return m["cs_E_emb"], m["MER"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--precision", default="nf4_bnb")
    a = ap.parse_args()
    out = manifest.path(f"results/sensitivity_{a.tag}_{a.precision}.json")
    tok, fe = load_processor("small")
    if os.path.exists(out):
        res = json.load(open(out))
    else:
        rows = [r for s in manifest.EVAL_VAL for r in manifest.load(s) if r["mixed"] and r["main"] in ("zh", "en")]
        kl = scan(manifest.path(a.model), rows, tok, fe, a.precision)
        res = {"kl": kl}
        json.dump(res, open(out, "w"), indent=1)
    kl = [(g, s) for g, s in res["kl"] if g != "ALL"]
    blocks = [g for g, _ in kl]
    ranked = sorted(kl, key=lambda x: -x[1])
    vals = manifest.EVAL_VAL
    # baselines on val: all-16-bit and all-quantized
    base16 = decode(a.model, a.tag, "fp16", "val", vals, None)
    baseq = decode(a.model, a.tag, a.precision, "val", vals, None)
    res["val_fp16"] = val_emb_error(base16)
    res["val_quant"] = val_emb_error(baseq)
    # 2) confirm top 5: only that block quantized
    res["confirm_top5"] = []
    for g, s in ranked[:5]:
        n = decode(a.model, a.tag, a.precision, f"val_only_{g.replace('model.', '').replace('.', '_')}", vals,
                   [b for b in blocks if b != g])
        e, m = val_emb_error(n)
        res["confirm_top5"].append(dict(block=g, kl=s, val_cs_E_emb=e, val_MER=m))
    # 3) mixed precision: top-k at 16-bit
    kmax = len(blocks) // 4
    ks = sorted({1, 2, max(1, kmax // 2), kmax})
    res["mixed"] = []
    for k in ks:
        keep = [g for g, _ in ranked[:k]]
        n = decode(a.model, a.tag, a.precision, f"val_mp{k}", vals, keep)
        e, m = val_emb_error(n)
        res["mixed"].append(dict(k=k, keep16=keep, val_cs_E_emb=e, val_MER=m))
    best = min(res["mixed"], key=lambda x: (x["val_cs_E_emb"], x["k"]))
    res["chosen"] = best
    json.dump(res, open(out, "w"), indent=1)
    # test decode with the chosen mixed-precision configuration
    decode(a.model, a.tag, a.precision, f"mp{best['k']}", manifest.EVAL_TEST, best["keep16"])
    print(json.dumps(res["chosen"]))


if __name__ == "__main__":
    main()
