#!/usr/bin/env python3
"""Small public sample of ASCEND and NSC Part 4 (about 20 MB of FLAC) so the code runs without the corpora.

Copies a fixed random subset (seed 0, clips of 1-10 s, mixed and monolingual in the corpus's own
proportion) of each split into sample/data/<corpus>/clips/ and writes manifests in the same format
under sample/data/<corpus>/manifests/. Setting CSASR_SAMPLE=1 makes every data/... path resolve
under sample/, so train.py, decode.py and score.py run on the sample unchanged. Our stress-set
recordings are never included (course staff only).

  python scripts/make_sample.py            # needs the full data/ (run on the machine that has it)
  CSASR_SAMPLE=1 python scripts/decode.py --model tiny --precision fp32 --method sample
"""
import argparse
import json
import os
import random
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from csasr.data import manifest  # noqa: E402

# set -> utterances kept (train/val only need to be big enough for a few training steps)
COUNTS = {"ascend_test": 60, "nsc_test": 60, "nsc_diff_test": 60,
          "ascend_train": 40, "nsc_train": 40, "ascend_val": 20, "nsc_val": 20}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="sample")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if os.environ.get("CSASR_SAMPLE"):
        sys.exit("unset CSASR_SAMPLE: the sample is built from the full data/")
    out = manifest.path(a.out)
    total = 0
    for name, n in COUNTS.items():
        rows = [r for r in manifest.load(name) if 1.0 <= r["duration"] <= 10.0]
        rng = random.Random(f"{a.seed}-{name}")
        mixed = [r for r in rows if r["mixed"]]
        mono = [r for r in rows if not r["mixed"]]
        k = round(n * len(mixed) / len(rows))
        keep = rng.sample(mixed, k) + rng.sample(mono, n - k)
        keep.sort(key=lambda r: r["utt_id"])
        for r in keep:
            dst = os.path.join(out, r["audio"])
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copyfile(manifest.path(r["audio"]), dst)
            total += os.path.getsize(dst)
        mf = os.path.join(out, manifest.SETS[name])
        os.makedirs(os.path.dirname(mf), exist_ok=True)
        with open(mf, "w", encoding="utf-8") as f:
            f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in keep)
        print(f"{name}: {len(keep)} utterances ({k} mixed), {sum(r['duration'] for r in keep) / 60:.1f} min")
    print(f"{total / 1e6:.1f} MB of audio in {out}")


if __name__ == "__main__":
    main()
