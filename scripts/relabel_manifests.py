#!/usr/bin/env python3
"""Re-derive ref / target / main / mixed of every manifest from ref_raw with the current
normalizer (run after any change to csasr/text)."""
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from csasr.data import manifest  # noqa: E402
from csasr.text.normalize import normalize  # noqa: E402
from csasr.text.units import label, training_target  # noqa: E402

for p in sorted(glob.glob(manifest.path("data/*/manifests/*.jsonl"))):
    rows = [json.loads(l) for l in open(p, encoding="utf-8")]
    changed = 0
    for r in rows:
        if not r.get("ref_raw"):
            continue
        ref = normalize(r["ref_raw"])
        lab = label(ref, already_normalized=True)
        new = dict(ref=ref, target=training_target(r["ref_raw"]), main=lab.main, mixed=lab.mixed)
        assert label(new["target"]).units == lab.units, (r["utt_id"], new["target"])
        changed += any(r.get(k) != v for k, v in new.items())
        r.update(new)
    rows = [r for r in rows if r.get("ref") or not r.get("ref_raw")]
    with open(p + ".tmp", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(p + ".tmp", p)
    print(f"{os.path.relpath(p, manifest.ROOT)}: {len(rows)} rows, {changed} relabelled")
