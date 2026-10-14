#!/usr/bin/env python3
"""Forced-align manifests -> data/align/<set>.npz (unit spans + 20 ms frame labels).

  CUDA_VISIBLE_DEVICES=4 python scripts/align.py ascend_train nsc_train ...
"""
import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from csasr.data import manifest  # noqa: E402
from csasr.lid.align import Aligner, align_rows, frame_labels  # noqa: E402


def main():
    al = Aligner()
    os.makedirs(manifest.path("data/align"), exist_ok=True)
    for name in sys.argv[1:]:
        out = manifest.path(f"data/align/{name}.pkl")
        if os.path.exists(out):
            continue
        rows = manifest.load(name)
        res, fail = {}, 0
        for r, units, spans in align_rows(al, rows, manifest.load_audio):
            if spans is None:
                fail += 1
                continue
            res[r["utt_id"]] = dict(units=units, spans=[(round(s, 3), round(e, 3), round(c, 3)) for s, e, c in spans],
                                    frames=frame_labels(units, spans, r["duration"]))
        pickle.dump(res, open(out, "wb"))
        print(f"{name}: aligned {len(res)} / {len(rows)} ({fail} failed)", flush=True)


if __name__ == "__main__":
    main()
