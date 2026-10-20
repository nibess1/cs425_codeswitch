#!/usr/bin/env python3
"""Section E data: Diff Room session split 70/10/20 for the recording-condition ablation.

  python scripts/prepare_diffroom_ablation.py
Writes data/nsc_e/manifests/{diff_train,diff_val,diff_test,mix_train}.jsonl and clips under
data/nsc_e/clips/. Sessions of speakers who appear in both rooms are excluded, so no E-test
speaker is in any training set (Same Room or Diff Room).
  diff_train  40 h Diff Room (all code-switched first, up to 30 h, then English-only)
  mix_train   20 h of the Same Room train manifest + 20 h of diff_train (random halves)
  diff_val    3 h,  diff_test  4 h  (random samples of the val / test sessions)
"""
import json
import os
import random
import sys
from multiprocessing import Pool

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from prepare_nsc import cut_clips, take_hours  # noqa: E402
from csasr.data import manifest  # noqa: E402

SEED = 425


def main():
    out = manifest.path("data/nsc_e")
    os.makedirs(os.path.join(out, "manifests"), exist_ok=True)
    pool = pd.read_parquet(manifest.path("data/nsc/pool.parquet"))
    same_spk = set(pool[pool.room == "same"].speaker)
    diff = pool[(pool.room == "diff") & pool.drop_reason.isin(["", "speaker_seen"])]
    dual = set(diff.speaker) & same_spk
    bad_sess = set(diff[diff.speaker.isin(dual)].session)
    diff = diff[~diff.session.isin(bad_sess) & (diff.drop_reason == "")]
    # some speakers recorded several Diff Room sessions: split connected speaker-session groups
    # (union-find), so no speaker is on two sides of the split
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for sess_id, spk in diff[["session", "speaker"]].drop_duplicates().itertuples(index=False):
        parent[find("s" + sess_id)] = find("p" + spk)
    comp = {sid: find("s" + sid) for sid in diff.session.unique()}
    groups = sorted(set(comp.values()))
    random.Random(SEED).shuffle(groups)
    hours = diff.groupby(diff.session.map(comp)).dur.sum() / 3600
    total, acc, gsplit = hours.sum(), 0.0, {}
    for g in groups:  # fill 70 / 10 / 20 % of the hours in shuffled group order
        gsplit[g] = "train" if acc < 0.7 * total else ("val" if acc < 0.8 * total else "test")
        acc += hours[g]
    diff = diff.assign(esplit=diff.session.map(comp).map(gsplit))
    print(f"{len(groups)} speaker-session groups; largest {diff.session.map(comp).value_counts().max()} utts")
    tr = diff[diff.esplit == "train"]
    mixed = take_hours(tr[tr.mixed], 30.0, SEED)
    other = take_hours(tr[~tr.mixed], 40.0 - mixed.dur.sum() / 3600, SEED)
    sel = {"diff_train": pd.concat([mixed, other]),
           "diff_val": take_hours(diff[diff.esplit == "val"], 3.0, SEED),
           "diff_test": take_hours(diff[diff.esplit == "test"], 4.0, SEED)}
    clip_dir = os.path.join(out, "clips")
    for s in sel:
        os.makedirs(os.path.join(clip_dir, s), exist_ok=True)
    allsel = pd.concat([d.assign(split=s) for s, d in sel.items()])
    jobs = [(w, g.to_dict("records"), clip_dir) for w, g in allsel.groupby("wav")]
    paths = {}
    with Pool(32) as p:
        for chunk in p.imap_unordered(cut_clips, jobs):
            paths.update(chunk)
    rows = {}
    for s, d in sel.items():
        rows[s] = [dict(utt_id="e" + r["utt_id"], corpus="nsc_diff_e", split=s.split("_")[1], audio=os.path.relpath(
            paths[r["utt_id"]], manifest.ROOT), duration=round(r["dur"], 3), speaker=r["speaker"], session=r["session"],
            ref_raw=r["ref_raw"], ref=r["ref"], target=r["target"], main=r["main"], mixed=bool(r["mixed"]))
            for r in d.sort_values("utt_id").to_dict("records")]
    # condition C: random half (by hours) of the Same Room train manifest + of diff_train
    same_tr = manifest.load("nsc_train")
    rng = random.Random(SEED)
    rng.shuffle(same_tr)
    half_same, h = [], 0.0
    for r in same_tr:
        if h + r["duration"] > 20 * 3600:
            break
        half_same.append(r); h += r["duration"]
    dtr = list(rows["diff_train"])
    rng.shuffle(dtr)
    half_diff, h = [], 0.0
    for r in dtr:
        if h + r["duration"] > 20 * 3600:
            break
        half_diff.append(r); h += r["duration"]
    rows["mix_train"] = half_same + half_diff
    for s, rs in rows.items():
        with open(os.path.join(out, "manifests", f"{s}.jsonl"), "w", encoding="utf-8") as f:
            for r in rs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        spk = {r["speaker"] for r in rs}
        print(f"{s}: {len(rs)} utts, {sum(r['duration'] for r in rs) / 3600:.1f} h, mixed "
              f"{sum(r['mixed'] for r in rs) / len(rs):.1%}, speakers {len(spk)}, sessions {len({r['session'] for r in rs})}")
    # speaker-disjointness checks
    e_test = {r["speaker"] for r in rows["diff_test"]}
    train_spk = {r["speaker"] for r in same_tr} | {r["speaker"] for r in rows["diff_train"]}
    assert not (e_test & train_spk), "E-test speaker in a training set"
    print(f"excluded {len(bad_sess)} sessions of {len(dual)} dual-room speakers; E-test speakers disjoint from all training: ok")


if __name__ == "__main__":
    main()
