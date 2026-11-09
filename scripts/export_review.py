#!/usr/bin/env python3
"""Files for the human checks of the automatic labels.

  python scripts/export_review.py d1     D1: every error block (all systems, ASCEND test + val) where
        embedded units were replaced by other-script text, deduplicated to unique (reference side,
        output side) pairs -> data/llm_labels/d1_pairs.csv (pair_id, ref_side, out_side, n_occurrences,
        n_systems, cedict_match, labse_sim). NSC is excluded until its licence is confirmed (PROCESS_LOG A4).
  python scripts/export_review.py d2     D2: critical-word candidates for ASCEND test + the stress
        sentences (NSC excluded, A4). Candidates = units/words flagged by the rule tagger OR a broad
        finder: digits and Chinese numerals; Latin words outside the 5,000 most common English words;
        fuzzy matches (rapidfuzz >= 85) to the critical term list; jieba person/place/org/other
        proper-noun tags (nr, ns, nt, nz). Deduplicated to unique word types with one example.
        -> data/llm_labels/d2_candidates.csv, and d2_read500.csv (500 random ASCEND utterances to
        read in full for words the finder misses).
  python scripts/export_review.py team --system ft_base__fp16__ft     the team's checks -> review/*.csv:
        critical_words.csv (every auto-tagged critical word in the test sets), translation_blocks.csv
        (100 candidate language-confusion blocks with LaBSE sim), alignments.csv (unit spans of 30
        random aligned test utterances), partner_leak.csv (20 random NSC test clips to listen to)
"""
import argparse
import collections
import csv
import glob
import json
import os
import pickle
import random
import re
import sys

import jieba.posseg as pseg
from rapidfuzz import fuzz, process
from wordfreq import top_n_list

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from csasr.data import manifest  # noqa: E402
from csasr.score import report as R  # noqa: E402
from csasr.score.align import score  # noqa: E402
from csasr.score.confusion import LaBSE, cedict_match  # noqa: E402
from csasr.score.metrics import _side, _structural_confusion  # noqa: E402
from csasr.text.critical import term_list  # noqa: E402
from csasr.text.normalize import normalize  # noqa: E402
from csasr.text.units import label, split_units, unit_lang  # noqa: E402

SETS = ("ascend_test", "ascend_val")  # d1
COMMON = set(top_n_list("en", 5000))  # d2
FORCE = False  # d1/d2: overwrite the labelled input lists (see main)
NUMERAL = re.compile(r"^[\d零〇一二两三四五六七八九十百千万亿]+$")


def d1():
    if not FORCE and os.path.exists(manifest.path("data/llm_labels/d1_pairs.csv")):  # the labelled pair list
        print("data/llm_labels/d1_pairs.csv exists (its pairs were labelled); --force rebuilds it from today's preds/")
        return
    pairs = collections.Counter()
    systems = collections.defaultdict(set)
    for p in sorted(glob.glob(manifest.path("preds/*.jsonl"))):
        name = os.path.basename(p)[:-6]
        for l in open(p, encoding="utf-8"):
            r = json.loads(l)
            if r.get("set") not in SETS:
                continue
            lab = label(r["ref"])
            if lab.main not in ("zh", "en") or not lab.mixed:
                continue
            hu = split_units(normalize(r["hyp"]))
            hl = [unit_lang(u) for u in hu]
            emb = "en" if lab.main == "zh" else "zh"
            for b in score(lab.units, hu).blocks:
                if not b.ref_idx or not b.hyp_idx or any(lab.roles[i] != "E" for i in b.ref_idx):
                    continue
                if not any(hl[j] == lab.main for j in b.hyp_idx):  # other-script text in the output
                    continue
                key = (_side(lab.units, b.ref_idx, emb), _side(hu, b.hyp_idx, lab.main if all(
                    hl[j] == lab.main for j in b.hyp_idx) else "en"))
                pairs[key] += 1
                systems[key].add(name)
    keys = sorted(pairs, key=lambda k: -pairs[k])
    sims = LaBSE(device=os.environ.get("CSASR_LABSE_DEVICE", "cpu")).sims(keys)
    os.makedirs(manifest.path("data/llm_labels"), exist_ok=True)
    with open(manifest.path("data/llm_labels/d1_pairs.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["pair_id", "ref_side", "out_side", "n_occurrences", "n_systems", "cedict_match", "labse_sim"])
        for i, (k, s) in enumerate(zip(keys, sims)):
            zh, en = (k[1], k[0]) if all(not c.isascii() for c in k[1].replace(" ", "")) else (k[0], k[1])
            w.writerow([f"d1_{i:05d}", k[0], k[1], pairs[k], len(systems[k]), cedict_match(zh, en), round(s, 3)])
    print(len(keys), "unique pairs;", sum(pairs.values()), "occurrences")


def words(lab):
    """Words: English units, digit strings, and jieba words over Chinese runs (with POS)."""
    out, i, n = [], 0, len(lab.units)
    while i < n:
        if lab.langs[i] == "zh":
            j = i
            while j < n and lab.langs[j] == "zh":
                j += 1
            for w, pos in pseg.cut("".join(lab.units[i:j])):
                out.append((w, pos, "zh"))
            i = j
        else:
            out.append((lab.units[i], "x", lab.langs[i]))
            i += 1
    return out


def d2():
    if not FORCE and os.path.exists(manifest.path("data/llm_labels/d2_candidates.csv")):  # the labelled item list
        print("data/llm_labels/d2_candidates.csv exists (its items were labelled); --force rebuilds it")
        return
    en_terms, zh_terms = term_list()
    terms = sorted(en_terms) + zh_terms
    rows = [(r["utt_id"], r["ref"]) for r in manifest.load("ascend_test")]
    stress = list(csv.DictReader(open(manifest.path("stress/sentences.tsv"), encoding="utf-8"), delimiter="\t"))
    rows += [(f"stress_{r['id']}", normalize(r["text"])) for r in stress]
    cand = collections.OrderedDict()
    for uid, ref in rows:
        lab = label(ref, already_normalized=True)
        rule = {("" if lab.langs[s] == "zh" else " ").join(lab.units[s:e]) for s, e in lab.critical}
        for w, pos, lang in words(lab):
            why = []
            if any(w == r or w in r.split() or w in r for r in rule):
                why.append("rule_tagger")
            if NUMERAL.match(w) and (any(ch.isdigit() for ch in w) or len(w) >= 2):
                why.append("numeral")
            if lang == "en" and w not in COMMON and len(w) > 1:
                why.append("rare_latin")
            if pos in ("nr", "ns", "nt", "nz"):
                why.append(f"jieba_{pos}")
            m = process.extractOne(w, terms, scorer=fuzz.ratio) if len(w) > 2 else None
            if m and m[1] >= 85:
                why.append("term_fuzzy")
            if not why:
                continue
            key = w
            if key not in cand:
                cand[key] = dict(word=w, example=ref, example_utt=uid, why=set(why), n=0,
                                 rule_tagger=False)
            cand[key]["n"] += 1
            cand[key]["why"] |= set(why)
            cand[key]["rule_tagger"] |= "rule_tagger" in why
    out = manifest.path("data/llm_labels")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "d2_candidates.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["item_id", "word", "example_sentence", "n_occurrences", "finder_reasons", "rule_tagger_critical"])
        for i, c in enumerate(cand.values()):
            w.writerow([f"d2_{i:05d}", c["word"], c["example"], c["n"], ";".join(sorted(c["why"])), c["rule_tagger"]])
    asc = [r for r in rows if r[0].startswith("ascend")]
    sample = random.Random(425).sample(asc, 500)
    with open(os.path.join(out, "d2_read500.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["utt_id", "reference", "finder_candidates"])
        for uid, ref in sample:
            lab = label(ref, already_normalized=True)
            found = [x for x, _, _ in words(lab) if x in cand]
            w.writerow([uid, ref, " | ".join(found)])
    print(len(cand), "unique candidate words;", sum(c["rule_tagger"] for c in cand.values()), "flagged by the rule tagger")
    print(collections.Counter(r for c in cand.values() for r in c["why"]))


def team(a):
    out = manifest.path("review")
    os.makedirs(out, exist_ok=True)
    rng = random.Random(425)
    with open(os.path.join(out, "critical_words.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["set", "utt_id", "reference", "critical_word", "keep(1/0)"])
        for s in manifest.EVAL_TEST:
            for r in manifest.load(s):
                lab = label(r["ref"], already_normalized=True)
                for st, en in lab.critical:
                    w.writerow([s, r["utt_id"], r["ref"], _side(lab.units, range(st, en), lab.langs[st]), ""])
    cands = []
    for r in R.read_preds(a.system):
        lab = label(r["ref"])
        hu = split_units(normalize(r["hyp"]))
        hl = [unit_lang(u) for u in hu]
        for b in score(lab.units, hu).blocks:
            if _structural_confusion(b, lab, hl):
                emb = "en" if lab.main == "zh" else "zh"
                cands.append((r["utt_id"], r["ref"], r["hyp"], _side(lab.units, b.ref_idx, emb), _side(hu, b.hyp_idx, lab.main)))
    rng.shuffle(cands)
    cands = cands[:100]
    sims = LaBSE().sims([(c[3], c[4]) for c in cands]) if cands else []
    with open(os.path.join(out, "translation_blocks.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["utt_id", "reference", "output", "ref_side", "out_side", "labse_sim", "is_translation(1/0)"])
        for c, sm in zip(cands, sims):
            w.writerow([*c, round(sm, 3), ""])
    al = pickle.load(open(manifest.path("data/align/nsc_test.pkl"), "rb")) if os.path.exists(
        manifest.path("data/align/nsc_test.pkl")) else {}
    al_a = pickle.load(open(manifest.path("data/align/ascend_test.pkl"), "rb")) if os.path.exists(
        manifest.path("data/align/ascend_test.pkl")) else {}
    rows = {r["utt_id"]: r for s in ("nsc_test", "ascend_test") for r in manifest.load(s)}
    pick = rng.sample(sorted(set(al) & set(rows)), min(15, len(al))) + rng.sample(sorted(set(al_a) & set(rows)), min(15, len(al_a)))
    with open(os.path.join(out, "alignments.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["utt_id", "audio", "unit", "start_s", "end_s", "true_start_s", "true_end_s"])
        for u in pick:
            v = (al.get(u) or al_a.get(u))
            for unit, (s0, e0, _) in zip(v["units"], v["spans"]):
                w.writerow([u, rows[u]["audio"], unit, s0, e0, "", ""])
    test = manifest.load("nsc_test")
    with open(os.path.join(out, "partner_leak.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["utt_id", "audio", "reference", "partner_speech_audible(1/0)"])
        for r in rng.sample(test, 20):
            w.writerow([r["utt_id"], r["audio"], r["ref_raw"], ""])
    print("wrote", os.listdir(out))


def main():
    ap = argparse.ArgumentParser(description="files for the human label checks (see the top of this file)")
    ap.add_argument("what", choices=["d1", "d2", "team"])
    ap.add_argument("--system", default="ft_base__fp16__ft", help="team: system whose outputs are exported")
    ap.add_argument("--force", action="store_true", help="d1/d2: overwrite the existing (already labelled) lists")
    a = ap.parse_args()
    global FORCE
    FORCE = a.force
    d1() if a.what == "d1" else d2() if a.what == "d2" else team(a)


if __name__ == "__main__":
    main()
