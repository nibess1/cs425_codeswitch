#!/usr/bin/env python3
"""D2: critical-word candidates for ASCEND test + the stress sentences (NSC excluded, A4).

Candidates = units/words flagged by the rule tagger OR a broad finder:
  digits and Chinese numerals; Latin words outside the 5,000 most common English words;
  fuzzy matches (rapidfuzz >= 85) to the critical term list; jieba person/place/org/other
  proper-noun tags (nr, ns, nt, nz). Deduplicated to unique word types with one example.
-> data/llm_labels/d2_candidates.csv, and d2_read500.csv (500 random ASCEND utterances to
   read in full for words the finder misses).
"""
import collections
import csv
import os
import random
import re
import sys

import jieba.posseg as pseg
from rapidfuzz import fuzz, process
from wordfreq import top_n_list

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from csasr.data import manifest  # noqa: E402
from csasr.text.critical import term_list  # noqa: E402
from csasr.text.normalize import normalize  # noqa: E402
from csasr.text.units import label  # noqa: E402

COMMON = set(top_n_list("en", 5000))
NUMERAL = re.compile(r"^[\d零〇一二两三四五六七八九十百千万亿]+$")


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


def main():
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


if __name__ == "__main__":
    main()
