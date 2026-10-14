"""Rule-based critical-word tagger (proposal §3.3).

Critical words: numbers, dates, amounts, names, drug names, technical terms.
This is the automatic labeller; test-set labels are PENDING_HUMAN review
(see PENDING_HUMAN.md). Returns [start, end) spans over reference units.
"""
import os
import re
from functools import lru_cache

_DIR = os.path.dirname(__file__)
_TERMS_FILE = os.path.join(_DIR, "critical_terms.txt")

# ambiguous everyday words ("may", "march", "one", "second") are deliberately left out
MONTHS_EN = {"january", "february", "april", "june", "july", "august", "september",
             "october", "november", "december", "feb", "apr", "jun", "jul", "aug", "sept", "nov"}
DAYS_EN = {"monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday", "tomorrow",
           "today", "tonight", "yesterday", "weekend", "deadline"}
NUM_EN = {"zero", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven",
          "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen",
          "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety", "hundred",
          "thousand", "million", "billion"}
MONEY_EN = {"dollar", "dollars", "cent", "cents", "sgd", "usd", "rmb", "percent", "bucks"}
# Chinese date/amount counters attached to digits
ZH_COUNTERS = set("号日月年点块元毛分岁%")
ZH_DATE_WORDS = ["星期", "礼拜", "周末", "明天", "后天", "昨天", "今天", "截止"]


@lru_cache(maxsize=1)
def term_list() -> tuple[set[str], list[str]]:
    """English (word-level) and Chinese (string-level) critical terms."""
    en, zh = set(), []
    if os.path.exists(_TERMS_FILE):
        for line in open(_TERMS_FILE, encoding="utf-8"):
            t = line.split("#")[0].strip().lower()
            if not t:
                continue
            if re.search(r"[一-鿿]", t):
                zh.append(t)
            else:
                en.update(t.split())
    zh.sort(key=len, reverse=True)
    return en, zh


def tag_critical(lab) -> list[tuple[int, int]]:
    units = lab.units
    n = len(units)
    crit = [False] * n
    en_terms, zh_terms = term_list()
    for i, u in enumerate(units):
        if u.isdigit():
            crit[i] = True
            if i + 1 < n and units[i + 1] in ZH_COUNTERS:
                crit[i + 1] = True
        elif u in NUM_EN or u in MONTHS_EN or u in DAYS_EN or u in MONEY_EN or u in en_terms:
            crit[i] = True
    # Chinese date words and terms: string match over the unit sequence
    starts = []
    pos = 0
    for u in units:
        starts.append(pos)
        pos += len(u)
    flat = "".join(units)
    for w in ZH_DATE_WORDS + zh_terms:
        for m in re.finditer(re.escape(w), flat):
            for i, s in enumerate(starts):
                if m.start() <= s < m.end():
                    crit[i] = True
    # group consecutive critical units into words ("10 号" is one word); a zh<->en
    # change splits words, neutral units (digits) join either side
    spans, i = [], 0
    while i < n:
        if not crit[i]:
            i += 1
            continue
        j, cur = i + 1, lab.langs[i]
        while j < n and crit[j] and (lab.langs[j] in ("neu", cur) or cur == "neu"):
            cur = lab.langs[j] if cur == "neu" else cur
            j += 1
        spans.append((i, j))
        i = j
    return spans
