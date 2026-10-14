"""Units, unit languages, main language and roles (proposal §1, §3.2, §3.3).

A unit is one Chinese character, one Latin-script word or one digit string.
Roles of reference units:
  E    embedded unit (language != main language)
  Ms   matrix unit directly before/after a change of language
  Mi   matrix unit elsewhere
  neu  neutral unit (digits, spelled letters, Singapore particles)
  bal  any non-neutral unit in a balanced sentence (tie in word counts)
"""
import re
from dataclasses import dataclass, field

import jieba

from .normalize import HAN, normalize

jieba.setLogLevel(60)

_UNIT = re.compile(rf"[{HAN}]|[a-z0-9]*[a-z][a-z0-9]*|\d+")
PARTICLES = {"lah", "leh", "lor", "meh", "hor", "sia"}
_REAL_ONE_LETTER = {"a", "i"}


def split_units(norm_text: str) -> list[str]:
    return _UNIT.findall(norm_text)


def unit_lang(u: str) -> str:
    if re.fullmatch(rf"[{HAN}]", u):
        return "zh"
    if u.isdigit() or u in PARTICLES or (len(u) == 1 and u not in _REAL_ONE_LETTER):
        return "neu"
    return "en"


def training_target(text: str) -> str:
    """Normalized text with a space at every change of script (no subword spans both)."""
    units = split_units(normalize(text))
    out = []
    for i, u in enumerate(units):
        if i == 0:
            out.append(u)
            continue
        prev_han = bool(re.fullmatch(rf"[{HAN}]", units[i - 1]))
        cur_han = bool(re.fullmatch(rf"[{HAN}]", u))
        out.append(u if (prev_han and cur_han) else " " + u)
    return "".join(out)


def main_language(units: list[str], langs: list[str] | None = None) -> str:
    """'zh', 'en' or 'balanced' by word count after jieba-segmenting the Chinese part."""
    langs = langs or [unit_lang(u) for u in units]
    n_en = sum(l == "en" for l in langs)
    runs, cur = [], []
    for u, l in zip(units, langs):
        if l == "zh":
            cur.append(u)
        elif cur:
            runs.append("".join(cur)); cur = []
    if cur:
        runs.append("".join(cur))
    n_zh = sum(len([w for w in jieba.lcut(r) if w.strip()]) for r in runs)
    if n_zh == n_en:
        return "balanced" if n_zh else "en"
    return "zh" if n_zh > n_en else "en"


@dataclass
class Labelled:
    text: str                    # normalized text
    units: list[str]
    langs: list[str]             # zh / en / neu
    main: str                    # zh / en / balanced
    roles: list[str]             # E / Ms / Mi / neu / bal
    switch: list[bool]           # unit next to a change of language
    mixed: bool                  # sentence contains both zh and en units
    critical: list[tuple[int, int]] = field(default_factory=list)  # [start, end) unit spans


def label(text: str, already_normalized: bool = False) -> Labelled:
    norm = text if already_normalized else normalize(text)
    units = split_units(norm)
    langs = [unit_lang(u) for u in units]
    main = main_language(units, langs)
    mixed = "zh" in langs and "en" in langs
    # switch points over the sequence of non-neutral units
    idx = [i for i, l in enumerate(langs) if l != "neu"]
    switch = [False] * len(units)
    for k, i in enumerate(idx):
        if k > 0 and langs[idx[k - 1]] != langs[i]:
            switch[i] = True
        if k + 1 < len(idx) and langs[idx[k + 1]] != langs[i]:
            switch[i] = True
    roles = []
    for i, l in enumerate(langs):
        if l == "neu":
            roles.append("neu")
        elif main == "balanced":
            roles.append("bal")
        elif l != main:
            roles.append("E")
        else:
            roles.append("Ms" if switch[i] else "Mi")
    from .critical import tag_critical  # local import avoids a cycle
    lab = Labelled(norm, units, langs, main, roles, switch, mixed)
    lab.critical = tag_critical(lab)
    return lab
