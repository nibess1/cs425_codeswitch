"""Shared text normalization (proposal §3.2).

The same `normalize()` is applied to references and model outputs. Never score
un-normalized text.
"""
import re
import unicodedata
from functools import lru_cache

import cn2an
from opencc import OpenCC

_t2s = OpenCC("t2s")

HAN = r"㐀-䶿一-鿿豈-﫿"

# --- step 4: corpus markers -------------------------------------------------
# NSC inline language tags: <mandarin>汉字:pinyin</mandarin> -> 汉字 (any language tag).
_LANG_TAG = re.compile(r"<(\w+)>([^<]*?)(?::[^<]*)?</\1>")
# Whole-token markers that are deleted (silence, noise, unknown, NSC xml-ish tags).
_DROP_TOKENS = re.compile(r"<[A-Za-z]+/?>|\[UNK\]|<UNK>", re.I)
# Fillers / paralinguistics in parentheses: (um) (uh) (ppl) (ppb) ... -> deleted.
_PAREN = re.compile(r"\([^)]*\)")
# Kept-word wrappers: [oh] !wah! #name# -> word.
_BRACKET = re.compile(r"\[([^\]]*)\]")
_BANG = re.compile(r"!([^!\s][^!]*?)!")
_HASH = re.compile(r"#([^#]*)#")
# Truncated words: engli~ -> deleted.
_TRUNC = re.compile(r"\S*~")
# Speaker tags like "A:" / "spk1:" at line start.
_SPK = re.compile(r"^\s*(?:spk\d*|speaker\s*\d*|[A-Z])\s*[:：]\s*", re.I)


def strip_markers(text: str) -> str:
    text = _LANG_TAG.sub(lambda m: f" {m.group(2)} ", text)
    text = _SPK.sub("", text)
    text = _DROP_TOKENS.sub(" ", text)
    text = _PAREN.sub(" ", text)
    text = _BRACKET.sub(r" \1 ", text)
    text = _BANG.sub(r" \1 ", text)
    text = _HASH.sub(r" \1 ", text)
    text = _TRUNC.sub(" ", text)
    return text


# --- step 2: Chinese numerals -> digits -------------------------------------
_NUM_CH = "零〇一二两三四五六七八九十百千万亿"
# Convert a numeral run when it is >= 2 chars, or 1 char followed by a counter
# that makes it clearly a number (date, time, money, age, percent).
_NUM_RUN = re.compile(rf"[{_NUM_CH}]{{2,}}|[{_NUM_CH}](?=[号日月年点块元岁%％])")
# Idioms / fixed words containing numerals that must not be converted.
_NUM_KEEP = ("一一", "一些", "一下", "一起", "一直", "一定", "一样", "一般", "一点点", "万一", "十分",
             "千万", "一万", "三三两两", "一边", "一切")


def _num_sub(m: re.Match) -> str:
    s = m.group(0)
    try:
        return str(cn2an.cn2an(s.replace("〇", "零"), "smart"))
    except Exception:
        return s


def chinese_numerals_to_digits(text: str) -> str:
    # protect fixed expressions with a placeholder
    keep = {}
    for i, w in enumerate(_NUM_KEEP):
        if w in text:
            ph = f"\x00{i}\x00"
            keep[ph] = w
            text = text.replace(w, ph)
    text = _NUM_RUN.sub(_num_sub, text)
    for ph, w in keep.items():
        text = text.replace(ph, w)
    return text


# --- step 3: English --------------------------------------------------------
VARIANTS = {
    "okay": "ok", "o.k": "ok", "e-mail": "email", "e-mails": "emails", "alright": "all right",
    "yah": "ya", "uhh": "uh", "mmm": "mm",
    "wi-fi": "wifi", "t-shirt": "tshirt", "t-shirts": "tshirts",
}
_PUNCT_KEEP_INSIDE = re.compile(r"(?<=\w)['’](?=\w)")


def _strip_punct(text: str, keep: str = "") -> str:
    text = _PUNCT_KEEP_INSIDE.sub("", text)          # don't -> dont, what's -> whats
    out = []
    for ch in text:
        if ch not in keep and unicodedata.category(ch)[0] in "PSZC":
            out.append(" ")
        else:
            out.append(ch)
    return "".join(out)


@lru_cache(maxsize=200_000)
def normalize(text: str) -> str:
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = strip_markers(text)
    text = _t2s.convert(text)
    text = chinese_numerals_to_digits(text)
    text = text.lower()
    # variants are matched with in-word "-" and "." still present (e-mail, o.k.)
    text = _strip_punct(text, keep="-.")
    text = " ".join(VARIANTS.get(t.strip(".-"), t) for t in text.split())
    text = _strip_punct(text)
    return " ".join(VARIANTS.get(t, t) for t in text.split())
