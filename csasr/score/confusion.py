"""Language-confusion (translation) detection for error blocks (proposal §4.2).

A block is a language confusion when its reference side is a run of embedded
units, its output side is entirely in the main language, and the output is a
translation: a CC-CEDICT entry, or LaBSE cosine similarity >= tau.
"""
import gzip
import json
import os
import re
from functools import lru_cache

CEDICT_PATH = os.environ.get("CSASR_CEDICT", os.path.expanduser("~/cs425-codeswitch-asr/data/lex/cedict_ts.u8"))
SIM_CACHE = os.environ.get("CSASR_SIMCACHE", os.path.expanduser("~/cs425-codeswitch-asr/data/lex/labse_cache.json.gz"))
DEFAULT_TAU = float(os.environ.get("CSASR_TAU", "0.70"))

_PAREN = re.compile(r"\([^)]*\)")
_STOP = {"a", "an", "the", "to", "of", "one's", "sb", "sth", "something", "someone"}


def _gloss_norm(g: str) -> str:
    g = _PAREN.sub("", g).lower().strip()
    g = re.sub(r"^(to|a|an|the) ", "", g)
    g = re.sub(r"[^a-z0-9' ]", " ", g)
    return " ".join(g.split())


def _sing(w: str) -> str:
    return w[:-1] if len(w) > 3 and w.endswith("s") and not w.endswith("ss") else w


@lru_cache(maxsize=1)
def cedict():
    """zh2en: simplified headword -> set of normalized glosses; en2zh: gloss -> [headwords]."""
    zh2en, en2zh = {}, {}
    if not os.path.exists(CEDICT_PATH):
        return zh2en, en2zh
    line_re = re.compile(r"^(\S+) (\S+) \[[^\]]*\] /(.*)/\s*$")
    for line in open(CEDICT_PATH, encoding="utf-8"):
        if line.startswith("#"):
            continue
        m = line_re.match(line)
        if not m:
            continue
        simp = m.group(2)
        for g in m.group(3).split("/"):
            if g.startswith(("CL:", "variant of", "old variant", "see ", "surname ", "abbr.")):
                continue
            ng = _gloss_norm(g)
            if not ng or len(ng.split()) > 4:
                continue
            zh2en.setdefault(simp, set()).add(ng)
            gw = [_sing(w) for w in ng.split()]
            en2zh.setdefault(" ".join(gw), []).append((0, simp))
            if 1 < len(gw) <= 3:  # head-word entry, lower priority ("school assignment" -> assignment)
                en2zh.setdefault(gw[-1], []).append((1, simp))
    import jieba
    jieba.initialize()  # FREQ is empty until the dictionary is loaded
    freq = jieba.get_FREQ
    for k, v in en2zh.items():  # exact gloss first, words over single characters, then most frequent
        best = {}
        for pri, w in v:
            best[w] = min(pri, best.get(w, 9))
        en2zh[k] = sorted(best, key=lambda w: (best[w], len(w) == 1, -(freq(w) or 0), len(w)))
    return zh2en, en2zh


def cedict_match(zh: str, en: str) -> bool:
    zh2en, _ = cedict()
    glosses = zh2en.get(zh)
    if not glosses:
        return False
    ew = [_sing(w) for w in en.split() if w not in _STOP] or en.split()
    for g in glosses:
        gw = [_sing(w) for w in g.split() if w not in _STOP]
        # exact gloss, or a short gloss whose head matches ("school assignment" ~ "assignment")
        if gw == ew or (len(gw) <= 3 and gw[-len(ew):] == ew):
            return True
    return False


def en_to_zh(en: str) -> str | None:
    """Most basic dictionary translation of an English word/phrase (for swap candidates)."""
    _, en2zh = cedict()
    key = " ".join(_sing(w) for w in en.lower().split())
    c = en2zh.get(key)
    return c[0] if c else None


def zh_to_en(zh: str) -> str | None:
    zh2en, _ = cedict()
    g = zh2en.get(zh)
    if not g:
        return None
    return sorted(g, key=lambda x: (len(x.split()), len(x)))[0]  # shortest gloss


class LaBSE:
    """Cached LaBSE cosine similarity. Loaded lazily; cache persisted to disk."""

    def __init__(self, device=None):
        self.model = None
        self.device = device
        self.cache = {}
        if os.path.exists(SIM_CACHE):
            try:
                self.cache = json.load(gzip.open(SIM_CACHE, "rt"))
            except Exception:
                self.cache = {}
        self._dirty = 0

    def _load(self):
        if self.model is None:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer("sentence-transformers/LaBSE", device=self.device)

    def sims(self, pairs: list[tuple[str, str]]) -> list[float]:
        todo = sorted({f"{a}\t{b}" for a, b in pairs} - set(self.cache))
        if todo:
            self._load()
            A = [t.split("\t")[0] for t in todo]
            B = [t.split("\t")[1] for t in todo]
            ea = self.model.encode(A, batch_size=256, normalize_embeddings=True, convert_to_numpy=True)
            eb = self.model.encode(B, batch_size=256, normalize_embeddings=True, convert_to_numpy=True)
            for t, x, y in zip(todo, ea, eb):
                self.cache[t] = max(0.0, float((x * y).sum()))
            self._dirty += len(todo)
        return [self.cache[f"{a}\t{b}"] for a, b in pairs]

    def save(self):
        if self._dirty:
            os.makedirs(os.path.dirname(SIM_CACHE), exist_ok=True)
            tmp = SIM_CACHE + ".tmp"
            json.dump(self.cache, gzip.open(tmp, "wt"))
            os.replace(tmp, SIM_CACHE)
            self._dirty = 0
