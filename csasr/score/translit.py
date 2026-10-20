"""C3: phonetic-transliteration detector for error blocks (embedded word -> other script).

The Han side is romanized with pypinyin (no tones), both sides are encoded with Metaphone,
and similarity = 1 - Levenshtein(codes) / max(code lengths). A block whose reference side is
embedded and whose output side is in the other script counts as a transliteration when
similarity >= THETA (chosen on the D1 labels, PROCESS_LOG) and it is not a translation.
"""
import os
import re
from functools import lru_cache

THETA = float(os.environ.get("CSASR_TRANSLIT_THETA", "0.1"))  # C3: best F1 vs D1 labels
_HAN = re.compile(r"[㐀-鿿豈-﫿]")


@lru_cache(maxsize=200_000)
def _code(s: str) -> str:
    import jellyfish
    from pypinyin import lazy_pinyin
    s = "".join(lazy_pinyin(s)) if _HAN.search(s) else s
    s = re.sub(r"[^a-z]", "", s.lower())
    return jellyfish.metaphone(s) if s else ""


@lru_cache(maxsize=200_000)
def phonetic_similarity(a: str, b: str) -> float:
    import jellyfish
    ca, cb = _code(a), _code(b)
    if not ca or not cb:
        return 0.0
    return 1.0 - jellyfish.levenshtein_distance(ca, cb) / max(len(ca), len(cb))


def is_cross_script(ref_side: str, out_side: str) -> bool:
    """One side is Han and the other Latin (the replacement changed script)."""
    rh, oh = bool(_HAN.search(ref_side)), bool(_HAN.search(out_side))
    rl, ol = bool(re.search(r"[a-z]", ref_side)), bool(re.search(r"[a-z]", out_side))
    return (rl and not rh and oh) or (rh and not rl and ol)
