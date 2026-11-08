"""IMDA NSC Part 4 (code-switching), Mandarin speakers only (proposal §3.1).

Files: sur_<session>_<speaker>_phn{s|d}_cs-chn.{wav,TextGrid}; s = Same Room, d = Diff Room.
Each wav is one speaker's phone channel (16 kHz mono, ~1 h); the TextGrid has one
interval tier with that speaker's transcript. Mandarin is written inline as
<mandarin>汉字:pinyin</mandarin>.
"""
import os
import re
from collections import defaultdict

FNAME = re.compile(r"sur_(\d+)_(\d+)_phn([sd])_cs-(\w+)\.(wav|TextGrid)$")
SILENCE = {"<Z>", "<S>", "", "<NON/>", "<NPS/>", "<SPK/>", "<STA/>", "<FIL/>"}
_IV = re.compile(r'xmin = ([\d.]+)\s*\n\s*xmax = ([\d.]+)\s*\n\s*text = "((?:[^"]|"")*)"\s*\n')


def read_textgrid(path: str) -> list[tuple[float, float, str]]:
    b = open(path, "rb").read()
    enc = "utf-16" if b[:2] in (b"\xfe\xff", b"\xff\xfe") else "utf-8-sig"
    t = b.decode(enc, errors="replace").replace("\r", "")
    # only the first tier (the speaker's transcript); one file has a second tier
    tiers = re.split(r"\n\s*item \[\d+\]:", t)
    body = tiers[1] if len(tiers) > 1 else t
    return [(float(a), float(c), x.replace('""', '"').strip()) for a, c, x in _IV.findall(body)]


def index(raw_dir: str, tongue: str = "chn") -> list[dict]:
    """Pair wav + TextGrid files for one mother-tongue group."""
    files = defaultdict(dict)
    for sub in ("Same Room Audio", "Same Room Scripts", "Diff Room Audio", "Diff Room Scripts"):
        d = os.path.join(raw_dir, sub)
        if not os.path.isdir(d):
            continue
        for f in os.listdir(d):
            m = FNAME.match(f)
            if m and m.group(4) == tongue:
                sess, spk, room, _, ext = m.groups()
                files[(sess, spk, room)][ext] = os.path.join(d, f)
    out = []
    for (sess, spk, room), v in sorted(files.items()):
        if "wav" in v and "TextGrid" in v:
            out.append(dict(session=sess, speaker=spk, room="same" if room == "s" else "diff",
                            wav=v["wav"], textgrid=v["TextGrid"]))
    return out


def overlap(a0, a1, ivs) -> float:
    """Seconds of [a0, a1] covered by intervals ivs (sorted list of (start, end))."""
    tot = 0.0
    for s, e in ivs:
        if e <= a0:
            continue
        if s >= a1:
            break
        tot += min(a1, e) - max(a0, s)
    return tot
