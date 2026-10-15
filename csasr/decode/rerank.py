"""Language-aware N-best re-ranking (proposal §4.3(d), Appendix A4).

s(y^) = log p_bar_theta(y^|x) + gamma log p_bar_CTC(y~|x) + eta a(y^)
a(y^) = 1/|y^| sum_u 1/|F_u| sum_{f in F_u} log q(l(u) | h_f)

Candidates: beam-4 outputs (4 returned) under <|zh|>, <|en|> and a free language token.
- log p_bar_theta: Whisper log-prob of the candidate (language token + text + eot) / tokens.
- log p_bar_CTC: MMS CTC Viterbi log-prob of the uroman-romanized candidate / characters.
- F_u: frames of unit u from the MMS alignment (with-MMS variant) or from Whisper
  cross-attention word timestamps (on-device variant, gamma = 0).
Critical terms are flagged (not corrected) when the top candidates disagree on them.
"""
import zlib

import librosa
import numpy as np
import torch
import torchaudio
from scipy.ndimage import median_filter

from ..lid.align import romanize_unit
from ..lid.head import FRAME
from ..text.normalize import normalize
from ..text.units import label, split_units, unit_lang

LANG_ID = {"en": 0, "zh": 1}


def dtw(cost: np.ndarray):
    """Monotonic alignment of tokens (rows) to frames (cols): librosa's numba DTW."""
    _, wp = librosa.sequence.dtw(C=cost, backtrack=True)
    wp = wp[::-1]
    return wp[:, 0], wp[:, 1]


def token_frames_from_attention(attn: np.ndarray, n_frames: int) -> list[tuple[int, int]]:
    """attn: [heads, T_text, 1500] cross-attention of alignment heads (text tokens only).
    Returns [start, end) frame per text token (OpenAI-style: normalise, median filter, DTW)."""
    w = attn[:, :, :n_frames]
    w = (w - w.mean(1, keepdims=True)) / (w.std(1, keepdims=True) + 1e-6)
    w = median_filter(w, size=(1, 1, 7))
    m = w.mean(0)
    rows, cols = dtw(-m.astype(np.float64))
    starts = np.zeros(m.shape[0], dtype=int)
    seen = set()
    for r, c in zip(rows, cols):
        if r not in seen:
            starts[r] = c; seen.add(r)
    ends = np.r_[starts[1:], n_frames]
    return [(int(s), int(max(e, s + 1))) for s, e in zip(starts, ends)]


def unit_frames_from_tokens(text: str, ids, tok_frames, spanner) -> tuple[list[str], list]:
    """Map token frame spans to units through byte spans. `text` is the exact tokenised
    string (a normalised training target, optionally with a leading space)."""
    from ..train.data import unit_byte_spans
    units = split_units(text)
    u_spans = unit_byte_spans(text, units)
    t_spans = spanner.spans(ids)
    out = []
    for us, ue in u_spans:
        fr = [tok_frames[k] for k, (ts, te) in enumerate(t_spans) if ts < ue and te > us]
        out.append((min(f[0] for f in fr), max(f[1] for f in fr)) if fr else None)
    return units, out


def agreement(units, frames, lid_logp: np.ndarray) -> float:
    """a(y^): mean over non-neutral units of the mean frame log q(lang(u))."""
    vals = []
    for u, fr in zip(units, frames):
        l = unit_lang(u)
        if l not in LANG_ID or fr is None:
            continue
        a, b = fr
        b = min(b, len(lid_logp))
        if b <= a:
            continue
        vals.append(float(lid_logp[a:b, LANG_ID[l]].mean()))
    return float(np.mean(vals)) if vals else 0.0


class CTCScorer:
    def __init__(self, aligner):
        self.al = aligner

    def emission(self, wav: np.ndarray):
        em, lens = self.al.emissions([wav])
        return em[0, : int(lens[0])]

    def score(self, em, text: str, wav_len: int):
        """(log p_CTC / chars, unit frame spans in 20 ms frames) or (-inf, None)."""
        units = split_units(normalize(text))
        words = [romanize_unit(u, self.al.allowed) for u in units]
        tokens = [self.al.dict[c] for w in words for c in w]
        repeats = sum(a == b for a, b in zip(tokens, tokens[1:]))
        if not tokens or len(tokens) + repeats > em.shape[0]:
            return -np.inf, units, None  # candidate too long for the audio: impossible under CTC
        try:
            ali, scores = torchaudio.functional.forced_align(em.unsqueeze(0), torch.tensor([tokens], device=em.device),
                                                             blank=0)
        except RuntimeError:
            return -np.inf, units, None
        lp = float(scores[0].sum())
        spans = torchaudio.functional.merge_tokens(ali[0], scores[0].exp())
        sec = wav_len / 16000 / em.shape[0]
        out, k = [], 0
        for w in words:
            ws = spans[k: k + len(w)]
            k += len(w)
            out.append((int(ws[0].start * sec / FRAME), int(np.ceil(ws[-1].end * sec / FRAME))))
        return lp / len(tokens), units, out


def critical_words(text: str) -> list[str]:
    lab = label(text)
    return [("" if lab.langs[s] == "zh" else " ").join(lab.units[s:e]) for s, e in lab.critical]


def flags_for(top: list[str]) -> list[str]:
    """Critical words of the best candidate that are not reproduced by the other top ones."""
    if len(top) < 2:
        return []
    others = [set(critical_words(t)) | set(split_units(normalize(t))) for t in top[1:]]
    flags = []
    for w in critical_words(top[0]):
        if not all(w in o or w.replace(" ", "") in normalize(t) for o, t in zip(others, top[1:])):
            flags.append(w)
    return flags


def ranked(cands, gamma, eta):
    def s(c):
        ctc = c["ctc"] if np.isfinite(c["ctc"]) else -50.0
        return c["lp"] + gamma * ctc + eta * (c["a_mms"] if gamma > 0 else c["a_wts"])
    return sorted(cands, key=s, reverse=True)



def choose(cands, gamma, eta):
    return ranked(cands, gamma, eta)[0]


def compression_ratio(text: str) -> float:
    """Whisper's repetition heuristic: UTF-8 length / zlib-compressed length (> 2.4 = loop)."""
    b = text.encode("utf-8")
    return len(b) / max(1, len(zlib.compress(b)))
