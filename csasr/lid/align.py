"""Unit-level forced alignment with the MMS aligner (torchaudio MMS_FA) on uroman-romanized
transcripts (proposal §3.3).

Every unit (Chinese character, English word, digit string) becomes one aligner "word":
Chinese characters are romanized one at a time (我 -> wo), English words are kept,
digit strings map to the wildcard '*'. Output: per-unit (start, end) seconds and 20 ms
frame labels {0: en, 1: zh, 2: non-speech, -1: ignore (neutral units)}.
"""
import numpy as np
import torch
import torchaudio

from ..text.units import split_units, unit_lang

FRAME = 0.02
LAB = {"en": 0, "zh": 1, "sil": 2, "neu": -1}
_ur = None


def uroman():
    global _ur
    if _ur is None:
        import uroman as U
        _ur = U.Uroman()
    return _ur


def romanize_unit(u: str, allowed: set) -> str:
    if u.isdigit():
        return "*"
    r = uroman().romanize_string(u).lower() if not u.isascii() else u.lower()
    r = "".join(c for c in r if c in allowed)
    return r or "*"


class Aligner:
    def __init__(self, device="cuda"):
        b = torchaudio.pipelines.MMS_FA
        self.model = b.get_model(with_star=True).to(device).eval()
        self.dict = b.get_dict(star="*")
        self.allowed = set(self.dict) - {"-", "*"}
        self.device = device

    @torch.no_grad()
    def emissions(self, wavs: list[np.ndarray]):
        lens = torch.tensor([len(w) for w in wavs])
        x = torch.zeros(len(wavs), int(lens.max()))
        for i, w in enumerate(wavs):
            x[i, : len(w)] = torch.from_numpy(w)
        em, out_lens = self.model(x.to(self.device), lens.to(self.device))
        return em.log_softmax(-1), out_lens  # idempotent if the model already emits log-probs

    def align_one(self, emission, n_frames, units, wav_len):
        words = [romanize_unit(u, self.allowed) for u in units]
        tokens = [self.dict[c] for w in words for c in w]
        if not tokens:
            return None
        em = emission[:n_frames].unsqueeze(0)
        if n_frames < len(tokens):
            return None
        ali, scores = torchaudio.functional.forced_align(em, torch.tensor([tokens], device=em.device), blank=0)
        spans = torchaudio.functional.merge_tokens(ali[0], scores[0].exp())
        sec_per_frame = wav_len / 16000 / n_frames
        out, k = [], 0
        for w in words:
            n = len(w)
            ws = spans[k: k + n]
            k += n
            out.append((ws[0].start * sec_per_frame, ws[-1].end * sec_per_frame,
                        float(np.mean([s.score for s in ws]))))
        return out


def frame_labels(units, spans, dur, frame=FRAME) -> np.ndarray:
    n = int(np.ceil(dur / frame))
    lab = np.full(n, LAB["sil"], dtype=np.int8)
    for u, (s, e, _) in zip(units, spans):
        l = unit_lang(u)
        a, b = int(np.floor(s / frame)), int(np.ceil(e / frame))
        # CTC lets a unit swallow neighbouring silence: label at most a plausible length and
        # mark the overflow as ignore rather than as language
        cap = 0.5 if l == "zh" else 0.2 + 0.12 * len(u)
        c = min(b, a + int(np.ceil(cap / frame)))
        lab[a:b] = LAB["neu"]
        lab[a:c] = LAB[l]
    return lab


def align_rows(aligner: Aligner, rows, load_audio, text_key="ref", batch=1):  # with_star breaks on batches
    """Yield (row, units, spans) for each manifest row; spans None if alignment failed."""
    order = sorted(range(len(rows)), key=lambda i: rows[i]["duration"])
    for b0 in range(0, len(order), batch):
        idx = order[b0: b0 + batch]
        wavs = [load_audio(rows[i]["audio"]) for i in idx]
        em, lens = aligner.emissions(wavs)
        for j, i in enumerate(idx):
            units = split_units(rows[i][text_key])
            try:
                spans = aligner.align_one(em[j], int(lens[j]), units, len(wavs[j]))
            except Exception:
                spans = None
            yield rows[i], units, spans

