"""Frame-level language-ID head on Whisper's frozen encoder (proposal §4.3(c), Appendix A3).

L_LID = - sum_f v_f log q(l_f | h_f) / sum_f v_f  over 20 ms frames f (1500 per 30 s window),
v_f = 0 for padding and for frames labelled ignore. Labels: 0 en, 1 zh, 2 non-speech.
Predictions are smoothed with a 5-frame median filter and segments < 200 ms are merged into
their neighbours.
"""
import numpy as np
import torch
import torch.nn as nn
from scipy.ndimage import median_filter

N_FRAMES = 1500
FRAME = 0.02
CLASSES = ("en", "zh", "sil")


class LIDHead(nn.Module):
    def __init__(self, d_model: int, hidden: int = 256, n_classes: int = 3, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, hidden), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(hidden, n_classes))

    def forward(self, h):
        return self.net(h)


def pad_labels(frames: np.ndarray) -> np.ndarray:
    out = np.full(N_FRAMES, -1, dtype=np.int64)  # padding -> ignore (v_f = 0)
    n = min(len(frames), N_FRAMES)
    out[:n] = frames[:n]
    return out


def smooth(pred: np.ndarray, min_seg: float = 0.2) -> np.ndarray:
    """5-frame median filter, then repeatedly merge the shortest segment (< min_seg) into
    its longer neighbour."""
    p = median_filter(pred, size=5, mode="nearest")
    min_len = int(round(min_seg / FRAME))
    while True:
        b = np.flatnonzero(np.diff(p)) + 1
        starts, ends = np.r_[0, b], np.r_[b, len(p)]
        if len(starts) <= 1:
            return p
        lens = ends - starts
        k = int(np.argmin(lens))
        if lens[k] >= min_len:
            return p
        if k == 0:
            fill = p[ends[0]]
        elif k == len(starts) - 1:
            fill = p[starts[k] - 1]
        else:
            fill = p[starts[k] - 1] if lens[k - 1] >= lens[k + 1] else p[ends[k]]
        p[starts[k]: ends[k]] = fill


def switch_points(lab: np.ndarray) -> list[int]:
    """Frame indices where the language (en/zh) changes, ignoring silence and ignore frames."""
    pts, last, last_f = [], None, None
    for f, l in enumerate(lab):
        if l in (0, 1):
            if last is not None and l != last:
                pts.append((last_f + f) // 2)
            last, last_f = l, f
    return pts


def boundary_error(true: np.ndarray, pred: np.ndarray) -> list[float]:
    """For each true switch, distance (s) to the nearest predicted switch (nan if none)."""
    tp, pp = switch_points(true), switch_points(pred)
    return [min(abs(t - p) for p in pp) * FRAME if pp else float("nan") for t in tp]
