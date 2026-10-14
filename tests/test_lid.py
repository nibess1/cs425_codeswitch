import numpy as np

from csasr.lid.head import boundary_error, smooth, switch_points


def test_smooth_merges_short_language_runs_keeps_pauses():
    p = np.array([0] * 30 + [1] * 4 + [0] * 30 + [1] * 40)
    s = smooth(p.copy())
    assert (s[:64] == 0).all() and (s[64:] == 1).all()
    q = np.array([0] * 20 + [2] * 6 + [0] * 20 + [2] * 6 + [1] * 20)
    t = smooth(q.copy())
    assert (t[20:26] == 2).all() and (t[46:52] == 2).all() and (t[52:] == 1).all()


def test_switch_points_ignore_silence():
    lab = np.array([0, 0, 2, 2, 1, 1, -1, 1, 0])
    assert switch_points(lab) == [2, 7]
    assert boundary_error(lab, lab) == [0.0, 0.0]
