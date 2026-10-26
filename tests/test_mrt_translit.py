"""Third run: transliteration-penalised MRT cost (csasr/train/mrt.py `mrt_cost`)."""
import pytest

pytest.importorskip("torch")

from csasr.score.metrics import analyze  # noqa: E402
from csasr.train.mrt import mrt_cost  # noqa: E402

REF = "我的 deadline 是明天"
PAIRS = [
    (REF, REF),
    (REF, "我的得来是明天"),          # transliteration of the embedded word
    (REF, "我的苹果是明天"),          # unrelated substitution, same semantic cost
    (REF, "我的期限是明天"),          # dictionary translation (partial credit)
    (REF, "我的 dateline 是明天"),
    ("then we go 吃饭 lah", "then we go chiffon lah"),
    ("我明天要交 assignment 啦", "我明天要交啊塞们啦"),
    ("ok lor", "好啦"),
]


def test_gamma_zero_is_old_cost():
    for s in analyze(PAIRS):
        assert mrt_cost(s, 0.0) == s["smer"]
        assert mrt_cost(s) == s["smer"]


def test_transliteration_costs_more():
    tl, other = analyze([(REF, "我的得来是明天"), (REF, "我的苹果是明天")])
    assert tl["N_tl_E"] >= 1 and other["N_tl_E"] == 0
    assert mrt_cost(tl, 0.0) == mrt_cost(other, 0.0)  # same semantic cost without the penalty
    for g in (0.5, 1.0):
        assert mrt_cost(tl, g) > mrt_cost(other, g)
        assert mrt_cost(tl, g) == tl["smer"] + g * tl["N_tl_E"]


def test_penalty_only_touches_transliterations():
    for s in analyze(PAIRS):
        if s["N_tl_E"] == 0:
            assert mrt_cost(s, 1.0) == s["smer"]
