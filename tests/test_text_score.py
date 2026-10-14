import random

import numpy as np
import pytest

from csasr.score import metrics as M
from csasr.score.align import score
from csasr.score.bootstrap import boot_idx, fragility, to_arrays
from csasr.text.normalize import normalize
from csasr.text.units import label, split_units, training_target


def test_normalize_basic():
    assert normalize("我明天要交 Assignment 啦！") == "我明天要交 assignment 啦"
    assert normalize("這個 E-mail，OK？") == "这个 email ok"
    assert normalize("十号") == normalize("10号") == "10号"
    assert normalize("一起去") == "一起去"


def test_normalize_markers():
    nsc = "<mandarin>特地:te di</mandarin> go and find you (uh) [lah] !wah! #jiak# engli~ <Z> (ppl)"
    assert normalize(nsc) == "特地 go and find you lah wah jiak"
    assert normalize("[UNK] 小朋友 when i was") == "小朋友 when i was"
    assert normalize("I don't know") == "i dont know"


def test_units_and_target():
    assert split_units("我明天要交 assignment 啦") == ["我", "明", "天", "要", "交", "assignment", "啦"]
    assert split_units("10号 mp3") == ["10", "号", "mp3"]
    assert training_target("我要交assignment啦") == "我要交 assignment 啦"


def test_roles_proposal_example():
    lab = label("我明天要交 assignment 啦")
    assert lab.main == "zh"
    assert lab.roles == ["Mi", "Mi", "Mi", "Mi", "Ms", "E", "Ms"]
    assert lab.switch == [False, False, False, False, True, True, True]


def test_neutral_and_balanced():
    lab = label("ok lah 我")
    assert lab.langs == ["en", "neu", "zh"]
    assert lab.main == "balanced"
    assert lab.roles == ["bal", "neu", "bal"]


def test_translation_block_charging():
    ref = split_units("我明天要交 assignment 啦")
    hyp = split_units("我明天要交作业啦")
    sc = score(ref, hyp)
    assert (sc.S, sc.D, sc.I) == (1, 0, 1)
    assert len(sc.blocks) == 1
    assert sc.kappa.tolist() == [0, 0, 0, 0, 0, 2, 0]  # nothing leaks onto 啦


def test_pure_insertion_shared():
    sc = score(["a", "b"], ["a", "x", "b"])
    assert sc.kappa.tolist() == [0.5, 0.5]
    sc = score(["a", "b"], ["a", "b", "x"])
    assert sc.kappa.tolist() == [0, 1]


def _lev(a, b):
    d = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        p, d[0] = d[0], i
        for j, y in enumerate(b, 1):
            p, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, p + (x != y))
    return d[-1]


def test_a7_identity_and_edit_distance_fuzz():
    rng = random.Random(0)
    V = list("abcde") + ["我", "你"]
    for _ in range(500):
        r = [rng.choice(V) for _ in range(rng.randint(0, 12))]
        h = [rng.choice(V) for _ in range(rng.randint(0, 12))]
        sc = score(r, h)
        assert sc.S + sc.D + sc.I == _lev(r, h)
        assert abs(sc.kappa.sum() + sc.orphan - sc.errors) < 1e-9


def test_role_rates_add_back_to_mer():
    pairs = [("我明天要交 assignment 啦", "我明天要交作业啦"),
             ("then we go 吃饭 lah", "then we go eat lah"),
             ("你好 hello", "你好 hello world"),
             ("ok lah 我", "ok 我")]
    st = M.analyze(pairs)
    tot_k = sum(s[f"K_{r}"] for s in st for r in M.ROLES) + sum(s["orphan"] for s in st)
    assert abs(tot_k - sum(s["err"] for s in st)) < 1e-9


def test_confusion_and_smer(monkeypatch):
    monkeypatch.setattr(M, "cedict_match", lambda zh, en: (zh, en) in {("作业", "assignment"), ("吃饭", "eat")})
    st = M.analyze([("我明天要交 assignment 啦", "我明天要交作业啦"), ("then we go 吃饭 lah", "then we go eat lah")])
    m = M.corpus_metrics(st)
    assert m["conf_rate"] == 1.0  # assignment (1 unit) + 吃饭 (2 units) all in translation blocks
    # dictionary-only mode (sim = 1): "assignment" is a critical term (rho = 1 -> cost 1),
    # 吃饭 is not (rho0 = 0.5 per unit -> cost 1); N = 7 + 6 (lah is a unit)
    assert abs(m["sMER"] - 2.0 / 13) < 1e-9
    assert m["MER"] == pytest.approx(4 / 13)


def test_critical_cwer():
    st = M.analyze([("deadline 是 十号", "deadline 是 11号")])
    m = M.corpus_metrics(st)
    assert m["N_crit"] == 2 and m["CWER"] == 0.5


def test_identity_zero_and_fragility_shape():
    pairs = [("我明天要交 assignment 啦", "我明天要交 assignment 啦")] * 5
    a = to_arrays(M.analyze(pairs))
    assert M.corpus_metrics(M.analyze(pairs))["MER"] == 0
    q = to_arrays(M.analyze([("我明天要交 assignment 啦", "我明天要交 啦")] * 5))
    f = fragility(a, q, boot_idx(5, 50))
    assert np.isnan(f["R"]) and not f["reportable"]  # matrix error unchanged -> R not reported
