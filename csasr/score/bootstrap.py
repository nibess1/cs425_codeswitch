"""Paired bootstrap confidence intervals, fragility ratio R and recovery rate Rec (§5).

Resample utterances with replacement B times, score both systems on the same
resample, report 2.5/97.5 percentiles. A difference is significant only if its
interval excludes 0. R is reported only if the CI of its denominator excludes 0.
"""
import numpy as np

from .metrics import NUM_FIELDS

B_DEFAULT = 1000


def to_arrays(stats: list[dict]) -> dict:
    a = {k: np.array([s[k] for s in stats], dtype=float) for k in NUM_FIELDS}
    a["cs_mask"] = np.array([bool(s["mixed"]) and s["main"] in ("zh", "en") for s in stats])
    return a


def _rate(a, idx, num, den, cs=False):
    """Vectorised ratio of sums over bootstrap index matrix idx (B x n)."""
    if cs:
        w = a["cs_mask"].astype(float)
        nu = (sum(a[k] for k in num) * w)[idx].sum(1)
        de = (sum(a[k] for k in den) * w)[idx].sum(1)
    else:
        nu = sum(a[k] for k in num)[idx].sum(1)
        de = sum(a[k] for k in den)[idx].sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return nu / de


METRICS = {
    "MER": (["err"], ["N"], False),
    "sMER": (["smer"], ["N"], False),
    "CWER": (["crit_err"], ["crit_n"], False),
    "E_sw": (["K_sw"], ["N_sw"], False),
    "E_E": (["K_E"], ["N_E"], False),
    "E_Ms": (["K_Ms"], ["N_Ms"], False),
    "E_Mi": (["K_Mi"], ["N_Mi"], False),
    "cs_E_emb": (["K_E"], ["N_E"], True),
    "cs_E_mat": (["K_Ms", "K_Mi"], ["N_Ms", "N_Mi"], True),
    "conf_rate": (["N_conf_E"], ["N_E"], False),
}


def boot_idx(n, B=B_DEFAULT, seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(0, n, size=(B, n))


def ci(x):
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return (float("nan"), float("nan"))
    return (float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5)))


def point(a, name):
    num, den, cs = METRICS[name]
    idx = np.arange(len(a["N"]))[None, :]
    return float(_rate(a, idx, num, den, cs)[0])


def metric_ci(a, name, idx):
    num, den, cs = METRICS[name]
    return ci(_rate(a, idx, num, den, cs))


def paired_diff(a, b, name, idx):
    """CI of metric(b) - metric(a); a and b must be aligned utterance-for-utterance."""
    num, den, cs = METRICS[name]
    d = _rate(b, idx, num, den, cs) - _rate(a, idx, num, den, cs)
    full = np.arange(len(a["N"]))[None, :]
    est = float(_rate(b, full, num, den, cs)[0] - _rate(a, full, num, den, cs)[0])
    lo, hi = ci(d)
    return {"diff": est, "lo": lo, "hi": hi, "sig": bool(lo > 0 or hi < 0)}


def fragility(a16, aq, idx):
    """R = (Eq_emb - E16_emb) / (Eq_mat - E16_mat) on code-switched utterances."""
    full = np.arange(len(a16["N"]))[None, :]
    e = lambda arr, ix: _rate(arr, ix, ["K_E"], ["N_E"], True)
    m = lambda arr, ix: _rate(arr, ix, ["K_Ms", "K_Mi"], ["N_Ms", "N_Mi"], True)
    num_b, den_b = e(aq, idx) - e(a16, idx), m(aq, idx) - m(a16, idx)
    num, den = float(e(aq, full)[0] - e(a16, full)[0]), float(m(aq, full)[0] - m(a16, full)[0])
    den_ci, num_ci = ci(den_b), ci(num_b)
    reportable = bool(den_ci[0] > 0 or den_ci[1] < 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        r_b = num_b / den_b
    return {"R": (num / den) if den else float("nan"), "R_ci": ci(r_b) if reportable else (None, None),
            "reportable": reportable, "d_emb": num, "d_emb_ci": num_ci, "d_mat": den, "d_mat_ci": den_ci}


def recovery(a16, aq, afix, idx):
    """Rec = (Eq_emb - Eqfix_emb) / (Eq_emb - E16_emb)."""
    full = np.arange(len(a16["N"]))[None, :]
    e = lambda arr, ix: _rate(arr, ix, ["K_E"], ["N_E"], True)
    num_b, den_b = e(aq, idx) - e(afix, idx), e(aq, idx) - e(a16, idx)
    num, den = float(e(aq, full)[0] - e(afix, full)[0]), float(e(aq, full)[0] - e(a16, full)[0])
    with np.errstate(invalid="ignore", divide="ignore"):
        rb = num_b / den_b
    den_ci = ci(den_b)
    return {"Rec": num / den if den else float("nan"),
            "Rec_ci": ci(rb) if (den_ci[0] > 0 or den_ci[1] < 0) else (None, None),
            "loss": den, "loss_ci": den_ci}
