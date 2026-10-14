"""Precision / recall of critical-term flags (proposal §4.3(d)).

A flag marks a critical word in the output. It is correct (TP) if any of its output units is
not aligned as correct to the reference. Recall: share of reference critical words with an
error (CWER numerator) whose aligned output units overlap a flag.
"""
from ..text.normalize import normalize
from ..text.units import label, split_units
from .align import C, align


def _flag_unit_idx(hyp_units, flags):
    idx = set()
    for f in flags:
        fu = split_units(normalize(f))
        n = len(fu)
        for i in range(len(hyp_units) - n + 1):
            if hyp_units[i: i + n] == fu:
                idx.update(range(i, i + n))
    return idx


def flag_counts(ref: str, hyp: str, flags: list[str]) -> dict:
    lab = label(ref)
    hu = split_units(normalize(hyp))
    ops = align(lab.units, hu)
    ok_hyp = {hj for op, _, hj in ops if op == C}
    ref2hyp = {}
    for op, ri, hj in ops:
        if ri is not None and hj is not None:
            ref2hyp[ri] = hj
    flagged = _flag_unit_idx(hu, flags)
    # group flagged output units into the flags themselves for precision
    tp = fp = 0
    for f in flags:
        fu = _flag_unit_idx(hu, [f])
        if not fu:
            continue
        if any(i not in ok_hyp for i in fu):
            tp += 1
        else:
            fp += 1
    ref_err = ref_err_flagged = 0
    correct_ref = {ri for op, ri, _ in ops if op == C}
    for s, e in lab.critical:
        if all(i in correct_ref for i in range(s, e)):
            continue
        ref_err += 1
        if any(ref2hyp.get(i) in flagged for i in range(s, e)):
            ref_err_flagged += 1
    return dict(tp=tp, fp=fp, ref_err=ref_err, ref_err_flagged=ref_err_flagged)


def flag_pr(rows) -> dict:
    t = dict(tp=0, fp=0, ref_err=0, ref_err_flagged=0)
    for r in rows:
        for k, v in flag_counts(r["ref"], r["hyp"], r.get("flags") or []).items():
            t[k] += v
    prec = t["tp"] / (t["tp"] + t["fp"]) if t["tp"] + t["fp"] else float("nan")
    rec = t["ref_err_flagged"] / t["ref_err"] if t["ref_err"] else float("nan")
    return dict(precision=prec, recall=rec, **t)
