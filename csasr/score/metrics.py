"""Per-utterance statistics and corpus metrics (proposal §5).

analyze() turns (ref, hyp) pairs into additive per-utterance counts; every corpus
metric is a ratio of sums of these counts, which makes the paired bootstrap cheap.
"""
import numpy as np

from ..text.normalize import normalize
from ..text.units import label, split_units, unit_lang
from .align import score
from .confusion import DEFAULT_TAU, cedict_match

ROLES = ("E", "Ms", "Mi", "neu", "bal")
NUM_FIELDS = (["N", "S", "D", "I", "err", "orphan", "N_sw", "K_sw", "N_conf_E", "smer", "crit_n", "crit_err", "mixed"]
              + [f"N_{r}" for r in ROLES] + [f"K_{r}" for r in ROLES] + [f"X_{r}" for r in ROLES])


def _structural_confusion(b, lab, hyp_langs):
    """Ref side all embedded, output side non-empty and entirely in the main language."""
    if not b.ref_idx or not b.hyp_idx or lab.main not in ("zh", "en"):
        return False
    if any(lab.roles[i] != "E" for i in b.ref_idx):
        return False
    return all(hyp_langs[j] == lab.main for j in b.hyp_idx)


def _side(units, idx, lang):
    sep = "" if lang == "zh" else " "
    return sep.join(units[i] for i in idx)


def analyze(pairs, labse=None, tau=DEFAULT_TAU, rho0=0.5, keep_units=False):
    """pairs: iterable of (ref_text, hyp_text). Returns list of dicts of additive counts.

    If `labse` (confusion.LaBSE) is given, translation = CEDICT or sim >= tau, and sim
    is used for sMER partial credit; otherwise CEDICT only with sim = 1.
    """
    prepared = []
    sim_pairs = []
    for ref, hyp in pairs:
        lab = label(ref)
        hyp_units = split_units(normalize(hyp))
        hyp_langs = [unit_lang(u) for u in hyp_units]
        sc = score(lab.units, hyp_units)
        cands = []
        for b in sc.blocks:
            if _structural_confusion(b, lab, hyp_langs):
                emb_lang = "en" if lab.main == "zh" else "zh"
                r = _side(lab.units, b.ref_idx, emb_lang)
                h = _side(hyp_units, b.hyp_idx, lab.main)
                zh, en = (h, r) if lab.main == "zh" else (r, h)
                cands.append((b, r, h, cedict_match(zh, en)))
                if labse is not None:
                    sim_pairs.append((r, h))
        prepared.append((lab, hyp_units, hyp_langs, sc, cands))

    sims = {}
    if labse is not None and sim_pairs:
        for (a, b), s in zip(sim_pairs, labse.sims(sim_pairs)):
            sims[(a, b)] = s

    out = []
    for lab, hyp_units, hyp_langs, sc, cands in prepared:
        for b, r, h, dict_hit in cands:
            sim = sims.get((r, h), 1.0 if dict_hit else 0.0) if labse is not None else (1.0 if dict_hit else 0.0)
            b.sim = sim
            b.confusion = dict_hit or (labse is not None and sim >= tau)
        out.append(_counts(lab, hyp_units, sc, rho0, keep_units))
    return out


def _counts(lab, hyp_units, sc, rho0, keep_units):
    roles = np.array(lab.roles) if lab.units else np.array([], dtype=str)
    d = {k: 0.0 for k in NUM_FIELDS}
    d.update(N=sc.n_ref, S=sc.S, D=sc.D, I=sc.I, err=sc.errors, orphan=sc.orphan, mixed=float(lab.mixed))
    for r in ROLES:
        m = roles == r
        d[f"N_{r}"] = float(m.sum())
        d[f"K_{r}"] = float(sc.kappa[m].sum())
        d[f"X_{r}"] = float(sc.unit_err[m].sum())
    sw = np.array(lab.switch, dtype=bool) if lab.units else np.array([], dtype=bool)
    d["N_sw"] = float(sw.sum())
    d["K_sw"] = float(sc.kappa[sw].sum())

    # critical words
    crit_unit = np.zeros(sc.n_ref, dtype=bool)
    for s, e in lab.critical:
        crit_unit[s:e] = True
        d["crit_n"] += 1
        d["crit_err"] += float(not sc.unit_correct[s:e].all())

    # semantic MER cost (A2 with fixed rho0): correct 0; translation block rho+(1-rho)(1-sim); else 1
    cost = 0.0
    for b in sc.blocks:
        if b.confusion:
            for i in b.ref_idx:
                rho = 1.0 if crit_unit[i] else rho0
                cost += rho + (1 - rho) * (1 - b.sim)
                d["N_conf_E"] += 1
        else:
            cost += len(b.ref_idx) + b.n_ins
    d["smer"] = cost
    d["main"] = lab.main
    if keep_units:
        d["_units"] = dict(units=lab.units, langs=lab.langs, roles=lab.roles, err=sc.unit_err.tolist(),
                           kappa=sc.kappa.tolist(), hyp=hyp_units)
    return d


# ---------------------------------------------------------------- aggregation
def _ratio(a, b):
    return float(a / b) if b else float("nan")


def corpus_metrics(stats: list[dict] | dict) -> dict:
    """stats: list of per-utterance dicts, or a dict of summed arrays."""
    if isinstance(stats, list):
        t = {k: sum(s[k] for s in stats) for k in NUM_FIELDS}
        cs = [s for s in stats if s["mixed"] and s["main"] in ("zh", "en")]
        tcs = {k: sum(s[k] for s in cs) for k in NUM_FIELDS}
    else:
        t, tcs = stats["all"], stats["cs"]
    m = {
        "n_utt": None,
        "N": t["N"],
        "MER": _ratio(t["err"], t["N"]),
        "sMER": _ratio(t["smer"], t["N"]),
        "CWER": _ratio(t["crit_err"], t["crit_n"]),
        "E_sw": _ratio(t["K_sw"], t["N_sw"]),
        "N_crit": t["crit_n"],
    }
    for r in ROLES:
        m[f"E_{r}"] = _ratio(t[f"K_{r}"], t[f"N_{r}"])
        m[f"N_{r}"] = t[f"N_{r}"]
    # on code-switched (non-balanced) utterances only: embedded vs matrix
    m["cs_E_emb"] = _ratio(tcs["K_E"], tcs["N_E"])
    m["cs_E_mat"] = _ratio(tcs["K_Ms"] + tcs["K_Mi"], tcs["N_Ms"] + tcs["N_Mi"])
    m["cs_E_Ms"] = _ratio(tcs["K_Ms"], tcs["N_Ms"])
    m["cs_E_Mi"] = _ratio(tcs["K_Mi"], tcs["N_Mi"])
    m["cs_N_emb"] = tcs["N_E"]
    m["conf_rate"] = _ratio(t["N_conf_E"], t["N_E"])
    return m
