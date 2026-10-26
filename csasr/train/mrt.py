"""Semantic minimum-risk loss (proposal §4.3(b), Appendix A2).

L_sem = sum_{y^ in N(x)} P^(y^|x) [C(y^, y) - C_bar],  P^ = p(y^|x) renormalised over N(x).
C counts 1 per wrong/missing reference unit, rho_i + (1 - rho_i)(1 - sim_i) for a unit in
a translation block (rho_i = 1 for critical words), and 1 per inserted unit outside
translation blocks.

Candidate set N(x): beam-4 outputs under <|zh|>, <|en|> and a free language token, plus one
candidate per embedded word in which that word is swapped for its dictionary translation.
Duplicates after normalisation are removed; clips whose candidates all cost the same are
skipped. Deviation from the proposal: candidates are generated from the current model on a
sub-batch every `mrt_every` steps (fresher than once per epoch), not precomputed.

Third run (26 Oct): optional transliteration penalty. The cost becomes
C(y^, y) + gamma * N_tl(y^), where N_tl counts embedded reference units that y^ renders as a
phonetic transliteration (csasr/score/translit.py, unchanged; `N_tl_E` from analyze()). C is a
raw unit count, so gamma is the extra cost per transliterated unit; gamma = 0 is the old loss.
"""
import time

import jieba
import torch
import torch.nn.functional as F

from ..decode.whisper_hf import Decoder
from ..score.confusion import LaBSE, en_to_zh, zh_to_en
from ..score.metrics import analyze
from ..text.normalize import normalize
from ..text.units import label


def swap_candidates(ref_text: str) -> list[str]:
    """One candidate per embedded word, with that word replaced by its dictionary translation."""
    lab = label(ref_text)
    if lab.main not in ("zh", "en") or not lab.mixed:
        return []
    out = []
    i, n = 0, len(lab.units)
    while i < n:
        if lab.roles[i] != "E":
            i += 1
            continue
        j = i
        while j < n and lab.roles[j] == "E":
            j += 1
        if lab.langs[i] == "en":  # each English word is its own embedded word
            words = [(k, k + 1) for k in range(i, j)]
        else:  # segment the embedded Chinese run into words
            words, k = [], i
            for w in jieba.lcut("".join(lab.units[i:j])):
                words.append((k, k + len(w))); k += len(w)
        for s, e in words:
            src = ("" if lab.langs[s] == "zh" else " ").join(lab.units[s:e])
            tr = en_to_zh(src) if lab.langs[s] == "en" else zh_to_en(src)
            if not tr:
                continue
            new = lab.units[:s] + [tr] + lab.units[e:]
            out.append(_join(new))
        i = j
    return out


def _join(units):
    s = ""
    for u in units:
        if s and (u[0].isascii() or s[-1].isascii()):
            s += " "
        s += u
    return s


def mrt_cost(stats: dict, gamma: float = 0.0) -> float:
    """Per-candidate MRT cost: semantic cost plus gamma per transliterated embedded unit."""
    if not gamma:
        return stats["smer"]
    return stats["smer"] + gamma * stats["N_tl_E"]


class SemanticMRT:
    def __init__(self, cfg):
        self.cfg = cfg
        self.labse = LaBSE(device="cuda")
        self.calls = 0

    def __call__(self, trainer, batch):
        cfg = self.cfg
        if trainer.step % cfg.mrt_every:
            return None
        rows = [trainer.train_rows[i] for i in batch["idx"][: cfg.mrt_batch]]
        feats = batch["input_features"][: cfg.mrt_batch].cuda()
        model, tok = trainer.model, trainer.tok
        dec = Decoder(model, tok, trainer.fe)
        dec.dtype = torch.float32

        # 1) candidates (no grad), from the current model
        model.eval()
        cands = [dict() for _ in rows]  # normalized text -> (lang, token ids)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for mode in ("zh", "en", "free"):
                r = dec.generate(feats, mode, num_beams=4, num_return_sequences=4)
                for k in range(len(r["texts"])):
                    ci = k // 4
                    lang = r["langs"][ci]
                    norm = normalize(r["texts"][k])
                    ids = [t for t in r["seqs"][k].tolist() if t != tok.eos_token_id]
                    cands[ci].setdefault(norm, (lang, ids))
        model.train()
        for ci, row in enumerate(rows):
            for s in swap_candidates(row["ref_raw"]):
                lang = "zh" if row["main"] == "zh" else "en"
                text = " " + s if s[:1].isascii() else s
                cands[ci].setdefault(normalize(s), (lang, tok(text, add_special_tokens=False).input_ids))

        # 2) costs C(y^, y)
        pairs, owners = [], []
        for ci, row in enumerate(rows):
            for norm in cands[ci]:
                pairs.append((row["ref_raw"], norm)); owners.append(ci)
        t_an = time.time()
        st = analyze(pairs, labse=self.labse, rho0=cfg.rho0)
        t_an = time.time() - t_an
        gamma = getattr(cfg, "translit_gamma", 0.0)
        costs = [mrt_cost(s, gamma) for s in st]
        diag = {"t_analyze": t_an, "tl_cand_share": sum(s["N_tl_E"] > 0 for s in st) / max(1, len(st)),
                "tl_units_mean": sum(s["N_tl_E"] for s in st) / max(1, len(st))}
        self.calls += 1
        if self.calls % 50 == 0:
            self.labse.save()

        # 3) sequence log-probs with grad, teacher forced
        seqs, cost_t, grp = [], [], []
        by = {}
        for (ci, norm_cost) in zip(owners, costs):
            by.setdefault(ci, []).append(norm_cost)
        keep = {ci for ci, cs in by.items() if len(cs) > 1 and max(cs) > min(cs)}
        if not keep:
            return {"loss": feats.new_zeros(()), "mrt_skipped": 1.0, **diag}
        k = 0
        for ci in range(len(rows)):
            for norm, (lang, ids) in cands[ci].items():
                if ci in keep:
                    seqs.append([dec.sot, dec.lang[lang], dec.trans, dec.notime] + ids[:128] + [dec.eot])
                    cost_t.append(costs[k]); grp.append(ci)
                k += 1
        T = max(len(s) for s in seqs)
        inp = torch.full((len(seqs), T - 1), tok.eos_token_id, device="cuda")
        tgt = torch.full((len(seqs), T - 1), -100, device="cuda")
        for i, s in enumerate(seqs):
            inp[i, : len(s) - 1] = torch.tensor(s[:-1]); tgt[i, : len(s) - 1] = torch.tensor(s[1:])
        tgt[:, 1:3] = -100  # task / no-timestamps tokens are forced, not scored
        sel = sorted(keep)
        pos = {ci: i for i, ci in enumerate(sel)}
        enc = model.get_encoder()(feats[sel]).last_hidden_state
        enc = enc[torch.tensor([pos[g] for g in grp], device="cuda")]
        logits = model(encoder_outputs=(enc,), decoder_input_ids=inp).logits.float()
        lp = -F.cross_entropy(logits.transpose(1, 2), tgt, ignore_index=-100, reduction="none").sum(1)
        cost_t = torch.tensor(cost_t, device="cuda")
        grp_t = torch.tensor(grp, device="cuda")
        loss = 0.0
        for ci in sel:
            m = grp_t == ci
            p = torch.softmax(lp[m], 0)
            c = cost_t[m]
            loss = loss + (p * (c - c.mean())).sum()
        loss = loss / len(sel)
        return {"loss": cfg.lambda_sem * loss, "l_sem": float(loss), "n_cand": len(seqs) / len(sel), **diag}
