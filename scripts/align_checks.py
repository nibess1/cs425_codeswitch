#!/usr/bin/env python3
"""D4: alignment quality without human labels -> results/align_checks.json

(a) MMS boundary error at KNOWN switch times: 300 fresh FLEURS splices built like the LID
    training splices but keeping each piece's text and true boundary; MMS aligns the joined
    text; error = |MMS switch time - true switch time| at every en<->zh change.
(b) MMS vs Whisper cross-attention (DTW over alignment heads, fine-tuned base, teacher-forced
    on the reference) at switch points, 500 ASCEND + NSC test utterances with switches.
(c) MMS first-word start / last-word end vs the NSC TextGrid interval edges (clips are cut at
    the interval edges, so the edges are 0 and the clip duration).
  CUDA_VISIBLE_DEVICES=7 python scripts/align_checks.py
"""
import json
import os
import pickle
import random
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from csasr.data import manifest  # noqa: E402
from csasr.decode import rerank as RR  # noqa: E402
from csasr.decode.whisper_hf import Decoder, load_model, load_processor  # noqa: E402
from csasr.lid.align import Aligner, align_rows  # noqa: E402
from csasr.text.units import split_units, training_target, unit_lang  # noqa: E402
from csasr.train.data import TokenSpans  # noqa: E402

FR = 0.02


def stats(errs):
    e = np.abs(np.array(errs, dtype=float))
    return dict(n=int(len(e)), median_ms=round(1000 * float(np.median(e)), 1) if len(e) else None,
                mean_ms=round(1000 * float(e.mean()), 1) if len(e) else None,
                within_50ms_pct=round(100 * float((e <= 0.05).mean()), 1) if len(e) else None,
                within_100ms_pct=round(100 * float((e <= 0.1).mean()), 1) if len(e) else None)


def switch_times(units, spans):
    """Times of en<->zh changes (midpoint between the two units), ignoring neutral units."""
    out, prev = [], None
    for u, sp in zip(units, spans):
        l = unit_lang(u)
        if l not in ("en", "zh") or sp is None:
            continue
        if prev is not None and l != prev[0]:
            out.append((prev[1] + sp[0]) / 2)
        prev = (l, sp[1])
    return out


def check_a(al, n=300, seed=425):
    import prepare_fleurs as PF
    rows = {l: manifest.load(f"data/fleurs/manifests/{l}_train.jsonl") for l in ("en", "zh")}
    pool = {"en": [], "zh": []}
    for l in ("en", "zh"):
        for r, units, spans in align_rows(al, rows[l][:600], PF.load16):
            if spans:
                pool[l].append((r, units, spans))
    rng = random.Random(seed)
    errs = []
    for _ in range(n):
        lang = rng.choice(["en", "zh"])
        wav, txt, t, true_sw = [], [], 0.0, []
        for p in range(rng.randint(3, 6)):
            r, units, spans = rng.choice(pool[lang])
            k = rng.randrange(len(units))
            m = min(len(units) - k, rng.choice([1, 2, 3]) if lang == "en" else rng.choice([2, 3, 4]))
            s0, e0 = spans[k][0], spans[k + m - 1][1]
            seg = PF.load16(r["audio"])[int(s0 * 16000): int(e0 * 16000)]
            if p > 0 and lang != prev_lang:
                true_sw.append(t - 0.05)  # midpoint of the 100 ms gap before this piece
            gap = np.zeros(int(0.1 * 16000), dtype=np.float32)
            wav += [seg, gap]
            txt += units[k: k + m]
            t += (len(seg) + len(gap)) / 16000
            prev_lang = lang
            if rng.random() < 0.6:
                lang = "zh" if lang == "en" else "en"
        w = np.concatenate(wav)
        em, lens = al.emissions([w])
        sp = al.align_one(em[0], int(lens[0]), txt, len(w))
        if not sp:
            continue
        mms_sw = switch_times(txt, [(a, b) for a, b, _ in sp])
        for ts in true_sw:
            if mms_sw:
                errs.append(min(abs(ts - m) for m in mms_sw))
    return stats(errs)


def check_b(n=500, seed=425):
    tok, fe = load_processor("small")
    model = load_model(manifest.path("ckpt/ft_base/best"), "fp16")
    dec = Decoder(model, tok, fe)
    heads = [tuple(h) for h in model.generation_config.alignment_heads]
    spanner = TokenSpans(tok)
    cand = []
    for s in ("ascend_test", "nsc_test"):
        al = pickle.load(open(manifest.path(f"data/align/{s}.pkl"), "rb"))
        for r in manifest.load(s):
            if r["mixed"] and r["utt_id"] in al:
                cand.append((r, al[r["utt_id"]]))
    random.Random(seed).shuffle(cand)
    errs = []
    for r, a in cand[:n]:
        tgt = training_target(r["ref_raw"])
        if not tgt:
            continue
        text = " " + tgt if tgt[0].isascii() else tgt
        ids = tok(text, add_special_tokens=False).input_ids[:128]
        wav = manifest.load_audio(r["audio"])
        x = torch.from_numpy(fe(wav, sampling_rate=16000, return_tensors="np").input_features).cuda().half()
        lang = r["main"] if r["main"] in ("zh", "en") else "zh"
        seq = [dec.sot, dec.lang[lang], dec.trans, dec.notime] + ids
        with torch.no_grad():
            o = model(input_features=x, decoder_input_ids=torch.tensor([seq]).cuda(), output_attentions=True)
        att = torch.stack([o.cross_attentions[l][0, h] for l, h in heads]).float().cpu().numpy()[:, 3: 3 + len(ids)]
        n_frames = min(1500, int(np.ceil(len(wav) / 16000 / FR)))
        tf = RR.token_frames_from_attention(att, n_frames)
        units, wts = RR.unit_frames_from_tokens(text, ids, tf, spanner)
        w_sw = switch_times(units, [(f[0] * FR, f[1] * FR) if f else None for f in wts])
        m_sw = switch_times(a["units"], [(s0, e0) for s0, e0, _ in a["spans"]])
        for t in m_sw:
            if w_sw:
                errs.append(min(abs(t - w) for w in w_sw))
    return stats(errs)


def check_c():
    out = {}
    for s in ("nsc_test", "nsc_diff_test"):
        al = pickle.load(open(manifest.path(f"data/align/{s}.pkl"), "rb"))
        dur = {r["utt_id"]: r["duration"] for r in manifest.load(s)}
        start = [v["spans"][0][0] for u, v in al.items() if v["spans"]]
        end = [dur[u] - v["spans"][-1][1] for u, v in al.items() if v["spans"] and u in dur]
        out[s] = dict(first_word_start_after_edge=stats(start), last_word_end_before_edge=stats(end))
    return out


def main():
    res = {}
    al = Aligner()
    res["a_fleurs_splices_known_switches"] = check_a(al)
    print("a", res["a_fleurs_splices_known_switches"], flush=True)
    del al
    torch.cuda.empty_cache()
    res["b_mms_vs_whisper_xattn_at_switches"] = check_b()
    print("b", res["b_mms_vs_whisper_xattn_at_switches"], flush=True)
    res["c_mms_vs_textgrid_edges"] = check_c()
    print("c", json.dumps(res["c_mms_vs_textgrid_edges"]), flush=True)
    lid = {}
    for p in ("ckpt/lid_ft_base/val_metrics.json", "ckpt/lid_full_base/val_metrics.json", "ckpt/lid_full_small/val_metrics.json"):
        if os.path.exists(manifest.path(p)):
            lid[p.split("/")[1]] = {k: round(1000 * v["boundary_median_s"]) for k, v in json.load(open(manifest.path(p))).items()}
    res["lid_boundary_median_ms_for_reference"] = lid
    json.dump(res, open(manifest.path("results/align_checks.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
