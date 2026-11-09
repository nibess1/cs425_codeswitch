#!/usr/bin/env python3
"""Language-aware N-best re-ranking (§4.3(d)): features -> tune on val -> test predictions.

  CUDA_VISIBLE_DEVICES=4 python scripts/rerank.py --model ckpt/ft_base/best --tag ft_base \
      --lid ckpt/lid_ft_base/head.pt
Writes preds/<tag>__fp16__rerank_{lp,mms,ondevice}.jsonl (with critical-term flags) and
results/rerank_<tag>.json (chosen weights, val grid).

--gate (C1, LID-gated re-ranking): re-rank a clip only when the LID head hears a minority-language
run of at least theta ms; otherwise keep the system's own greedy output. No retraining: reuses the
cached N-best features (preds/<tag>__rerank_feats.pkl) and the weights chosen on val.
Rule (pre-registered, PROCESS_LOG 20 Oct): theta in {0, 200, 400, 800, inf}; choose the lowest
pooled val MER, ties -> larger theta (fewer clips re-ranked).

  CUDA_VISIBLE_DEVICES=7 python scripts/rerank.py --gate --tag full_base --greedy full_base__fp16__full \
      --lid ckpt/lid_ft_base/head.pt --out_tag full_base
"""
import argparse
import itertools
import json
import os
import pickle
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from csasr.data import manifest  # noqa: E402
from csasr.decode import rerank as RR  # noqa: E402
from csasr.decode.whisper_hf import Decoder, load_model, load_processor  # noqa: E402
from csasr.lid.align import Aligner  # noqa: E402
from csasr.lid.head import FRAME, LIDHead, smooth  # noqa: E402
from csasr.score import report as R  # noqa: E402
from csasr.score.metrics import analyze, corpus_metrics  # noqa: E402
from csasr.text.units import training_target  # noqa: E402
from csasr.train.data import TokenSpans  # noqa: E402

GAMMAS = [0.0, 0.25, 0.5, 1.0, 2.0]
ETAS = [0.0, 0.25, 0.5, 1.0, 2.0]
THETAS = [0, 200, 400, 800, float("inf")]


def feats_for_set(rows, dec, model, lid_enc, head, ctc, spanner, heads, bs=8):
    out = {}
    tok = dec.tok
    for b0 in range(0, len(rows), bs):
        chunk = rows[b0: b0 + bs]
        wavs = [manifest.load_audio(r["audio"]) for r in chunk]
        x = torch.from_numpy(np.stack([dec.fe(w, sampling_rate=16000, return_tensors="np").input_features[0]
                                       for w in wavs]))
        cands = [dict() for _ in chunk]
        for mode in ("zh", "en", "free"):
            g = dec.generate(x, mode, num_beams=4, num_return_sequences=4)
            for k, t in enumerate(g["texts"]):
                ci = k // 4
                tgt = training_target(t)
                if tgt:
                    cands[ci].setdefault(tgt, g["langs"][ci])
        with torch.no_grad():
            lid = head(lid_enc(x.cuda().half()).last_hidden_state.float()).log_softmax(-1).cpu().numpy()
            enc = model.get_encoder()(x.cuda().half()).last_hidden_state
        for ci, r in enumerate(chunk):
            n_frames = min(1500, int(np.ceil(len(wavs[ci]) / 16000 / RR.FRAME)))
            em = ctc.emission(wavs[ci])
            items = list(cands[ci].items())
            if not items:
                out[r["utt_id"]] = []
                continue
            seqs, texts = [], []
            for tgt, lang in items:
                text = " " + tgt if tgt[0].isascii() else tgt
                ids = tok(text, add_special_tokens=False).input_ids[:128]
                seqs.append(([dec.sot, dec.lang[lang], dec.trans, dec.notime] + ids + [dec.eot], ids))
                texts.append(text)
            T = max(len(s) for s, _ in seqs)
            inp = torch.full((len(seqs), T - 1), dec.eot, device="cuda")
            tgt_t = torch.full((len(seqs), T - 1), -100, device="cuda")
            for i, (s, _) in enumerate(seqs):
                inp[i, : len(s) - 1] = torch.tensor(s[:-1]); tgt_t[i, : len(s) - 1] = torch.tensor(s[1:])
            tgt_t[:, 1:3] = -100  # task / no-timestamps are forced
            with torch.no_grad():
                o = model(encoder_outputs=(enc[ci: ci + 1].expand(len(seqs), -1, -1),), decoder_input_ids=inp,
                          output_attentions=True)
                lp_tok = -torch.nn.functional.cross_entropy(o.logits.float().transpose(1, 2), tgt_t,
                                                             ignore_index=-100, reduction="none")
                n_tok = (tgt_t != -100).sum(1)
                lp = (lp_tok.sum(1) / n_tok).tolist()
                att = torch.stack([o.cross_attentions[l][:, h] for l, h in heads], 1).float().cpu().numpy()
            res = []
            for i, ((s, ids), text) in enumerate(zip(seqs, texts)):
                # text-token rows: positions 3 .. 3+len(ids) predict ids (decoder input index 4.. hold ids)
                a = att[i][:, 3: 3 + len(ids)]
                tok_frames = RR.token_frames_from_attention(a, n_frames) if len(ids) else []
                units, wts = RR.unit_frames_from_tokens(text, ids, tok_frames, spanner)
                c, _, mms = ctc.score(em, text, len(wavs[ci]))
                res.append(dict(text=text.strip(), lang=items[i][1], lp=lp[i], ctc=c,
                                a_wts=RR.agreement(units, wts, lid[ci]),
                                a_mms=RR.agreement(units, mms, lid[ci]) if mms else RR.agreement(units, wts, lid[ci])))
            out[r["utt_id"]] = res
    return out


def mer_of(rows, feats, gamma, eta):
    pairs = [(r["ref_raw"], RR.choose(feats[r["utt_id"]], gamma, eta)["text"] if feats[r["utt_id"]] else "")
             for r in rows]
    return corpus_metrics(analyze(pairs))["MER"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", help="checkpoint (feature extraction; not needed with --loop_filter or --gate)")
    ap.add_argument("--tag", required=True, help="feature-cache tag (preds/<tag>__rerank_feats.pkl)")
    ap.add_argument("--lid", required=True)
    ap.add_argument("--val_limit", type=int, default=1500)
    ap.add_argument("--loop_filter", action="store_true",
                    help="post-hoc variant: drop candidates with gzip compression ratio > 2.4 (Whisper's "
                         "repetition heuristic) before re-ranking; writes *_filt systems; needs cached features")
    ap.add_argument("--gate", action="store_true", help="LID-gated re-ranking (see top); needs cached features")
    ap.add_argument("--greedy", help="--gate: system id of the greedy run (test sets)")
    ap.add_argument("--greedy_val", default=None, help="--gate: system id of its val run (default <greedy>_val)")
    ap.add_argument("--weights", default=None, help="--gate: results/rerank_<tag>_filt.json (default)")
    ap.add_argument("--out_tag", help="--gate: model tag of the output systems")
    ap.add_argument("--suffix", default="gatedv2", help="--gate: method suffix of the output systems")
    a = ap.parse_args()
    if a.gate:
        return gate(a)
    cache = manifest.path(f"preds/{a.tag}__rerank_feats.pkl")
    sets = {s: manifest.load(s)[: a.val_limit] for s in manifest.EVAL_VAL}
    sets.update({s: manifest.load(s) for s in manifest.EVAL_TEST})
    if a.loop_filter:
        allf = pickle.load(open(cache, "rb"))
        allf = {s: {u: ([c for c in cs if RR.compression_ratio(c["text"]) <= 2.4] or cs) for u, cs in d.items()}
                for s, d in allf.items()}
        return tune_and_write(a, sets, allf, suffix="_filt")
    tok, fe = load_processor("small")
    model = load_model(manifest.path(a.model), "fp16")
    dec = Decoder(model, tok, fe)
    heads = [tuple(h) for h in model.generation_config.alignment_heads]
    ck = torch.load(manifest.path(a.lid), map_location="cpu")
    lid_model = load_model(manifest.path(ck["encoder"]), "fp16")
    lid_enc = lid_model.get_encoder().eval()
    head = LIDHead(ck["d_model"]).cuda().eval()
    head.load_state_dict(ck["state"])
    ctc = RR.CTCScorer(Aligner())
    spanner = TokenSpans(tok)
    allf = pickle.load(open(cache, "rb")) if os.path.exists(cache) else {}
    for s, rows in sets.items():
        if s in allf:
            continue
        t0 = time.time()
        allf[s] = feats_for_set(rows, dec, model, lid_enc, head, ctc, spanner, heads)
        pickle.dump(allf, open(cache, "wb"))
        print(f"{s}: features for {len(rows)} utts in {time.time() - t0:.0f}s", flush=True)

    tune_and_write(a, sets, allf)


def tune_and_write(a, sets, allf, suffix=""):
    # tune on pooled validation
    vrows = [(r, s) for s in manifest.EVAL_VAL for r in sets[s]]
    vfeat = {r["utt_id"]: allf[s][r["utt_id"]] for r, s in vrows}
    vr = [r for r, _ in vrows]
    grid = {f"{g},{e}": mer_of(vr, vfeat, g, e) for g, e in itertools.product(GAMMAS, ETAS)}
    best_mms = min(((g, e) for g, e in itertools.product(GAMMAS, ETAS) if g > 0),
                   key=lambda ge: grid[f"{ge[0]},{ge[1]}"])
    best_dev = min(((0.0, e) for e in ETAS), key=lambda ge: grid[f"{ge[0]},{ge[1]}"])
    variants = {f"rerank_lp{suffix}": (0.0, 0.0), f"rerank_mms{suffix}": best_mms, f"rerank_ondevice{suffix}": best_dev}
    json.dump(dict(grid_val_MER=grid, chosen={k: v for k, v in variants.items()}),
              open(manifest.path(f"results/rerank_{a.tag}{suffix}.json"), "w"), indent=1)
    for name, (g, e) in variants.items():
        with open(manifest.path(f"preds/{a.tag}__fp16__{name}.jsonl"), "w", encoding="utf-8") as f:
            for s in manifest.EVAL_TEST + manifest.EVAL_VAL:
                for r in sets[s]:
                    c = allf[s][r["utt_id"]]
                    rk = RR.ranked(c, g, e) if c else []
                    hyp = rk[0]["text"] if rk else ""
                    flags = RR.flags_for([x["text"] for x in rk[:3]]) if rk else []
                    f.write(json.dumps(dict(utt_id=r["utt_id"], corpus=r["corpus"], split=r["split"], set=s,
                                            model=a.tag, precision="fp16", method=name, ref=r["ref_raw"], hyp=hyp,
                                            lang_token=rk[0]["lang"] if rk else "", flags=flags, gamma=g, eta=e),
                                       ensure_ascii=False) + "\n")
        json.dump(dict(model=a.tag, method=name, gamma=g, eta=e), open(
            manifest.path(f"preds/{a.tag}__fp16__{name}.meta.json"), "w"))
    print(json.dumps(variants))


def minority_run_ms(pred: np.ndarray) -> float:
    """Longest run (over speech frames, pauses ignored) of the less frequent language, in ms;
    -1 if only one language is heard."""
    sp = pred[(pred == 0) | (pred == 1)]
    if len(sp) == 0 or (sp == 0).all() or (sp == 1).all():
        return -1.0
    minority = 0 if (sp == 0).sum() < (sp == 1).sum() else 1
    best = cur = 0
    for x in sp:
        cur = cur + 1 if x == minority else 0
        best = max(best, cur)
    return best * FRAME * 1000


@torch.no_grad()
def lid_runs(rows, head_ckpt, set_name, bs=32):
    ck = torch.load(manifest.path(head_ckpt), map_location="cpu", weights_only=False)
    model = load_model(manifest.path(ck["encoder"]), "fp16")
    enc = model.get_encoder().eval()
    head = LIDHead(ck["d_model"]).cuda().eval()
    head.load_state_dict(ck["state"])
    _, fe = load_processor("small")
    out = {}
    for b in range(0, len(rows), bs):
        chunk = rows[b: b + bs]
        wavs = [manifest.load_audio(r["audio"]) for r in chunk]
        x = torch.from_numpy(np.stack([fe(w, sampling_rate=16000, return_tensors="np").input_features[0] for w in wavs]))
        pred = head(enc(x.cuda().half()).last_hidden_state.float()).argmax(-1).cpu().numpy()
        for r, w, p in zip(chunk, wavs, pred):
            n = min(1500, int(np.ceil(len(w) / 16000 / FRAME)))
            out[(set_name, r["utt_id"])] = minority_run_ms(smooth(p[:n].copy()))
    return out


def gate(a):
    feats = pickle.load(open(manifest.path(f"preds/{a.tag}__rerank_feats.pkl"), "rb"))
    feats = {s: {u: ([c for c in cs if RR.compression_ratio(c["text"]) <= 2.4] or cs) for u, cs in d.items()}
             for s, d in feats.items()}  # same loop filter as the *_filt variants
    weights = json.load(open(manifest.path(a.weights or f"results/rerank_{a.tag}_filt.json")))["chosen"]
    greedy = {}
    for name in (a.greedy, a.greedy_val or a.greedy + "_val"):
        for r in R.read_preds(name):
            greedy[(r["set"], r["utt_id"])] = r["hyp"]
    sets = {s: manifest.load(s)[: a.val_limit] for s in manifest.EVAL_VAL}
    sets.update({s: manifest.load(s) for s in manifest.EVAL_TEST})
    cache = manifest.path(f"preds/{a.out_tag}__lidruns_v2_{os.path.basename(os.path.dirname(a.lid))}.pkl")
    runs = pickle.load(open(cache, "rb")) if os.path.exists(cache) else {}
    for s, rows in sets.items():
        todo = [r for r in rows if (s, r["utt_id"]) not in runs]
        if todo:
            runs.update(lid_runs(todo, a.lid, s))
            pickle.dump(runs, open(cache, "wb"))
    report = {"lid_head": a.lid, "thetas": {}, "chosen": {}, "share_reranked": {}}
    for variant in ("rerank_ondevice_filt", "rerank_mms_filt"):
        g, e = weights[variant]

        def hyp(r, theta):
            c = feats[r["set_name"]].get(r["utt_id"]) or []
            k = (r["set_name"], r["utt_id"])  # ASCEND val and test reuse utt_ids, so key by set
            if c and runs[k] >= theta and runs[k] >= 0:
                return RR.choose(c, g, e)["text"], True
            return greedy[k], False

        vrows = [dict(r, set_name=s) for s in manifest.EVAL_VAL for r in sets[s]]
        val = {}
        for th in THETAS:
            hs = [hyp(r, th)[0] for r in vrows]
            val[str(th)] = corpus_metrics(analyze([(r["ref_raw"], h) for r, h in zip(vrows, hs)]))["MER"]
        best = min(THETAS, key=lambda t: (round(val[str(t)], 6), -t))
        report["thetas"][variant] = val
        report["chosen"][variant] = best
        name = f"{a.out_tag}__fp16__{variant}_{a.suffix}"
        share = {}
        with open(manifest.path(f"preds/{name}.jsonl"), "w", encoding="utf-8") as f:
            for s in manifest.EVAL_TEST + manifest.EVAL_VAL:
                n_rr = 0
                for r in sets[s]:
                    h, rr = hyp(dict(r, set_name=s), best)
                    n_rr += rr
                    f.write(json.dumps(dict(utt_id=r["utt_id"], corpus=r["corpus"], split=r["split"], set=s,
                                            model=a.out_tag, precision="fp16", method=f"{variant}_{a.suffix}",
                                            ref=r["ref_raw"], hyp=h, reranked=rr), ensure_ascii=False) + "\n")
                share[s] = n_rr / len(sets[s])
        report["share_reranked"][variant] = share
        json.dump(dict(model=a.out_tag, method=f"{variant}_{a.suffix}", theta_ms=best, lid=a.lid, share_reranked=share),
                  open(manifest.path(f"preds/{name}.meta.json"), "w"), indent=1)
    json.dump(report, open(manifest.path(f"results/gated_{a.out_tag}_{a.suffix}.json"), "w"), indent=1)
    print(json.dumps({k: report[k] for k in ("chosen", "share_reranked")}))


if __name__ == "__main__":
    main()
