#!/usr/bin/env python3
"""Language-aware N-best re-ranking (§4.3(d)): features -> tune on val -> test predictions.

  CUDA_VISIBLE_DEVICES=4 python scripts/rerank.py --model ckpt/ft_base/best --tag ft_base \
      --lid ckpt/lid_ft_base/head.pt
Writes preds/<tag>__fp16__rerank_{lp,mms,ondevice}.jsonl (with critical-term flags) and
results/rerank_<tag>.json (chosen weights, val grid).
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
from csasr.lid.head import LIDHead  # noqa: E402
from csasr.score.metrics import analyze, corpus_metrics  # noqa: E402
from csasr.text.units import training_target  # noqa: E402
from csasr.train.data import TokenSpans  # noqa: E402

GAMMAS = [0.0, 0.25, 0.5, 1.0, 2.0]
ETAS = [0.0, 0.25, 0.5, 1.0, 2.0]


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
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--lid", required=True)
    ap.add_argument("--val_limit", type=int, default=1500)
    ap.add_argument("--loop_filter", action="store_true",
                    help="post-hoc variant: drop candidates with gzip compression ratio > 2.4 (Whisper's "
                         "repetition heuristic) before re-ranking; writes *_filt systems; needs cached features")
    a = ap.parse_args()
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


if __name__ == "__main__":
    main()
