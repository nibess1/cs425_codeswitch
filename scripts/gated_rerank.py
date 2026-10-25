#!/usr/bin/env python3
"""C1: LID-gated re-ranking. Re-rank a clip only when the LID head hears a minority-language run
of at least theta ms; otherwise keep the system's own greedy output. No retraining: reuses the
cached N-best features (preds/<feats_tag>__rerank_feats.pkl) and the weights chosen on val.

Rule (pre-registered, PROCESS_LOG 20 Oct): theta in {0, 200, 400, 800, inf}; choose the lowest
pooled val MER, ties -> larger theta (fewer clips re-ranked).

  CUDA_VISIBLE_DEVICES=7 python scripts/gated_rerank.py --tag full_base --greedy full_base__fp16__full \
      --lid ckpt/lid_ft_base/head.pt --out_tag full_base
"""
import argparse
import json
import os
import pickle
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from csasr.data import manifest  # noqa: E402
from csasr.decode import rerank as RR  # noqa: E402
from csasr.decode.whisper_hf import load_model, load_processor  # noqa: E402
from csasr.lid.head import FRAME, LIDHead, smooth  # noqa: E402
from csasr.score import report as R  # noqa: E402
from csasr.score.metrics import analyze, corpus_metrics  # noqa: E402

THETAS = [0, 200, 400, 800, float("inf")]


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, help="feature-cache tag (preds/<tag>__rerank_feats.pkl)")
    ap.add_argument("--greedy", required=True, help="system id of the greedy run (test sets)")
    ap.add_argument("--greedy_val", default=None, help="system id of its val run (default <greedy>_val)")
    ap.add_argument("--weights", default=None, help="results/rerank_<tag>_filt.json (default)")
    ap.add_argument("--lid", required=True)
    ap.add_argument("--out_tag", required=True, help="model tag of the output systems")
    ap.add_argument("--suffix", default="gatedv2")
    ap.add_argument("--val_limit", type=int, default=1500)
    a = ap.parse_args()
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
