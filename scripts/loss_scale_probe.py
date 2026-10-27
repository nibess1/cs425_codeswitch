#!/usr/bin/env python3
"""Fourth-run diagnosis (27 Oct): how big is each loss term's gradient?

Loss VALUES mislead for MRT (the cost is mean-subtracted, so L_sem averages ~0 whatever its
gradient), so this measures gradient L2 norms on the same training batches, from one checkpoint:
  role   : role-weighted CE (alpha = beta = 0.5), full batch of 32, as in training
  sem    : MRT with the semantic cost only (lambda = 1)
  tl     : MRT with cost = N_tl only (lambda = gamma = 1); normalised over only the clips where N_tl varies,
           so it overstates the term; use tl_in_combined = |grad(sem+tl) - grad(sem)| instead
  sem+tl : MRT with cost = smer + N_tl (gamma = 1, lambda = 1)
No weights are updated. Writes results/loss_scale_probe.json.

  CUDA_VISIBLE_DEVICES=5 python scripts/loss_scale_probe.py --model ckpt/sem_base_l0.1_r0.5/best --batches 40
"""
import argparse
import json
import os
import sys
from functools import partial

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import csasr.train.mrt as M  # noqa: E402
from csasr.data import manifest  # noqa: E402
from csasr.train.data import collate  # noqa: E402
from csasr.train.trainer import TrainConfig, Trainer, role_weighted_ce  # noqa: E402

COSTS = {
    "sem": lambda s, g: s["smer"],
    "tl": lambda s, g: s["N_tl_E"],
    "sem+tl": lambda s, g: s["smer"] + s["N_tl_E"],
}


def flat_grad(model):
    return torch.cat([p.grad.detach().float().flatten() if p.grad is not None else torch.zeros_like(p).flatten()
                      for p in model.parameters()])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--batches", type=int, default=40)
    ap.add_argument("--mrt_batch", type=int, default=8)
    ap.add_argument("--out", default="results/loss_scale_probe.json")
    a = ap.parse_args()
    cfg = TrainConfig(run="probe_lossscale", model=a.model, alpha=0.5, beta=0.5, lambda_sem=1.0, rho0=0.5,
                      mrt_batch=a.mrt_batch, translit_gamma=1.0, eval_every=0, num_workers=8)
    tr = Trainer(cfg, extra_loss=None)
    mrt = M.SemanticMRT(cfg)
    tr.step = 0  # MRT fires when step % mrt_every == 0
    model = tr.model
    model.train()
    g = torch.Generator().manual_seed(0)
    dl = DataLoader(tr.ds, batch_size=cfg.batch, shuffle=True, generator=g, num_workers=8,
                    collate_fn=partial(collate, pad_id=tr.tok.eos_token_id), drop_last=True)
    rec = {k: [] for k in ["role", *COSTS, "tl_in_combined"]}
    cos = {k: [] for k in COSTS}
    fired = {k: [] for k in COSTS}
    orig = M.mrt_cost
    for bi, batch in enumerate(dl):
        if bi >= a.batches:
            break
        model.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(input_features=batch["input_features"].cuda(), decoder_input_ids=batch["decoder_input_ids"].cuda())
            loss = role_weighted_ce(out.logits, batch["labels"].cuda(), batch["roles"].cuda(), cfg.alpha, cfg.beta)
        loss.backward()
        gr = flat_grad(model)
        rec["role"].append(float(gr.norm()))
        keep = {}
        for name, fn in COSTS.items():
            M.mrt_cost = fn
            model.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                ex = mrt(tr, batch)
            if ex is None or not torch.is_tensor(ex["loss"]) or not ex["loss"].requires_grad:
                rec[name].append(0.0); cos[name].append(np.nan); fired[name].append(0)
                continue
            ex["loss"].backward()
            gm = flat_grad(model)
            n = float(gm.norm())
            if name in ("sem", "sem+tl"):
                keep[name] = gm
            rec[name].append(n); fired[name].append(int(n > 0))
            cos[name].append(float(torch.nn.functional.cosine_similarity(gm, gr, dim=0)) if n > 0 else np.nan)
        M.mrt_cost = orig
        # the tl part as it acts inside the combined loss (same 1/len(sel) normalisation; MRT grad is linear in cost)
        rec["tl_in_combined"].append(float((keep["sem+tl"] - keep["sem"]).norm()) if len(keep) == 2 else 0.0)
        print(bi, {k: round(v[-1], 4) for k, v in rec.items()}, flush=True)
    res = {"model": a.model, "batches": a.batches, "mrt_batch": a.mrt_batch,
           "note": "grad L2 norms; MRT terms at lambda = 1 on an MRT step; in training MRT fires every 4th step",
           "mean_norm": {k: float(np.mean(v)) for k, v in rec.items()},
           "median_norm": {k: float(np.median(v)) for k, v in rec.items()},
           "share_batches_nonzero": {k: float(np.mean(v)) for k, v in fired.items()},
           "mean_norm_when_nonzero": {k: float(np.mean([x for x in rec[k] if x > 0])) if any(rec[k]) else 0.0 for k in COSTS},
           "cos_with_role_grad": {k: float(np.nanmean(v)) if not all(np.isnan(v)) else None for k, v in cos.items()}}
    r = res["mean_norm"]
    res["ratio_to_role_at_lambda0.1_per_mrt_step"] = {k: 0.1 * r[k] / r["role"] for k in r if k != "role"}
    res["ratio_to_role_at_lambda0.1_averaged_over_steps"] = {k: 0.1 * r[k] / r["role"] / 4 for k in r if k != "role"}
    json.dump(res, open(manifest.path(a.out), "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
