"""Full fine-tuning of Whisper with the role-weighted loss (A1) and optional extra loss terms.

L_role = - sum_t w_t log p(y_t | y_<t, x) / sum_t w_t,  w_t = 1 + alpha e_t + beta m_t
alpha = beta = 0 is plain fine-tuning. Checkpoints are selected on validation MER.
"""
import json
import math
import os
import random
import shutil
import time
from dataclasses import asdict, dataclass, field
from functools import partial

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..data import manifest
from ..decode.whisper_hf import Decoder, hub_name, load_processor
from ..score.metrics import analyze, corpus_metrics
from .data import ASRDataset, collate


@dataclass
class TrainConfig:
    run: str
    model: str = "base"                 # tiny | base | small | checkpoint dir
    train_sets: list = field(default_factory=lambda: ["ascend_train", "nsc_train"])
    val_sets: list = field(default_factory=lambda: ["ascend_val", "nsc_val"])
    val_limit: int = 600                # utterances per val set used for checkpoint selection
    lr: float = 2.5e-5
    weight_decay: float = 0.01
    warmup: int = 500
    epochs: float = 3.0
    batch: int = 32
    grad_accum: int = 1
    alpha: float = 0.0
    beta: float = 0.0
    eval_every: int = 1000
    save_every: int = 500
    seed: int = 425
    num_workers: int = 12
    max_grad_norm: float = 1.0
    # semantic minimum-risk term (A2); lambda_sem = 0 disables it
    lambda_sem: float = 0.0
    rho0: float = 0.5
    mrt_every: int = 4                  # apply the MRT term on every k-th step (cost control)
    mrt_batch: int = 8
    limit_train: int | None = None
    lora: bool = False                  # QLoRA QAT (csasr/quant/qat.py builds the model)


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def role_weighted_ce(logits, labels, roles, alpha, beta):
    ce = F.cross_entropy(logits.float().transpose(1, 2), labels, ignore_index=-100, reduction="none")
    valid = (labels != -100).float()
    w = 1.0 + alpha * (roles == 2).float() + beta * (roles == 1).float()
    w = w * valid
    return (w * ce).sum() / w.sum().clamp_min(1.0)


class Trainer:
    def __init__(self, cfg: TrainConfig, model=None, extra_loss=None):
        self.cfg = cfg
        set_seed(cfg.seed)
        self.out = manifest.path(f"ckpt/{cfg.run}")
        self.log_dir = manifest.path(f"runs/{cfg.run}")
        os.makedirs(self.out, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        json.dump(asdict(cfg), open(os.path.join(self.log_dir, "config.json"), "w"), indent=1)
        self.tok, self.fe = load_processor("small")  # all multilingual sizes share tokenizer + features
        if model is None:
            from transformers import WhisperForConditionalGeneration
            src = manifest.path(cfg.model) if os.path.isdir(manifest.path(cfg.model)) else hub_name(cfg.model)
            model = WhisperForConditionalGeneration.from_pretrained(src, torch_dtype=torch.float32)
            model.config.forced_decoder_ids = None
            model.generation_config.forced_decoder_ids = None
        quantized = getattr(model, "is_loaded_in_4bit", False) or hasattr(model, "peft_config")
        self.model = model if quantized else model.cuda()
        self.extra_loss = extra_loss
        rows = [r for s in cfg.train_sets for r in manifest.load(s)]
        random.Random(cfg.seed).shuffle(rows)
        if cfg.limit_train:
            rows = rows[:cfg.limit_train]
        self.train_rows = rows
        self.ds = ASRDataset(rows, self.tok, self.fe)
        self.val = {s: manifest.load(s)[: cfg.val_limit] for s in cfg.val_sets}
        steps_per_epoch = math.ceil(len(rows) / (cfg.batch * cfg.grad_accum))
        self.total = int(steps_per_epoch * cfg.epochs)
        params = [p for p in self.model.parameters() if p.requires_grad]
        self.opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay, betas=(0.9, 0.98), eps=1e-6)
        w, T = cfg.warmup, self.total
        self.sched = torch.optim.lr_scheduler.LambdaLR(  # linear warm-up, then linear decay to 0
            self.opt, lambda st: (st + 1) / w if st < w else max(0.0, (T - st) / max(1, T - w)))
        self.step = 0
        self.best = float("inf")
        self.logf = open(os.path.join(self.log_dir, "log.jsonl"), "a")

    def log(self, **kw):
        kw.update(step=self.step, time=time.strftime("%H:%M:%S"))
        self.logf.write(json.dumps(kw) + "\n"); self.logf.flush()
        print(json.dumps(kw), flush=True)

    @torch.no_grad()
    def evaluate(self):
        self.model.eval()
        dec = Decoder(self.model, self.tok, self.fe)
        res = {}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            for s, rows in self.val.items():
                hyps = dec.transcribe(rows, "free", batch_size=48, num_workers=8)
                m = corpus_metrics(analyze([(r["ref_raw"], h) for r, (h, _) in zip(rows, hyps)]))
                res[s] = dict(MER=m["MER"], E_E=m["E_E"], lang_acc=float(np.mean(
                    [l == (r["main"] if r["main"] in ("zh", "en") else l) for r, (_, l) in zip(rows, hyps)])))
        self.model.train()
        res["mean_MER"] = float(np.mean([v["MER"] for v in res.values()]))
        return res

    def save(self, tag):
        d = os.path.join(self.out, tag)
        tmp = d + ".tmp"
        if os.path.exists(tmp):
            shutil.rmtree(tmp)
        self.model.save_pretrained(tmp, safe_serialization=True)
        self.tok.save_pretrained(tmp)
        self.fe.save_pretrained(tmp)
        json.dump(dict(step=self.step, best=self.best), open(os.path.join(tmp, "trainer_state.json"), "w"))
        if os.path.exists(d):
            shutil.rmtree(d)
        os.replace(tmp, d)

    def train(self):
        cfg = self.cfg
        dl = DataLoader(self.ds, batch_size=cfg.batch, shuffle=True, num_workers=cfg.num_workers,
                        collate_fn=partial(collate, pad_id=self.tok.eos_token_id), drop_last=True,
                        persistent_workers=True, prefetch_factor=4)
        self.model.train()
        r0 = self.evaluate() if cfg.eval_every else None
        if r0:
            self.log(event="eval", **r0)
        t0, acc = time.time(), []
        while self.step < self.total:
            for batch in dl:
                feats = batch["input_features"].cuda(non_blocking=True)
                dec = batch["decoder_input_ids"].cuda(non_blocking=True)
                labels = batch["labels"].cuda(non_blocking=True)
                roles = batch["roles"].cuda(non_blocking=True)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    out = self.model(input_features=feats, decoder_input_ids=dec)
                    loss = role_weighted_ce(out.logits, labels, roles, cfg.alpha, cfg.beta)
                    parts = {"l_role": loss.item()}
                    if self.extra_loss is not None:
                        extra = self.extra_loss(self, batch)
                        if extra is not None:
                            loss = loss + extra["loss"]
                            parts.update({k: v for k, v in extra.items() if k != "loss"})
                (loss / cfg.grad_accum).backward()
                acc.append(parts)
                if (len(acc)) % cfg.grad_accum:
                    continue
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.max_grad_norm)
                self.opt.step(); self.sched.step(); self.opt.zero_grad(set_to_none=True)
                self.step += 1
                if self.step % 50 == 0:
                    keys = {k for a in acc for k in a}
                    self.log(event="train", lr=self.sched.get_last_lr()[0],
                             **{k: float(np.mean([a[k] for a in acc if k in a])) for k in keys},
                             it_s=round(50 / (time.time() - t0), 2))
                    acc, t0 = [], time.time()
                if cfg.save_every and self.step % cfg.save_every == 0:
                    self.save("last")
                if cfg.eval_every and (self.step % cfg.eval_every == 0 or self.step == self.total):
                    r = self.evaluate()
                    self.log(event="eval", **r)
                    if r["mean_MER"] < self.best:
                        self.best = r["mean_MER"]
                        self.save("best")
                    t0 = time.time()
                if self.step >= self.total:
                    break
        self.save("last")
        if not cfg.eval_every:
            self.save("best")
        self.log(event="done", best=self.best)
        open(os.path.join(self.log_dir, "done.flag"), "w").write(str(self.best))
