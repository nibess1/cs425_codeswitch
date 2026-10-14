#!/usr/bin/env python3
"""Train the language-ID head on a frozen Whisper encoder.

  CUDA_VISIBLE_DEVICES=4 python scripts/train_lid.py --encoder ckpt/ft_base/best --run lid_ft_base
Data: ASCEND + NSC train frames labelled by forced alignment, plus FLEURS splices.
Eval: frame accuracy and boundary error on ASCEND / NSC val (after smoothing).
"""
import argparse
import json
import os
import pickle
import random
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from csasr.data import manifest  # noqa: E402
from csasr.decode.whisper_hf import load_model, load_processor  # noqa: E402
from csasr.lid.head import LIDHead, boundary_error, pad_labels, smooth  # noqa: E402

TRAIN = [("ascend_train", "data/ascend/manifests/train.jsonl"), ("nsc_train", "data/nsc/manifests/train.jsonl"),
         ("fleurs_splice_train", "data/fleurs/manifests/splice_train.jsonl")]
VAL = [("ascend_val", "data/ascend/manifests/val.jsonl"), ("nsc_val", "data/nsc/manifests/val.jsonl")]


class FrameSet(Dataset):
    def __init__(self, pairs, fe, limit=None):
        self.items = []
        for name, man in pairs:
            al = pickle.load(open(manifest.path(f"data/align/{name}.pkl"), "rb"))
            rows = [r for r in manifest.load(man) if r["utt_id"] in al]
            if limit:
                rows = rows[:limit]
            self.items += [(r, al[r["utt_id"]]["frames"]) for r in rows]
        self.fe = fe

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        r, fr = self.items[i]
        wav = manifest.load_audio(r["audio"])
        feats = self.fe(wav, sampling_rate=16000, return_tensors="np").input_features[0]
        return feats, pad_labels(fr), min(len(fr), 1500)


def collate(b):
    return (torch.from_numpy(np.stack([x[0] for x in b])), torch.from_numpy(np.stack([x[1] for x in b])),
            torch.tensor([x[2] for x in b]))


@torch.no_grad()
def encode(enc, feats):
    return enc(feats.cuda().half()).last_hidden_state.float()


def evaluate(enc, head, dl):
    head.eval()
    acc_n = acc_d = 0
    raw_n = 0
    lang_n = lang_d = 0
    bes = []
    per_class = np.zeros((3, 3))
    for feats, lab, n in dl:
        logits = head(encode(enc, feats))
        pred = logits.argmax(-1).cpu().numpy()
        lab = lab.numpy()
        for k in range(len(pred)):
            L, P = lab[k, : n[k]], pred[k, : n[k]]
            Ps = smooth(P.copy())
            v = L >= 0
            acc_n += (Ps[v] == L[v]).sum(); raw_n += (P[v] == L[v]).sum(); acc_d += v.sum()
            sp = (L == 0) | (L == 1)  # language accuracy on speech frames (what re-ranking uses)
            lang_n += (logits[k, : n[k]][:, :2].argmax(-1).cpu().numpy()[sp] == L[sp]).sum(); lang_d += sp.sum()
            for t, p in zip(L[v], Ps[v]):
                per_class[t, p] += 1
            bes += boundary_error(L, Ps)
    head.train()
    bes = np.array(bes)
    return dict(frame_acc=float(acc_n / acc_d), frame_acc_raw=float(raw_n / acc_d),
                speech_lang_acc=float(lang_n / max(lang_d, 1)),
                boundary_median_s=float(np.nanmedian(bes)) if len(bes) else None,
                boundary_within_100ms=float(np.mean(bes <= 0.1)) if len(bes) else None,
                switch_detect_rate=float(np.mean(~np.isnan(bes))) if len(bes) else None,
                confusion=per_class.astype(int).tolist())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--eval_only", action="store_true", help="re-evaluate the saved head")
    a = ap.parse_args()
    random.seed(0); torch.manual_seed(0)
    tok, fe = load_processor("small")
    model = load_model(manifest.path(a.encoder) if os.path.isdir(manifest.path(a.encoder)) else a.encoder, "fp16")
    enc = model.get_encoder().eval()
    head = LIDHead(model.config.d_model).cuda()
    out = manifest.path(f"ckpt/{a.run}")
    va = {n: DataLoader(FrameSet([(n, m)], fe, limit=800), batch_size=64, num_workers=8, collate_fn=collate)
          for n, m in VAL}
    if a.eval_only:
        head.load_state_dict(torch.load(os.path.join(out, "head.pt"))["state"])
        res = {n: evaluate(enc, head, d) for n, d in va.items()}
        json.dump(res, open(os.path.join(out, "val_metrics.json"), "w"), indent=1)
        print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "confusion"} for k, v in res.items()}))
        return
    tr = FrameSet(TRAIN, fe)
    dl = DataLoader(tr, batch_size=a.batch, shuffle=True, num_workers=12, collate_fn=collate, drop_last=True,
                    persistent_workers=True)
    opt = torch.optim.AdamW(head.parameters(), lr=a.lr, weight_decay=0.01)
    total = int(len(dl) * a.epochs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=total, pct_start=0.05)
    os.makedirs(out, exist_ok=True)
    log = open(manifest.path(f"runs/{a.run}.log.jsonl"), "w")
    step = 0
    while step < total:
        for feats, lab, _ in dl:
            h = encode(enc, feats)
            logits = head(h)
            loss = F.cross_entropy(logits.transpose(1, 2), lab.cuda(), ignore_index=-1)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step(); step += 1
            if step % 100 == 0:
                print(json.dumps(dict(step=step, loss=loss.item())), file=log, flush=True)
            if step >= total:
                break
    res = {n: evaluate(enc, head, d) for n, d in va.items()}
    torch.save(dict(state=head.state_dict(), d_model=model.config.d_model, encoder=a.encoder), os.path.join(out, "head.pt"))
    json.dump(res, open(os.path.join(out, "val_metrics.json"), "w"), indent=1)
    print(json.dumps(res, indent=1), file=log, flush=True)
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "confusion"} for k, v in res.items()}))


if __name__ == "__main__":
    main()
