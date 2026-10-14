#!/usr/bin/env python3
"""Fine-tune Whisper from a YAML config (configs/*.yaml); CLI flags override the YAML.

  CUDA_VISIBLE_DEVICES=4 python scripts/train.py configs/ft_base.yaml [--alpha 1 --run ft_base_a1]
"""
import argparse
import dataclasses
import os
import sys

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from csasr.train.trainer import TrainConfig, Trainer  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    fields = {f.name: f for f in dataclasses.fields(TrainConfig)}
    for name, f in fields.items():
        ap.add_argument(f"--{name}", default=None)
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config)) or {}
    for name in fields:
        v = getattr(a, name)
        if v is not None:
            cfg[name] = yaml.safe_load(v)
    cfg = TrainConfig(**cfg)
    extra = None
    if cfg.lambda_sem > 0:
        from csasr.train.mrt import SemanticMRT
        extra = SemanticMRT(cfg)
    model = None
    if cfg.lora:
        from csasr.quant.qat import build_qlora_model
        model = build_qlora_model(cfg.model)
    Trainer(cfg, model=model, extra_loss=extra).train()


if __name__ == "__main__":
    main()
