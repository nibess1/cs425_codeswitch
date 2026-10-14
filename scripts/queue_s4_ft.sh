#!/usr/bin/env bash
# S4: plain full fine-tuning of base, small, tiny (base first: methods are tuned on it).
set -u
cd "$(dirname "$0")/.."
PY=${PY:-~/micromamba/envs/cs425/bin/python}
for m in base small tiny; do
  [ -f ckpt/ft_$m/best/trainer_state.json ] && grep -q done runs/ft_$m/log.jsonl 2>/dev/null || \
    $PY scripts/train.py configs/ft_$m.yaml
done
echo QUEUE_S4_DONE
