#!/usr/bin/env bash
# S4: plain full fine-tuning of base, small, tiny (base first: methods are tuned on it).
set -u
cd "$(dirname "$0")/.."
PY=${PY:-~/micromamba/envs/cs425/bin/python}
for m in base small tiny; do
  [ -f runs/ft_$m/done.flag ] || \
    $PY scripts/train.py configs/ft_$m.yaml
done
echo QUEUE_S4_DONE
