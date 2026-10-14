#!/usr/bin/env bash
# Queue B: FLEURS splices -> LID head on ft_base encoder -> role-weighted grid on base.
set -u
cd "$(dirname "$0")/.."
PY=${PY:-~/micromamba/envs/cs425/bin/python}
bash scripts/wait_for.sh run:ft_base
[ -f data/align/fleurs_splice_train.pkl ] || $PY scripts/prepare_fleurs.py
until grep -q ALIGN_DONE runs/align.log; do sleep 60; done
[ -f ckpt/lid_ft_base/head.pt ] || $PY scripts/train_lid.py --encoder ckpt/ft_base/best --run lid_ft_base
for a in 0.5 1 2; do for b in 0 0.5; do
  run=rw_base_a${a}_b${b}
  grep -qs "\"event\": \"done\"" runs/$run/log.jsonl || $PY scripts/train.py configs/ft_base.yaml --run $run --alpha $a --beta $b
done; done
echo QUEUE_B_DONE
