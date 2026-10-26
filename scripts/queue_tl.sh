#!/usr/bin/env bash
# Third run (26 Oct): transliteration-penalised MRT on base. One queue per GPU:
#   bash scripts/queue_tl.sh <gpu> <gamma> [<gamma> ...]
# Each gamma: train fulltl_base_g<gamma> (full_base settings) then decode VAL only (fp16 greedy).
# Test is decoded only after the selection is recorded (docs/PROCESS_LOG.md, 26 Oct).
set -u
cd "$(dirname "$0")/.."
PY=${PY:-~/micromamba/envs/cs425/bin/python}
GPU=$1; shift
export CUDA_VISIBLE_DEVICES=$GPU
trained() { grep -qs '"event": "done"' runs/$1/log.jsonl; }
for G in "$@"; do
  RUN=fulltl_base_g$G
  trained $RUN || $PY scripts/train.py configs/ft_base.yaml --run $RUN --alpha 0.5 --beta 0.5 \
      --lambda_sem 0.1 --rho0 0.5 --translit_gamma $G --save_epochs true || { echo "TRAIN_FAIL $RUN"; continue; }
  [ -f preds/${RUN}__fp16__fulltl_val.meta.json ] || \
    $PY scripts/decode.py --model ckpt/$RUN/best --model_tag $RUN --precision fp16 --lang free \
        --method fulltl_val --sets ascend_val nsc_val --batch 48
  echo "DONE $RUN"
done
echo QUEUE_TL_GPU${GPU}_DONE
