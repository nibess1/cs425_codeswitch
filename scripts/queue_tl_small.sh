#!/usr/bin/env bash
# Third run: one small run at the val-chosen gamma (results/chosen_tl.json), full_small settings.
#   bash scripts/queue_tl_small.sh <gpu>
set -u
cd "$(dirname "$0")/.."
PY=${PY:-~/micromamba/envs/cs425/bin/python}
export CUDA_VISIBLE_DEVICES=$1
G=$($PY -c "import json;g=json.load(open('results/chosen_tl.json'))['gamma'];print('' if g is None else ('%g'%g))")
[ -n "$G" ] || { echo "no gamma chosen (negative result); not training small"; exit 1; }
RUN=fulltl_small_g$G
grep -qs '"event": "done"' runs/$RUN/log.jsonl || $PY scripts/train.py configs/ft_small.yaml --run $RUN \
    --alpha 0.5 --beta 0.5 --lambda_sem 0.1 --rho0 0.5 --mrt_batch 4 --translit_gamma $G --save_epochs true || exit 1
[ -f preds/${RUN}__fp16__fulltl_val.meta.json ] || \
  $PY scripts/decode.py --model ckpt/$RUN/best --model_tag $RUN --precision fp16 --lang free \
      --method fulltl_val --sets ascend_val nsc_val --batch 48
echo QUEUE_TL_SMALL_DONE
