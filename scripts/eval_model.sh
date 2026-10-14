#!/usr/bin/env bash
# Decode a checkpoint on the test sets at fp16 / bnb int8 / bnb NF4 (+ fp16 on val).
#   bash scripts/eval_model.sh ckpt/ft_base/best ft_base ft
set -u
cd "$(dirname "$0")/.."
PY=${PY:-~/micromamba/envs/cs425/bin/python}
CK=$1; TAG=$2; METHOD=$3; shift 3
PRECS=${PRECS:-"fp16 int8_bnb nf4_bnb"}
for p in $PRECS; do
  [ -f preds/${TAG}__${p}__${METHOD}.meta.json ] || \
    $PY scripts/decode.py --model $CK --model_tag $TAG --precision $p --lang free --method $METHOD --batch 48
done
[ -f preds/${TAG}__fp16__${METHOD}_val.meta.json ] || \
  $PY scripts/decode.py --model $CK --model_tag $TAG --precision fp16 --lang free --method ${METHOD}_val \
      --sets ascend_val nsc_val --batch 48
