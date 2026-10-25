#!/usr/bin/env bash
# Queue E7 (GPU 7): section E recording-condition ablation (A = existing ft_base).
set -u
cd "$(dirname "$0")/.."
PY=${PY:-~/micromamba/envs/cs425/bin/python}
trained() { grep -qs '"event": "done"' runs/$1/log.jsonl; }
VS='[ascend_val, nsc_val, nsce_diff_val]'
trained ediff_base || { rm -rf runs/ediff_base ckpt/ediff_base; $PY scripts/train.py configs/ft_base.yaml --run ediff_base \
    --train_sets '[ascend_train, nsce_diff_train]' --val_sets "$VS"; }
trained emix_base || { rm -rf runs/emix_base ckpt/emix_base; $PY scripts/train.py configs/ft_base.yaml --run emix_base \
    --train_sets '[ascend_train, nsce_mix_train]' --val_sets "$VS"; }
for spec in ft_base:esame_base ediff_base:ediff_base emix_base:emix_base; do
  ck=${spec%%:*}; tag=${spec##*:}
  [ -f preds/${tag}__fp16__eabl.meta.json ] || $PY scripts/decode.py --model ckpt/$ck/best --model_tag $tag \
      --precision fp16 --lang free --method eabl --sets ascend_test nsc_test nsce_diff_test --batch 48
done
echo QUEUE_E7_DONE
