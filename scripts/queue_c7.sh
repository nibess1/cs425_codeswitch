#!/usr/bin/env bash
# Queue C7 (GPU 7): C2 per-model LID heads, C1 LID-gated re-ranking (existing + own heads),
# E Diff Room recording-condition ablation.
set -u
cd "$(dirname "$0")/.."
PY=${PY:-~/micromamba/envs/cs425/bin/python}
trained() { grep -qs '"event": "done"' runs/$1/log.jsonl; }
FULL=$($PY -c "import json;print(json.load(open('results/chosen.json'))['full_base'])")
# C2: heads on each re-ranked model's own frozen encoder
[ -f ckpt/lid_full_base/head.pt ] || $PY scripts/train_lid.py --encoder ckpt/$FULL/best --run lid_full_base
[ -f ckpt/lid_full_small/head.pt ] || $PY scripts/train_lid.py --encoder ckpt/full_small/best --run lid_full_small
# C1 with the existing head (ft_base encoder)
$PY scripts/gated_rerank.py --tag full_base --greedy full_base__fp16__full --lid ckpt/lid_ft_base/head.pt --out_tag full_base
$PY scripts/gated_rerank.py --tag full_small --greedy full_small__fp16__full --lid ckpt/lid_ft_base/head.pt --out_tag full_small
# C2 -> C1: re-ranking features with each model's own head, then filtered + gated variants
$PY scripts/rerank.py --model ckpt/$FULL/best --tag fullown_base --lid ckpt/lid_full_base/head.pt
$PY scripts/rerank.py --model ckpt/$FULL/best --tag fullown_base --lid ckpt/lid_full_base/head.pt --loop_filter
$PY scripts/gated_rerank.py --tag fullown_base --greedy full_base__fp16__full --lid ckpt/lid_full_base/head.pt --out_tag fullown_base
$PY scripts/rerank.py --model ckpt/full_small/best --tag fullown_small --lid ckpt/lid_full_small/head.pt
$PY scripts/rerank.py --model ckpt/full_small/best --tag fullown_small --lid ckpt/lid_full_small/head.pt --loop_filter
$PY scripts/gated_rerank.py --tag fullown_small --greedy full_small__fp16__full --lid ckpt/lid_full_small/head.pt --out_tag fullown_small
# E: recording-condition ablation (A = existing ft_base)
until grep -q EPREP_DONE runs/prep_e.log; do sleep 60; done
if [ -f data/nsc_e/manifests/diff_test.jsonl ]; then
  VS='[ascend_val, nsc_val, nsce_diff_val]'
  trained ediff_base || $PY scripts/train.py configs/ft_base.yaml --run ediff_base --train_sets '[ascend_train, nsce_diff_train]' --val_sets "$VS"
  trained emix_base || $PY scripts/train.py configs/ft_base.yaml --run emix_base --train_sets '[ascend_train, nsce_mix_train]' --val_sets "$VS"
  for spec in ft_base:esame_base ediff_base:ediff_base emix_base:emix_base; do
    ck=${spec%%:*}; tag=${spec##*:}
    [ -f preds/${tag}__fp16__eabl.meta.json ] || $PY scripts/decode.py --model ckpt/$ck/best --model_tag $tag \
        --precision fp16 --lang free --method eabl --sets ascend_test nsc_test nsce_diff_test --batch 48
  done
fi
echo QUEUE_C7_DONE
