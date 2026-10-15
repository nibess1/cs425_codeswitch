#!/usr/bin/env bash
# Queue D, second pass: the steps that failed in queue D (full_small OOM -> everything that
# needed it; re-ranking CTC length bug; QAT learning-rate parsing). Settings from results/chosen.json.
set -u
cd "$(dirname "$0")/.."
PY=${PY:-~/micromamba/envs/cs425/bin/python}
trained() { grep -qs '"event": "done"' runs/$1/log.jsonl; }
get() { $PY -c "import json;print(json.load(open('results/chosen.json'))['$1'])"; }
A=$(get alpha); B=$(get beta); L=$(get lambda_sem); RHO=$(get rho0); FULL=$(get full_base)
# small + MRT does not fit next to other jobs with an 8-clip MRT sub-batch
trained full_small || { rm -rf runs/full_small ckpt/full_small; $PY scripts/train.py configs/ft_small.yaml --run full_small \
    --alpha $A --beta $B --lambda_sem $L --rho0 $RHO --mrt_batch 4; }
bash scripts/eval_model.sh ckpt/full_small/best full_small full
$PY scripts/rerank.py --model ckpt/ft_base/best --tag ft_base --lid ckpt/lid_ft_base/head.pt
$PY scripts/rerank.py --model ckpt/$FULL/best --tag full_base --lid ckpt/lid_ft_base/head.pt
$PY scripts/rerank.py --model ckpt/full_small/best --tag full_small --lid ckpt/lid_ft_base/head.pt
$PY scripts/sensitivity.py --model ckpt/full_small/best --tag full_small
trained qat_ft_small || { rm -rf runs/qat_ft_small; $PY scripts/train.py configs/ft_small.yaml --run qat_ft_small \
    --model ckpt/ft_small/best --lora true --lr 1e-4 --epochs 1 --warmup 100; }
trained qat_full_small || { rm -rf runs/qat_full_small; $PY scripts/train.py configs/ft_small.yaml --run qat_full_small \
    --model ckpt/full_small/best --lora true --lr 1e-4 --epochs 1 --warmup 100 --alpha $A --beta $B \
    --lambda_sem $L --rho0 $RHO --mrt_batch 4; }
for spec in "ft_small:ft:qat_ft_small" "full_small:full:qat_full_small"; do
  IFS=: read tag src run <<< "$spec"
  [ -f preds/${tag}__nf4_qlora__qat_${src}.meta.json ] || $PY scripts/decode.py --model ckpt/$tag/best \
      --adapter ckpt/$run/best --model_tag $tag --precision nf4_qlora --method qat_${src} --batch 48
  [ -d ckpt/${run}_merged ] || $PY -c "from csasr.quant.qat import merge_adapter; merge_adapter('ckpt/$tag/best','ckpt/$run/best','ckpt/${run}_merged')"
  PRECS="nf4_bnb" bash scripts/eval_model.sh ckpt/${run}_merged $tag qat_${src}_merged
done
echo QUEUE_D2_DONE
