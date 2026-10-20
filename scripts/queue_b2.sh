#!/usr/bin/env bash
# Queue B2 (GPU 5): quantization fixes on the models where 4-bit hurts embedded units
# (ft_tiny, ft_base, full_base): A6 sensitivity scan + mixed precision, then QLoRA QAT.
set -u
cd "$(dirname "$0")/.."
PY=${PY:-~/micromamba/envs/cs425/bin/python}
trained() { grep -qs '"event": "done"' runs/$1/log.jsonl; }
get() { $PY -c "import json;print(json.load(open('results/chosen.json'))['$1'])"; }
A=$(get alpha); B=$(get beta); L=$(get lambda_sem); RHO=$(get rho0); FULL=$(get full_base)
# B1: sensitivity + mixed precision (k chosen on val inside sensitivity.py, then one test run)
$PY scripts/sensitivity.py --model ckpt/ft_tiny/best --tag ft_tiny
$PY scripts/sensitivity.py --model ckpt/ft_base/best --tag ft_base
$PY scripts/sensitivity.py --model ckpt/$FULL/best --tag full_base
# B3: QLoRA QAT, same loss as each model's own training
trained qat_ft_tiny || $PY scripts/train.py configs/ft_tiny.yaml --run qat_ft_tiny --model ckpt/ft_tiny/best \
    --lora true --lr 1e-4 --epochs 1 --warmup 100
trained qat_ft_base || $PY scripts/train.py configs/ft_base.yaml --run qat_ft_base --model ckpt/ft_base/best \
    --lora true --lr 1e-4 --epochs 1 --warmup 100
trained qat_full_base || $PY scripts/train.py configs/ft_base.yaml --run qat_full_base --model ckpt/$FULL/best \
    --lora true --lr 1e-4 --epochs 1 --warmup 100 --alpha $A --beta $B --lambda_sem $L --rho0 $RHO \
    --batch 16 --grad_accum 2 --mrt_batch 4
for spec in "ft_tiny:ft_tiny:ft:qat_ft_tiny" "ft_base:ft_base:ft:qat_ft_base" "$FULL:full_base:full:qat_full_base"; do
  IFS=: read ck tag src run <<< "$spec"
  [ -f preds/${tag}__nf4_qlora__qat_${src}.meta.json ] || $PY scripts/decode.py --model ckpt/$ck/best \
      --adapter ckpt/$run/best --model_tag $tag --precision nf4_qlora --method qat_${src} --batch 48
  [ -f ckpt/${run}_merged/vocab.json ] || $PY -c "from csasr.quant.qat import merge_adapter; merge_adapter('ckpt/$ck/best','ckpt/$run/best','ckpt/${run}_merged')"
  PRECS="nf4_bnb" bash scripts/eval_model.sh ckpt/${run}_merged $tag qat_${src}_merged
  touch ckpt/${run}_merged/MERGED_DONE
done
echo QUEUE_B2_DONE
