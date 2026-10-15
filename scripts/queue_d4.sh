#!/usr/bin/env bash
# Queue D4: QLoRA QAT of the full small system (ran out of memory in queue D2 at batch 16 with
# the semantic loss); batch 8 x 4 accumulation keeps the effective batch at 32.
set -u
cd "$(dirname "$0")/.."
PY=${PY:-~/micromamba/envs/cs425/bin/python}
get() { $PY -c "import json;print(json.load(open('results/chosen.json'))['$1'])"; }
A=$(get alpha); B=$(get beta); L=$(get lambda_sem); RHO=$(get rho0)
rm -rf runs/qat_full_small ckpt/qat_full_small
$PY scripts/train.py configs/ft_small.yaml --run qat_full_small --model ckpt/full_small/best --lora true \
    --lr 1e-4 --epochs 1 --warmup 100 --alpha $A --beta $B --lambda_sem $L --rho0 $RHO \
    --batch 8 --grad_accum 4 --mrt_batch 2
$PY scripts/decode.py --model ckpt/full_small/best --adapter ckpt/qat_full_small/best --model_tag full_small \
    --precision nf4_qlora --method qat_full --batch 48
$PY -c "from csasr.quant.qat import merge_adapter; merge_adapter('ckpt/full_small/best','ckpt/qat_full_small/best','ckpt/qat_full_small_merged')"
PRECS="nf4_bnb" bash scripts/eval_model.sh ckpt/qat_full_small_merged full_small qat_full_merged
$PY scripts/rerank.py --model ckpt/sem_base_l0.1_r0.5/best --tag full_base --lid ckpt/lid_ft_base/head.pt --loop_filter || true
$PY scripts/rerank.py --model ckpt/full_small/best --tag full_small --lid ckpt/lid_ft_base/head.pt --loop_filter || true
echo QUEUE_D4_DONE
