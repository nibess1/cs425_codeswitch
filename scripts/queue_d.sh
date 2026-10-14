#!/usr/bin/env bash
# Queue D (after queue B): select role weights, semantic grid, ablations, small transfer,
# evaluation at every precision, re-ranking, sensitivity scans, QLoRA QAT.
set -u
cd "$(dirname "$0")/.."
PY=${PY:-~/micromamba/envs/cs425/bin/python}
until grep -q QUEUE_B_DONE runs/queue_b.log; do sleep 60; done
cfgval() { $PY -c "import json;print(json.load(open('runs/$1/config.json'))['$2'])"; }
trained() { grep -qs '"event": "done"' runs/$1/log.jsonl; }

# 1) role-weighted grid -> best alpha, beta (val)
RW=$($PY scripts/select_run.py --grid rw --baseline ft_base \
      --runs rw_base_a0.5_b0 rw_base_a0.5_b0.5 rw_base_a1_b0 rw_base_a1_b0.5 rw_base_a2_b0 rw_base_a2_b0.5 | tail -1)
A=$(cfgval $RW alpha); B=$(cfgval $RW beta)
echo "chosen role weights: $RW alpha=$A beta=$B"

# 2) semantic minimum-risk grid on top of the chosen role weights
SEMRUNS=""
for l in 0.1 0.3; do for r in 0.2 0.5; do
  run=sem_base_l${l}_r${r}; SEMRUNS="$SEMRUNS $run"
  trained $run || $PY scripts/train.py configs/ft_base.yaml --run $run --alpha $A --beta $B --lambda_sem $l --rho0 $r
done; done
FULL=$($PY scripts/select_run.py --grid sem --baseline $RW --runs $SEMRUNS | tail -1)
L=$(cfgval $FULL lambda_sem); RHO=$(cfgval $FULL rho0)
echo "chosen semantic loss: $FULL lambda=$L rho0=$RHO"

# 3) ablation: semantic loss alone (alpha = beta = 0)
trained abl_base_semonly || $PY scripts/train.py configs/ft_base.yaml --run abl_base_semonly --lambda_sem $L --rho0 $RHO

# 4) transfer to small, once, with the base settings: full system + best single-method subset
trained full_small || $PY scripts/train.py configs/ft_small.yaml --run full_small --alpha $A --beta $B --lambda_sem $L --rho0 $RHO
SUB=$($PY - <<PY
import json
v=json.load(open("results/select_rw.json"))["val"]
print("rw" if v["$RW"]["cs_E_emb"] <= v.get("abl_base_semonly", {"cs_E_emb": 9})["cs_E_emb"] else "sem")
PY
)
if [ "$SUB" = rw ]; then
  trained sub_small || $PY scripts/train.py configs/ft_small.yaml --run sub_small --alpha $A --beta $B
else
  trained sub_small || $PY scripts/train.py configs/ft_small.yaml --run sub_small --lambda_sem $L --rho0 $RHO
fi
echo "best subset for small: $SUB"
echo "{\"rw\": \"$RW\", \"full_base\": \"$FULL\", \"alpha\": $A, \"beta\": $B, \"lambda_sem\": $L, \"rho0\": $RHO, \"small_subset\": \"$SUB\"}" > results/chosen.json

# 5) evaluate every final model at fp16 / int8 / NF4
for spec in "$RW:rw" "$FULL:full" "abl_base_semonly:semonly" "full_small:full" "sub_small:sub"; do
  run=${spec%%:*}; m=${spec##*:}
  size=$(cfgval $run model)
  bash scripts/eval_model.sh ckpt/$run/best ${m}_${size} $m
done

# 6) re-ranking (needs the LID head): plain FT base, full base, full small
$PY scripts/rerank.py --model ckpt/ft_base/best --tag ft_base --lid ckpt/lid_ft_base/head.pt
$PY scripts/rerank.py --model ckpt/$FULL/best --tag full_base --lid ckpt/lid_ft_base/head.pt
$PY scripts/rerank.py --model ckpt/full_small/best --tag full_small --lid ckpt/lid_ft_base/head.pt

# 7) sensitivity scan + mixed precision (NF4) on small: plain FT and full system
$PY scripts/sensitivity.py --model ckpt/ft_small/best --tag ft_small
$PY scripts/sensitivity.py --model ckpt/full_small/best --tag full_small

# 8) QLoRA QAT on the NF4 small models, same loss as their 16-bit training
trained qat_ft_small || $PY scripts/train.py configs/ft_small.yaml --run qat_ft_small --model ckpt/ft_small/best \
    --lora true --lr 1e-4 --epochs 1 --warmup 100
trained qat_full_small || $PY scripts/train.py configs/ft_small.yaml --run qat_full_small --model ckpt/full_small/best \
    --lora true --lr 1e-4 --epochs 1 --warmup 100 --alpha $A --beta $B --lambda_sem $L --rho0 $RHO
for spec in "ft_small:ft:qat_ft_small" "full_small:full:qat_full_small"; do
  IFS=: read tag src run <<< "$spec"
  [ -f preds/${tag}__nf4_qlora__qat_${src}.meta.json ] || $PY scripts/decode.py --model ckpt/$tag/best \
      --adapter ckpt/$run/best --model_tag $tag --precision nf4_qlora --method qat_${src} --batch 48
  [ -d ckpt/${run}_merged ] || $PY -c "from csasr.quant.qat import merge_adapter; merge_adapter('ckpt/$tag/best','ckpt/$run/best','ckpt/${run}_merged')"
  PRECS="nf4_bnb" bash scripts/eval_model.sh ckpt/${run}_merged $tag qat_${src}_merged
done
echo QUEUE_D_DONE
