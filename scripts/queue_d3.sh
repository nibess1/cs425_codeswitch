#!/usr/bin/env bash
# Queue D3: replaces the tail of queue D2. Re-ranking of full_small is skipped (decided 15 Oct:
# re-ranking hurt NSC accuracy on base, see docs/PROCESS_LOG.md). Waits for the full_base
# re-ranking to finish, stops queue D2, then runs sensitivity + QAT.
set -u
cd "$(dirname "$0")/.."
PY=${PY:-~/micromamba/envs/cs425/bin/python}
trained() { grep -qs '"event": "done"' runs/$1/log.jsonl; }
get() { $PY -c "import json;print(json.load(open('results/chosen.json'))['$1'])"; }
A=$(get alpha); B=$(get beta); L=$(get lambda_sem); RHO=$(get rho0)
bash scripts/wait_for.sh preds/full_base__fp16__rerank_ondevice.meta.json
sleep 20
tmux kill-session -t cs425_d2 2>/dev/null
pkill -f "rerank.py --model ckpt/full_small" 2>/dev/null
echo "queue D2 stopped after full_base re-ranking"
# post-hoc loop-filter variant for full_base, from its cached features (CPU only)
$PY scripts/rerank.py --model ckpt/sem_base_l0.1_r0.5/best --tag full_base --lid ckpt/lid_ft_base/head.pt --loop_filter || true
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
echo QUEUE_D3_DONE
