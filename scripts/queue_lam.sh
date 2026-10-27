#!/usr/bin/env bash
# Fourth run (27 Oct): lambda retune of the transliteration-penalised MRT on base (rules in
# docs/PROCESS_LOG.md, "27 Oct: fourth run"). GPUs 5 and 7 only.
#   1. train fulltl_base_l<lambda>_g<gamma> for lambda in {0.4, 1} x gamma in {0, 3}; decode VAL
#   2. apply the val rule (select_tl.py --tag lam --cwer_margin 1.0) and RECORD it in PROCESS_LOG
#   3. decode TEST for all four runs; tables 26 / 26b / 26c
set -u
cd "$(dirname "$0")/.."
PY=${PY:-~/micromamba/envs/cs425/bin/python}
LOG=docs/PROCESS_LOG.md
trained() { grep -qs '"event": "done"' runs/$1/log.jsonl && [ -f preds/$1__fp16__fulltl_val.meta.json ]; }
run_val() {  # run_val <gpu> <lambda> <gamma>
  local RUN=fulltl_base_l$2_g$3
  trained $RUN && return 0
  CUDA_VISIBLE_DEVICES=$1 $PY scripts/train.py configs/ft_base.yaml --run $RUN --alpha 0.5 --beta 0.5 \
      --lambda_sem $2 --rho0 0.5 --translit_gamma $3 --save_epochs true && \
  CUDA_VISIBLE_DEVICES=$1 $PY scripts/decode.py --model ckpt/$RUN/best --model_tag $RUN --precision fp16 \
      --lang free --method fulltl_val --sets ascend_val nsc_val --batch 48 || echo "FAIL $RUN"
}
test_dec() {  # test_dec <gpu> <run> ...
  local g=$1; shift
  for RUN in "$@"; do
    [ -f preds/${RUN}__fp16__fulltl.meta.json ] || CUDA_VISIBLE_DEVICES=$g $PY scripts/decode.py --model ckpt/$RUN/best \
        --model_tag $RUN --precision fp16 --lang free --method fulltl --batch 48
  done
}
( run_val 5 0.4 0 > runs/queue_lam_gpu5.log 2>&1; run_val 5 1 0 >> runs/queue_lam_gpu5.log 2>&1 ) &
( run_val 7 0.4 3 > runs/queue_lam_gpu7.log 2>&1; run_val 7 1 3 >> runs/queue_lam_gpu7.log 2>&1 ) &
wait
echo "grid done $(date)"
for R in l0.4_g0 l1_g0 l0.4_g3 l1_g3; do trained fulltl_base_$R || { echo "MISSING fulltl_base_$R"; exit 1; }; done

$PY scripts/select_tl.py --tag lam --cwer_margin 1.0 --baseline full_base__fp16__full_val \
    --runs fulltl_base_l0.4_g3 fulltl_base_l1_g3 --control fulltl_base_l0.4_g0 fulltl_base_l1_g0 \
    > runs/select_lam.log 2>&1 || { echo SELECT_FAIL; exit 1; }
CHOSEN=$($PY -c "import json;print(json.load(open('results/chosen_lam.json'))['chosen'] or '')")
{
  echo
  echo "### $(date +'%d %b %H:%M'): fourth-run (λ retune) selection recorded, before any test decode"
  echo "Applied by \`scripts/select_tl.py --tag lam --cwer_margin 1.0\`; val numbers in \`results/lam_val.csv\` (fractions)."
  echo '```'
  grep -v -i "warn" runs/select_lam.log | tail -8
  echo '```'
  if [ -n "$CHOSEN" ]; then echo "- Chosen: **$CHOSEN**."; else echo "- **No run qualified: negative result.**"; fi
} >> $LOG
echo "selection: ${CHOSEN:-none}"

test_dec 5 fulltl_base_l0.4_g0 fulltl_base_l0.4_g3 > runs/queue_lam_test_gpu5.log 2>&1 &
test_dec 7 fulltl_base_l1_g0 fulltl_base_l1_g3 > runs/queue_lam_test_gpu7.log 2>&1 &
wait
P=""
for R in l0.4_g0 l0.4_g3 l1_g0 l1_g3; do P="$P full_base__fp16__full:fulltl_base_${R}__fp16__fulltl"; done
P="$P fulltl_base_l0.4_g0__fp16__fulltl:fulltl_base_l0.4_g3__fp16__fulltl fulltl_base_l1_g0__fp16__fulltl:fulltl_base_l1_g3__fp16__fulltl"
$PY scripts/make_tl_tables.py --pairs $P --names 26_lambda_retune,26b_lambda_retune_val,26c_lambda_retune_absolute \
    --val_csv results/lam_val.csv > runs/lam_tables.log 2>&1 && echo TABLES_DONE
echo QUEUE_LAM_DONE $(date)
