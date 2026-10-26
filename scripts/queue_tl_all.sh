#!/usr/bin/env bash
# Third run, unattended tail (26 Oct; smu2 access ending). Runs after queue_tl.sh on GPUs 5/7:
#   1. wait for the base grid; any base run that crashed is retrained under a NEW id <id>r
#      (never overwriting), split over GPUs 7 and 5, each followed by its val decode
#   2. apply the val selection rule (scripts/select_tl.py) and RECORD it in PROCESS_LOG
#   3. GPU 7: small at the chosen gamma (+ its val, then test); GPU 5: test decodes of base runs
#   4. tables 25 / 25b / 25c (scripts/make_tl_tables.py)
# No test file is decoded before step 2 is written (queue_tl_test.sh refuses otherwise).
set -u
cd "$(dirname "$0")/.."
PY=${PY:-~/micromamba/envs/cs425/bin/python}
LOG=docs/PROCESS_LOG.md
trained() { grep -qs '"event": "done"' runs/$1/log.jsonl && [ -f preds/$1__fp16__fulltl_val.meta.json ]; }
until grep -q QUEUE_TL_GPU5_DONE runs/queue_tl_gpu5.log && grep -q QUEUE_TL_GPU7_DONE runs/queue_tl_gpu7.log; do sleep 60; done
echo "base grid done $(date)"

retrain() {  # retrain <gpu> <gamma>: new id fulltl_base_g<gamma>r
  local RUN=fulltl_base_g$2r
  CUDA_VISIBLE_DEVICES=$1 $PY scripts/train.py configs/ft_base.yaml --run $RUN --alpha 0.5 --beta 0.5 \
      --lambda_sem 0.1 --rho0 0.5 --translit_gamma $2 --save_epochs true && \
  CUDA_VISIBLE_DEVICES=$1 $PY scripts/decode.py --model ckpt/$RUN/best --model_tag $RUN --precision fp16 \
      --lang free --method fulltl_val --sets ascend_val nsc_val --batch 48
}
FAILED=""
for G in 0 0.5 1; do trained fulltl_base_g$G || FAILED="$FAILED $G"; done
echo "failed base runs:${FAILED:- none}"
GPUS=(7 5); i=0; declare -A Q
for G in $FAILED; do g=${GPUS[$((i % 2))]}; Q[$g]="${Q[$g]:-} $G"; i=$((i + 1)); done
for g in "${!Q[@]}"; do ( for G in ${Q[$g]}; do retrain $g $G > runs/queue_tl_retrain_g$G.log 2>&1; done ) & done
wait
rid() { trained fulltl_base_g$1 && echo fulltl_base_g$1 || { trained fulltl_base_g$1r && echo fulltl_base_g$1r; }; }
G0=$(rid 0); G05=$(rid 0.5); G1=$(rid 1)
RUNS="$G05 $G1"
[ -n "$(echo $RUNS)" ] || { echo "NO_BASE_RUNS"; exit 1; }

$PY scripts/select_tl.py --baseline full_base__fp16__full_val --runs $RUNS \
    ${G0:+--control $G0} > runs/select_tl.log 2>&1 || { echo SELECT_FAIL; exit 1; }
CHOSEN=$($PY -c "import json;print(json.load(open('results/chosen_tl.json'))['chosen'] or '')")
{
  echo
  echo "### $(date +'%d %b %H:%M'): third-run selection recorded (automatic, before any test decode)"
  echo "Applied by \`scripts/select_tl.py\` (rule above); val numbers in \`results/tl_val.csv\` (fractions, not %)."
  [ -n "$FAILED" ] && echo "- Retrained after a crash (new id with suffix r, same settings):$FAILED"
  echo '```'
  grep -v "Warn\|warn" runs/select_tl.log | tail -8
  echo '```'
  if [ -n "$CHOSEN" ]; then echo "- Chosen: **$CHOSEN**. Small is trained at this γ."
  else echo "- **No γ qualified: negative result.** Small was not trained (the team could not be asked: smu2 access ended)."; fi
} >> $LOG
echo "selection: ${CHOSEN:-none}"

if [ -n "$CHOSEN" ]; then
  ( bash scripts/queue_tl_small.sh 7 > runs/queue_tl_small.log 2>&1 && \
    G=$($PY -c "import json;print('%g'%json.load(open('results/chosen_tl.json'))['gamma'])") && \
    bash scripts/queue_tl_test.sh 7 fulltl_small_g$G > runs/queue_tl_test_gpu7.log 2>&1 ) &
fi
bash scripts/queue_tl_test.sh 5 $G0 $RUNS > runs/queue_tl_test_gpu5.log 2>&1
wait
$PY scripts/make_tl_tables.py > runs/tl_tables.log 2>&1 && echo TABLES_DONE
echo QUEUE_TL_ALL_DONE $(date)
