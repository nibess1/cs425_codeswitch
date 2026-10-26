#!/usr/bin/env bash
# Third run, unattended tail (26 Oct; smu2 access ending). Runs after queue_tl.sh on GPUs 5/7:
#   1. wait for the base grid (g0, g1 on GPU 5; g0.5 on GPU 7) and their val decodes
#   2. apply the val selection rule (scripts/select_tl.py) and RECORD it in PROCESS_LOG
#   3. GPU 7: small at the chosen gamma (+ its val, then test); GPU 5: test decodes of base runs
#   4. tables 25 / 25b / 25c (scripts/make_tl_tables.py)
# No test file is decoded before step 2 is written (queue_tl_test.sh refuses otherwise).
set -u
cd "$(dirname "$0")/.."
PY=${PY:-~/micromamba/envs/cs425/bin/python}
LOG=docs/PROCESS_LOG.md
until grep -q QUEUE_TL_GPU5_DONE runs/queue_tl_gpu5.log && grep -q QUEUE_TL_GPU7_DONE runs/queue_tl_gpu7.log; do sleep 60; done
echo "base grid done $(date)"

$PY scripts/select_tl.py --baseline full_base__fp16__full_val --runs fulltl_base_g0.5 fulltl_base_g1 \
    --control fulltl_base_g0 > runs/select_tl.log 2>&1 || { echo SELECT_FAIL; exit 1; }
CHOSEN=$($PY -c "import json;print(json.load(open('results/chosen_tl.json'))['chosen'] or '')")
{
  echo
  echo "### $(date +'%d %b %H:%M'): third-run selection recorded (automatic, before any test decode)"
  echo "Applied by \`scripts/select_tl.py\` (rule above); val numbers in \`results/tl_val.csv\` (all % as fractions)."
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
bash scripts/queue_tl_test.sh 5 fulltl_base_g0 fulltl_base_g0.5 fulltl_base_g1 > runs/queue_tl_test_gpu5.log 2>&1
wait
$PY scripts/make_tl_tables.py > runs/tl_tables.log 2>&1 && echo TABLES_DONE
echo QUEUE_TL_ALL_DONE $(date)
