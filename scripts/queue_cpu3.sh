#!/usr/bin/env bash
# CPU queue 3 (16 workers): full_base whisper.cpp Q4_0/Q5_0, deployed mixed precision (B2),
# then CPU exports of the QAT-merged models (B3).
set -u
cd "$(dirname "$0")/.."
PY=${PY:-~/micromamba/envs/cs425/bin/python}
W=${W:-16}
FULL=$($PY -c "import json;print(json.load(open('results/chosen.json'))['full_base'])")
$PY scripts/deploy_eval.py --model ckpt/$FULL/best --tag full_base --method full --workers $W --formats ggml_q4_0
for spec in; do  # B2 deployed mixed precision: whisper.cpp loader cannot load per-tensor types (see PROCESS_LOG)
  $PY scripts/deploy_mixed.py --tag ${spec%%:*} --method ${spec##*:} --workers $W
done
for spec in "ft_tiny:ft:qat_ft_tiny" "ft_base:ft:qat_ft_base" "full_base:full:qat_full_base"; do
  IFS=: read tag src run <<< "$spec"
  bash scripts/wait_for.sh ckpt/${run}_merged/MERGED_DONE
  $PY scripts/deploy_eval.py --model ckpt/${run}_merged --tag $tag --method qat_${src}_merged --workers $W \
      --formats ggml_q4_0 ct2_int8
done
$PY scripts/deploy_eval.py --model ckpt/$FULL/best --tag full_base --method full --workers $W --formats ggml_q5_0
echo QUEUE_CPU3_DONE
