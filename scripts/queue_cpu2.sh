#!/usr/bin/env bash
# CPU queue 2: deployed formats needed for the quantization fixes.
#  - full_base had no CPU exports yet: CTranslate2 int8 / f32 and whisper.cpp Q4_0 / F16 (+Q8/Q5)
#  - QAT-merged models: whisper.cpp Q4_0 and CTranslate2 int8 (16-bit references are the
#    unmerged model's same-backend runs)
set -u
cd "$(dirname "$0")/.."
PY=${PY:-~/micromamba/envs/cs425/bin/python}
W=${W:-48}
FULL=$($PY -c "import json;print(json.load(open('results/chosen.json'))['full_base'])")
$PY scripts/deploy_eval.py --model ckpt/$FULL/best --tag full_base --method full --workers $W \
    --formats ct2_f32 ct2_int8 ggml_f16 ggml_q8_0 ggml_q5_0 ggml_q4_0
for spec in "ft_tiny:ft:qat_ft_tiny" "ft_base:ft:qat_ft_base" "full_base:full:qat_full_base"; do
  IFS=: read tag src run <<< "$spec"
  bash scripts/wait_for.sh ckpt/${run}_merged/MERGED_DONE
  $PY scripts/deploy_eval.py --model ckpt/${run}_merged --tag $tag --method qat_${src}_merged --workers $W \
      --formats ct2_int8 ggml_q4_0
done
echo QUEUE_CPU2_DONE
