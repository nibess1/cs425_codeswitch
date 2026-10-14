#!/usr/bin/env bash
# Queue E (CPU only): deployed formats (CTranslate2 int8, whisper.cpp Q8_0/Q5_0/Q4_0) for
# off-the-shelf and fine-tuned models, as their checkpoints appear.
set -u
cd "$(dirname "$0")/.."
PY=${PY:-~/micromamba/envs/cs425/bin/python}
W=${W:-32}
for m in tiny base small; do
  $PY scripts/deploy_eval.py --model $m --tag $m --method offshelf_free --workers $W
done
for m in base small tiny; do
  bash scripts/wait_for.sh run:ft_$m
  $PY scripts/deploy_eval.py --model ckpt/ft_$m/best --tag ft_$m --method ft --workers $W
done
bash scripts/wait_for.sh run:full_small
$PY scripts/deploy_eval.py --model ckpt/full_small/best --tag full_small --method full --workers $W
for spec in "ft_small:ft:qat_ft_small" "full_small:full:qat_full_small"; do
  IFS=: read tag src run <<< "$spec"
  bash scripts/wait_for.sh ckpt/${run}_merged/config.json
  $PY scripts/deploy_eval.py --model ckpt/${run}_merged --tag $tag --method qat_${src}_merged --workers $W
done
echo QUEUE_E_DONE
