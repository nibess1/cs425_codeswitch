#!/usr/bin/env bash
# Queue E2 (CPU): same-backend 16-bit references (CTranslate2 float32, whisper.cpp F16) for every
# deployed model, so deployed R compares the same decoder implementation.
set -u
cd "$(dirname "$0")/.."
PY=${PY:-~/micromamba/envs/cs425/bin/python}
W=${W:-32}
for spec in tiny:tiny:offshelf_free base:base:offshelf_free small:small:offshelf_free \
            ckpt/ft_tiny/best:ft_tiny:ft ckpt/ft_base/best:ft_base:ft ckpt/ft_small/best:ft_small:ft \
            ckpt/full_small/best:full_small:full; do
  IFS=: read model tag method <<< "$spec"
  if [[ $model == ckpt/* ]]; then r=${model#ckpt/}; bash scripts/wait_for.sh run:${r%/best}; fi
  $PY scripts/deploy_eval.py --model $model --tag $tag --method $method --formats ct2_f32 ggml_f16 --workers $W
done
echo QUEUE_E2_DONE
