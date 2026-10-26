#!/usr/bin/env bash
# Third run: decode TEST (fp16 greedy, free language) for the given runs. Only after the val
# selection is recorded in docs/PROCESS_LOG.md.   bash scripts/queue_tl_test.sh <gpu> <run> ...
set -u
cd "$(dirname "$0")/.."
PY=${PY:-~/micromamba/envs/cs425/bin/python}
export CUDA_VISIBLE_DEVICES=$1; shift
grep -qs "selection recorded" docs/PROCESS_LOG.md || { echo "selection not recorded; refusing"; exit 1; }
for RUN in "$@"; do
  [ -f preds/${RUN}__fp16__fulltl.meta.json ] || \
    $PY scripts/decode.py --model ckpt/$RUN/best --model_tag $RUN --precision fp16 --lang free \
        --method fulltl --batch 48
  echo "TEST_DONE $RUN"
done
