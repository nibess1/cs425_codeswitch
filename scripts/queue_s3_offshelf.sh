#!/usr/bin/env bash
# S3: off-the-shelf Whisper baselines. fp16 with all four language prompts; bnb int8 / NF4 with
# the free (predicted) language token. Greedy decoding, 128-token cap.
set -u
cd "$(dirname "$0")/.."
PY=${PY:-~/micromamba/envs/cs425/bin/python}
for m in tiny base small; do
  for lang in free zh en both; do
    [ -f preds/${m}__fp16__offshelf_${lang}.meta.json ] || \
      $PY scripts/decode.py --model $m --precision fp16 --lang $lang --method offshelf_$lang --batch 48
  done
  for p in int8_bnb nf4_bnb; do
    [ -f preds/${m}__${p}__offshelf_free.meta.json ] || \
      $PY scripts/decode.py --model $m --precision $p --lang free --method offshelf_free --batch 48
  done
done
echo QUEUE_S3_DONE
