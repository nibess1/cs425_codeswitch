#!/usr/bin/env bash
# C1 rerun for full_base (GPU 7): its N-best feature cache was lost, so rebuild it under a new tag
# (fullshared_base = full_base + the shared lid_ft_base head; new system ids, nothing overwritten).
set -u
cd "$(dirname "$0")/.."
PY=${PY:-~/micromamba/envs/cs425/bin/python}
until grep -q QUEUE_C7B_DONE runs/queue_c7b.log; do sleep 60; done
FULL=$($PY -c "import json;print(json.load(open(\"results/chosen.json\"))[\"full_base\"])")
$PY scripts/rerank.py --model ckpt/$FULL/best --tag fullshared_base --lid ckpt/lid_ft_base/head.pt
$PY scripts/rerank.py --model ckpt/$FULL/best --tag fullshared_base --lid ckpt/lid_ft_base/head.pt --loop_filter
$PY scripts/gated_rerank.py --tag fullshared_base --greedy full_base__fp16__full --lid ckpt/lid_ft_base/head.pt --out_tag full_base
echo QUEUE_C7C_DONE
