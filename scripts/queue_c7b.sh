#!/usr/bin/env bash
# C1/C2 rerun (GPU 7): gated re-ranking keyed by (set, utt_id); see PROCESS_LOG 25 Oct.
set -u
cd "$(dirname "$0")/.."
PY=${PY:-~/micromamba/envs/cs425/bin/python}
$PY scripts/gated_rerank.py --tag full_base --greedy full_base__fp16__full --lid ckpt/lid_ft_base/head.pt --out_tag full_base
$PY scripts/gated_rerank.py --tag full_small --greedy full_small__fp16__full --lid ckpt/lid_ft_base/head.pt --out_tag full_small
$PY scripts/gated_rerank.py --tag fullown_base --greedy full_base__fp16__full --lid ckpt/lid_full_base/head.pt --out_tag fullown_base
$PY scripts/gated_rerank.py --tag fullown_small --greedy full_small__fp16__full --lid ckpt/lid_full_small/head.pt --out_tag fullown_small
echo QUEUE_C7B_DONE
