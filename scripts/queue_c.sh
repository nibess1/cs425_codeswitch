#!/usr/bin/env bash
# Queue C: evaluate each plain fine-tuned model at every simulated precision as it appears.
set -u
cd "$(dirname "$0")/.."
until grep -q QUEUE_S3_DONE runs/queue_s3.log; do sleep 60; done  # avoid 3 decoders on one GPU
for m in base small tiny; do
  bash scripts/wait_for.sh run:ft_$m
  bash scripts/eval_model.sh ckpt/ft_$m/best ft_$m ft
done
echo QUEUE_C_DONE
