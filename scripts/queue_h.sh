#!/usr/bin/env bash
# Queue H (7 Nov): human stress-set recordings (set stress_test) through the D5 decode list, plus the
# NF4 repairs of table 14 (H4). Method suffix _human, so no synthetic *_stress prediction is touched.
# Plan: docs/TESTING_PLAN.md.  Usage: GPU=5 bash scripts/queue_h.sh
set -u
cd "$(dirname "$0")/.."
PY=${PY:-~/micromamba/envs/cs425/bin/python}
GPU=${GPU:?set GPU to an idle GPU index}
SETS="stress_test"
[ $(wc -l < data/stress/manifests/test.jsonl) -eq 120 ] || { echo "stress_test manifest is not 120 utts"; exit 1; }
FULL=$($PY -c "import json;print(json.load(open('results/chosen.json'))['full_base'])")

gpu_idle() {  # no process and < 2 GB used on $GPU, else stop
  local used=$(nvidia-smi -i $GPU --query-gpu=memory.used --format=csv,noheader,nounits)
  local procs=$(nvidia-smi -i $GPU --query-compute-apps=pid --format=csv,noheader | grep -c .)
  echo "[$(date +%T)] GPU $GPU: ${used} MiB used, $procs processes"
  [ "$used" -lt 2048 ] && [ "$procs" -eq 0 ] || { echo "GPU $GPU is busy: stopping"; exit 1; }
}
dec() { [ -f preds/$2__$3__$4.meta.json ] || CUDA_VISIBLE_DEVICES=$GPU $PY scripts/decode.py --model $1 --model_tag $2 \
          --precision $3 --lang free --method $4 --sets $SETS --batch 48 "${@:5}"; }

# H1 + H2: GPU, HF fp16 / LLM.int8 / NF4
gpu_idle
dec small small fp16 offshelf_free_human
for spec in ckpt/ft_tiny/best:ft_tiny:ft ckpt/ft_base/best:ft_base:ft ckpt/ft_small/best:ft_small:ft \
            ckpt/$FULL/best:full_base:full ckpt/full_small/best:full_small:full; do
  IFS=: read ck tag m <<< "$spec"
  for p in fp16 int8_bnb nf4_bnb; do dec $ck $tag $p ${m}_human; done
done
echo H2_DONE

# H4 (GPU part): mixed precision with the val-chosen blocks, QLoRA adapters on NF4
gpu_idle
dec ckpt/ft_tiny/best ft_tiny nf4_bnb mp2_human --skip_modules model.encoder.layers.0 model.decoder.layers.3
dec ckpt/$FULL/best full_base nf4_bnb mp3_human --skip_modules model.encoder.layers.0 model.decoder.layers.5 model.encoder.layers.3
dec ckpt/ft_tiny/best ft_tiny nf4_qlora qat_ft_human --adapter ckpt/qat_ft_tiny/best
dec ckpt/$FULL/best full_base nf4_qlora qat_full_human --adapter ckpt/qat_full_base/best
echo H4_GPU_DONE

# H3: deployed formats on CPU (reuse existing exports), then H4 merged QAT -> Q4_0
for spec in ft_tiny:ft ft_base:ft ft_small:ft full_base:full full_small:full; do
  IFS=: read tag m <<< "$spec"
  CUDA_VISIBLE_DEVICES= $PY scripts/deploy_eval.py --model none --tag $tag --method ${m}_human --export_dir ${tag}_${m} \
      --sets $SETS --workers 16 --formats ct2_f32 ct2_int8 ggml_f16 ggml_q8_0 ggml_q4_0
done
for spec in ft_tiny:ft full_base:full; do
  IFS=: read tag m <<< "$spec"
  CUDA_VISIBLE_DEVICES= $PY scripts/deploy_eval.py --model none --tag $tag --method qat_${m}_merged_human \
      --export_dir ${tag}_qat_${m}_merged --sets $SETS --workers 16 --formats ggml_q4_0
done
echo QUEUE_H_DONE
