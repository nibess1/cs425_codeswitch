#!/usr/bin/env bash
# Queue D5 (GPU 7 + CPU): synthetic stress set (CosyVoice 2) -> phone version -> decode with
# off-the-shelf small, fine-tuned tiny/base/small, full base/small, 8/4-bit simulated and deployed.
set -u
cd "$(dirname "$0")/.."
PY=${PY:-~/micromamba/envs/cs425/bin/python}
until grep -q COSY_ENV_DONE /tmp/mk_cosy.log 2>/dev/null; do grep -qiE "^ERROR|Traceback" /tmp/mk_cosy.log 2>/dev/null && { echo "cosyvoice env failed"; exit 1; }; sleep 60; done
~/micromamba/envs/cosyvoice/bin/python scripts/tts_stress.py || { echo "TTS failed"; exit 1; }
[ $(find data/stress/tts/clean -name "*.wav" | wc -l) -ge 400 ] || { echo "too few TTS clips"; exit 1; }
$PY scripts/prepare_stress_tts.py
FULL=$($PY -c "import json;print(json.load(open('results/chosen.json'))['full_base'])")
SETS="stress_tts_clean stress_tts_phone"
dec() { [ -f preds/$2__$3__$4.meta.json ] || $PY scripts/decode.py --model $1 --model_tag $2 --precision $3 --lang free \
          --method $4 --sets $SETS --batch 48; }
dec small small fp16 offshelf_free_stress
for spec in ckpt/ft_tiny/best:ft_tiny:ft ckpt/ft_base/best:ft_base:ft ckpt/ft_small/best:ft_small:ft \
            ckpt/$FULL/best:full_base:full ckpt/full_small/best:full_small:full; do
  IFS=: read ck tag m <<< "$spec"
  for p in fp16 int8_bnb nf4_bnb; do dec $ck $tag $p ${m}_stress; done
done
# deployed formats on CPU (reuse the existing exports)
for spec in ft_tiny:ft ft_base:ft ft_small:ft full_base:full full_small:full; do
  IFS=: read tag m <<< "$spec"
  CUDA_VISIBLE_DEVICES= $PY scripts/deploy_eval.py --model none --tag $tag --method ${m}_stress --export_dir ${tag}_${m} \
      --sets $SETS --workers 16 --formats ct2_f32 ct2_int8 ggml_f16 ggml_q8_0 ggml_q4_0
done
echo QUEUE_D5_DONE
