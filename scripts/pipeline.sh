#!/usr/bin/env bash
# Every experiment in the report, in the order we ran it, with the settings each run finally used.
# Replaces the 26 run-by-run queue_*.sh files (crash reruns and recovery steps included), which are
# kept at git tag paper-final; docs/PROCESS_LOG.md maps each old queue to a stage below.
#
#   GPU=<idle id> bash scripts/pipeline.sh <stage> [<stage> ...]
#   DRY=1 bash scripts/pipeline.sh all          # print every command without running anything
#
# Stages (in order): data align baselines finetune quant lid rolew semantic transfer rerank repairs
#   deploy ablation translit stress_synth stress_human analysis tables;  "all" runs them in order.
# Steps whose output already exists are skipped (unless DRY=1), so a stage can be rerun after a crash.
# Selections on validation (role weights, semantic loss, transliteration gamma/lambda) are made by
# scripts/select_run.py and written to results/chosen*.json before any test decode that depends on them.
set -u
cd "$(dirname "$0")/.."
PY=${PY:-python}
DRY=${DRY:-0}
W=${W:-16}                          # CPU workers for the deployed formats
export CUDA_VISIBLE_DEVICES=${GPU:-0}

run() { echo "+ $(printf '%q ' "$@")"; [ "$DRY" = 1 ] || "$@" || { echo "FAILED: $*"; exit 1; }; }
cpu() { run env CUDA_VISIBLE_DEVICES= "$@"; }
missing() { [ "$DRY" = 1 ] || ! [ -e "$1" ]; }                       # output file not there yet
untrained() { [ "$DRY" = 1 ] || ! grep -qs '"event": "done"' "runs/$1/log.jsonl"; }
get() { $PY -c "import json;print(json.load(open('results/$1.json'))['$2'])"; }

gpu_idle() {  # stop unless the GPU has no processes and < 2 GB in use
  [ "$DRY" = 1 ] && return
  local used procs
  used=$(nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-gpu=memory.used --format=csv,noheader,nounits)
  procs=$(nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-compute-apps=pid --format=csv,noheader | grep -c .)
  echo "[$(date +%T)] GPU $CUDA_VISIBLE_DEVICES: ${used} MiB used, $procs processes"
  [ "$used" -lt 2048 ] && [ "$procs" -eq 0 ] || { echo "GPU $CUDA_VISIBLE_DEVICES is busy: stopping"; exit 1; }
}

train() {  # train <config> <run> [flags...]
  local cfg=$1 r=$2; shift 2
  untrained "$r" && run $PY scripts/train.py "configs/$cfg.yaml" --run "$r" "$@"
  return 0
}

decode() {  # decode <checkpoint|size> <tag> <precision> <method> [flags...]
  local ck=$1 tag=$2 p=$3 m=$4; shift 4
  missing "preds/${tag}__${p}__${m}.meta.json" && \
    run $PY scripts/decode.py --model "$ck" --model_tag "$tag" --precision "$p" --lang free --method "$m" --batch 48 "$@"
  return 0
}

evaluate() {  # evaluate <checkpoint> <tag> <method> [precisions]: test sets at each precision + fp16 on val
  local ck=$1 tag=$2 m=$3 p
  for p in ${4:-fp16 int8_bnb nf4_bnb}; do decode "$ck" "$tag" "$p" "$m"; done
  decode "$ck" "$tag" fp16 "${m}_val" --sets ascend_val nsc_val
}

settings() {  # role weights and semantic loss chosen on validation (results/chosen.json)
  A=$(get chosen alpha); B=$(get chosen beta); L=$(get chosen lambda_sem); RHO=$(get chosen rho0)
  FULL=$(get chosen full_base); RW=$(get chosen rw)
}

stage_data() {
  run huggingface-cli download CAiRE/ASCEND --repo-type dataset --local-dir data/ascend/hf
  run $PY scripts/prepare_ascend.py
  run $PY scripts/fetch_nsc_dropbox.py fetch data/nsc_part4 -j 2 --only cs-chn
  run $PY scripts/prepare_nsc.py
  run $PY scripts/prepare_diffroom_ablation.py      # section E split
  run $PY scripts/prepare_stress.py                 # the team's recordings (data/stress/audio/<reader>/<id>.wav)
  run $PY scripts/data_counts.py
}

stage_align() { gpu_idle; run $PY scripts/align.py ascend_val nsc_val ascend_train nsc_train ascend_test nsc_test nsc_diff_test; }

stage_baselines() {  # off-the-shelf Whisper: fp16 with each language prompt, 8/4-bit with the free token
  gpu_idle
  local m l
  for m in tiny base small; do
    for l in free zh en both; do
      missing "preds/${m}__fp16__offshelf_$l.meta.json" && \
        run $PY scripts/decode.py --model $m --precision fp16 --lang $l --method offshelf_$l --batch 48
    done
    for p in int8_bnb nf4_bnb; do decode $m $m $p offshelf_free; done
  done
}

stage_finetune() { gpu_idle; for m in base small tiny; do train ft_$m ft_$m; done; }

stage_quant() { gpu_idle; for m in base small tiny; do evaluate ckpt/ft_$m/best ft_$m ft; done; }

stage_lid() {  # spliced FLEURS clips -> language-ID head on the plain fine-tuned base encoder
  gpu_idle
  run $PY scripts/prepare_fleurs.py
  missing ckpt/lid_ft_base/head.pt && run $PY scripts/train_lid.py --encoder ckpt/ft_base/best --run lid_ft_base
  return 0
}

stage_rolew() {  # role-weighted loss grid on base, chosen on validation
  gpu_idle
  local a b runs=""
  for a in 0.5 1 2; do for b in 0 0.5; do train ft_base rw_base_a${a}_b${b} --alpha $a --beta $b; runs="$runs rw_base_a${a}_b${b}"; done; done
  run $PY scripts/select_run.py --grid rw --baseline ft_base --runs $runs
}

stage_semantic() {  # semantic MRT grid on top of the chosen role weights; ablation without them
  gpu_idle
  local rw a b l r runs=""
  rw=$(get select_rw chosen)
  a=$($PY -c "import json;print(json.load(open('runs/$rw/config.json'))['alpha'])")
  b=$($PY -c "import json;print(json.load(open('runs/$rw/config.json'))['beta'])")
  for l in 0.1 0.3; do for r in 0.2 0.5; do
    train ft_base sem_base_l${l}_r${r} --alpha $a --beta $b --lambda_sem $l --rho0 $r; runs="$runs sem_base_l${l}_r${r}"
  done; done
  run $PY scripts/select_run.py --grid sem --baseline "$rw" --runs $runs
  [ "$DRY" = 1 ] || $PY - "$rw" "$a" "$b" <<'PY'
import json, sys
rw, a, b = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
full = json.load(open("results/select_sem.json"))["chosen"]
cfg = json.load(open(f"runs/{full}/config.json"))
v = json.load(open("results/select_rw.json"))["val"]
sub = "rw" if v[rw]["cs_E_emb"] <= v.get("abl_base_semonly", {"cs_E_emb": 9})["cs_E_emb"] else "sem"
json.dump(dict(rw=rw, full_base=full, alpha=a, beta=b, lambda_sem=cfg["lambda_sem"], rho0=cfg["rho0"],
               small_subset=sub), open("results/chosen.json", "w"))
PY
  settings
  train ft_base abl_base_semonly --lambda_sem $L --rho0 $RHO
}

stage_transfer() {  # full system on small (MRT sub-batch 4 to fit), best single method on small; all precisions
  gpu_idle
  settings
  train ft_small full_small --alpha $A --beta $B --lambda_sem $L --rho0 $RHO --mrt_batch 4
  if [ "$(get chosen small_subset)" = rw ]; then train ft_small sub_small --alpha $A --beta $B
  else train ft_small sub_small --lambda_sem $L --rho0 $RHO; fi
  evaluate ckpt/$RW/best rw_base rw
  evaluate ckpt/$FULL/best full_base full
  evaluate ckpt/abl_base_semonly/best semonly_base semonly
  evaluate ckpt/full_small/best full_small full
  evaluate ckpt/sub_small/best sub_small sub
}

stage_rerank() {  # N-best re-ranking (shared head, then each model's own head), loop filter, LID gate
  gpu_idle
  settings
  local spec ck tag lid
  for spec in ft_base:ft_base:lid_ft_base $FULL:full_base:lid_ft_base full_small:full_small:lid_ft_base; do
    IFS=: read -r ck tag lid <<< "$spec"
    run $PY scripts/rerank.py --model ckpt/$ck/best --tag $tag --lid ckpt/$lid/head.pt
    run $PY scripts/rerank.py --model ckpt/$ck/best --tag $tag --lid ckpt/$lid/head.pt --loop_filter
  done
  missing ckpt/lid_full_base/head.pt && run $PY scripts/train_lid.py --encoder ckpt/$FULL/best --run lid_full_base
  missing ckpt/lid_full_small/head.pt && run $PY scripts/train_lid.py --encoder ckpt/full_small/best --run lid_full_small
  for spec in $FULL:fullown_base:lid_full_base full_small:fullown_small:lid_full_small; do
    IFS=: read -r ck tag lid <<< "$spec"
    run $PY scripts/rerank.py --model ckpt/$ck/best --tag $tag --lid ckpt/$lid/head.pt
    run $PY scripts/rerank.py --model ckpt/$ck/best --tag $tag --lid ckpt/$lid/head.pt --loop_filter
  done
  for spec in full_base:full_base:lid_ft_base full_small:full_small:lid_ft_base \
              fullown_base:full_base:lid_full_base fullown_small:full_small:lid_full_small; do
    IFS=: read -r tag greedy lid <<< "$spec"
    run $PY scripts/rerank.py --gate --tag $tag --greedy ${greedy}__fp16__full --lid ckpt/$lid/head.pt --out_tag $tag
  done
}

stage_repairs() {  # sensitivity scan + mixed precision (A6/B1), QLoRA QAT on NF4 (B3), merged QAT models
  gpu_idle
  settings
  local spec ck tag src r
  for spec in ft_small:ft_small full_small:full_small ft_tiny:ft_tiny ft_base:ft_base $FULL:full_base; do
    run $PY scripts/sensitivity.py --model ckpt/${spec%%:*}/best --tag ${spec##*:}
  done
  local QAT="--lora true --lr 1e-4 --epochs 1 --warmup 100"
  train ft_small qat_ft_small --model ckpt/ft_small/best $QAT
  train ft_small qat_full_small --model ckpt/full_small/best $QAT --alpha $A --beta $B --lambda_sem $L --rho0 $RHO \
        --batch 8 --grad_accum 4 --mrt_batch 2
  train ft_tiny qat_ft_tiny --model ckpt/ft_tiny/best $QAT
  train ft_base qat_ft_base --model ckpt/ft_base/best $QAT
  train ft_base qat_full_base --model ckpt/$FULL/best $QAT --alpha $A --beta $B --lambda_sem $L --rho0 $RHO \
        --batch 16 --grad_accum 2 --mrt_batch 4
  for spec in ft_small:ft_small:ft full_small:full_small:full ft_tiny:ft_tiny:ft ft_base:ft_base:ft $FULL:full_base:full; do
    IFS=: read -r ck tag src <<< "$spec"
    r=qat_${tag}
    decode ckpt/$ck/best $tag nf4_qlora qat_$src --adapter ckpt/$r/best
    missing ckpt/${r}_merged/vocab.json && \
      run $PY -c "from csasr.quant.qat import merge_adapter; merge_adapter('ckpt/$ck/best','ckpt/$r/best','ckpt/${r}_merged')"
    evaluate ckpt/${r}_merged $tag qat_${src}_merged nf4_bnb
  done
}

stage_deploy() {  # deployed CPU formats (CTranslate2, whisper.cpp) incl. same-backend 16-bit references
  settings
  local spec model tag m F="ct2_f32 ct2_int8 ggml_f16 ggml_q8_0 ggml_q5_0 ggml_q4_0"
  for spec in tiny:tiny:offshelf_free base:base:offshelf_free small:small:offshelf_free \
              ckpt/ft_base/best:ft_base:ft ckpt/ft_small/best:ft_small:ft ckpt/ft_tiny/best:ft_tiny:ft \
              ckpt/full_small/best:full_small:full ckpt/$FULL/best:full_base:full; do
    IFS=: read -r model tag m <<< "$spec"
    cpu $PY scripts/deploy_eval.py --model $model --tag $tag --method $m --workers $W --formats $F
  done
  for spec in ft_small:ft:"ct2_int8 ggml_q8_0 ggml_q5_0 ggml_q4_0" full_small:full:"ct2_int8 ggml_q8_0 ggml_q5_0 ggml_q4_0" \
              ft_tiny:ft:"ct2_int8 ggml_q4_0" ft_base:ft:"ct2_int8 ggml_q4_0" full_base:full:"ct2_int8 ggml_q4_0"; do
    IFS=: read -r tag m F <<< "$spec"
    cpu $PY scripts/deploy_eval.py --model ckpt/qat_${tag}_merged --tag $tag --method qat_${m}_merged --workers $W --formats $F
  done
}

stage_ablation() {  # section E: recording condition (A = ft_base trained on Same Room)
  gpu_idle
  local VS='[ascend_val, nsc_val, nsce_diff_val]' spec
  train ft_base ediff_base --train_sets '[ascend_train, nsce_diff_train]' --val_sets "$VS"
  train ft_base emix_base --train_sets '[ascend_train, nsce_mix_train]' --val_sets "$VS"
  for spec in ft_base:esame_base ediff_base:ediff_base emix_base:emix_base; do
    decode ckpt/${spec%%:*}/best ${spec##*:} fp16 eabl --sets ascend_test nsc_test nsce_diff_test
  done
}

stage_translit() {  # third run (gamma) and fourth run (lambda) of the transliteration-penalised MRT
  gpu_idle
  local g r l FT="--alpha 0.5 --beta 0.5 --rho0 0.5 --save_epochs true"
  for g in 0 0.5 1; do
    train ft_base fulltl_base_g$g $FT --lambda_sem 0.1 --translit_gamma $g
    decode ckpt/fulltl_base_g$g/best fulltl_base_g$g fp16 fulltl_val --sets ascend_val nsc_val
  done
  run $PY scripts/select_run.py --grid tl --baseline full_base__fp16__full_val --runs fulltl_base_g0.5 fulltl_base_g1 \
      --control fulltl_base_g0
  g=$( [ "$DRY" = 1 ] || $PY -c "import json;g=json.load(open('results/chosen_tl.json'))['gamma'];print('' if g is None else '%g'%g)")
  if [ -n "$g" ]; then  # small only when a gamma qualified (it did not: negative result)
    train ft_small fulltl_small_g$g $FT --lambda_sem 0.1 --translit_gamma $g --mrt_batch 4
    decode ckpt/fulltl_small_g$g/best fulltl_small_g$g fp16 fulltl_val --sets ascend_val nsc_val
    decode ckpt/fulltl_small_g$g/best fulltl_small_g$g fp16 fulltl
  fi
  for g in 0 0.5 1; do decode ckpt/fulltl_base_g$g/best fulltl_base_g$g fp16 fulltl; done
  run $PY scripts/make_tl_tables.py
  for r in l0.4_g0 l1_g0 l0.4_g3 l1_g3; do
    l=${r%%_g*}
    train ft_base fulltl_base_$r $FT --lambda_sem ${l#l} --translit_gamma ${r##*_g}
    decode ckpt/fulltl_base_$r/best fulltl_base_$r fp16 fulltl_val --sets ascend_val nsc_val
  done
  run $PY scripts/select_run.py --grid tl --tag lam --cwer_margin 1.0 --baseline full_base__fp16__full_val \
      --runs fulltl_base_l0.4_g3 fulltl_base_l1_g3 --control fulltl_base_l0.4_g0 fulltl_base_l1_g0
  local P=""
  for r in l0.4_g0 l0.4_g3 l1_g0 l1_g3; do
    decode ckpt/fulltl_base_$r/best fulltl_base_$r fp16 fulltl
    P="$P full_base__fp16__full:fulltl_base_${r}__fp16__fulltl"
  done
  P="$P fulltl_base_l0.4_g0__fp16__fulltl:fulltl_base_l0.4_g3__fp16__fulltl fulltl_base_l1_g0__fp16__fulltl:fulltl_base_l1_g3__fp16__fulltl"
  run $PY scripts/make_tl_tables.py --pairs $P --names 26_lambda_retune,26b_lambda_retune_val,26c_lambda_retune_absolute \
      --val_csv results/lam_val.csv
  run $PY scripts/loss_scale_probe.py --model ckpt/$(get chosen full_base)/best --batches 40
}

stress_decode() {  # stress_decode <sets> <suffix>: the stress-set decode list (GPU, then CPU formats)
  local sets=$1 sfx=$2 spec ck tag m p
  decode small small fp16 offshelf_free_$sfx --sets $sets
  for spec in ckpt/ft_tiny/best:ft_tiny:ft ckpt/ft_base/best:ft_base:ft ckpt/ft_small/best:ft_small:ft \
              ckpt/$FULL/best:full_base:full ckpt/full_small/best:full_small:full; do
    IFS=: read -r ck tag m <<< "$spec"
    for p in fp16 int8_bnb nf4_bnb; do decode $ck $tag $p ${m}_$sfx --sets $sets; done
  done
  for spec in ft_tiny:ft ft_base:ft ft_small:ft full_base:full full_small:full; do
    IFS=: read -r tag m <<< "$spec"
    cpu $PY scripts/deploy_eval.py --model none --tag $tag --method ${m}_$sfx --export_dir ${tag}_${m} \
        --sets $sets --workers $W --formats ct2_f32 ct2_int8 ggml_f16 ggml_q8_0 ggml_q4_0
  done
}

stage_stress_synth() {  # D5: CosyVoice 2 voices (own env), phone-channel copy, then the decode list
  gpu_idle
  settings
  run ${COSY_PY:-~/micromamba/envs/cosyvoice/bin/python} scripts/tts_stress.py
  run $PY scripts/prepare_stress.py --source tts
  stress_decode "stress_tts_clean stress_tts_phone" stress
}

stage_stress_human() {  # the team's 120 phone recordings, plus the NF4 repairs of table 14
  gpu_idle
  settings
  stress_decode stress_test human
  decode ckpt/ft_tiny/best ft_tiny nf4_bnb mp2_human --sets stress_test --skip_modules model.encoder.layers.0 model.decoder.layers.3
  decode ckpt/$FULL/best full_base nf4_bnb mp3_human --sets stress_test \
         --skip_modules model.encoder.layers.0 model.decoder.layers.5 model.encoder.layers.3
  decode ckpt/ft_tiny/best ft_tiny nf4_qlora qat_ft_human --sets stress_test --adapter ckpt/qat_ft_tiny/best
  decode ckpt/$FULL/best full_base nf4_qlora qat_full_human --sets stress_test --adapter ckpt/qat_full_base/best
  local spec
  for spec in ft_tiny:ft full_base:full; do
    cpu $PY scripts/deploy_eval.py --model none --tag ${spec%%:*} --method qat_${spec##*:}_merged_human \
        --export_dir ${spec%%:*}_qat_${spec##*:}_merged --sets stress_test --workers $W --formats ggml_q4_0
  done
}

stage_analysis() {  # RQ1 regression, corpus statistics, alignment checks, partner-leakage labels
  gpu_idle
  run $PY scripts/rq1.py small__fp16__offshelf_free base__fp16__offshelf_free ft_base__fp16__ft ft_small__fp16__ft
  run $PY scripts/switch_stats.py
  run $PY scripts/align_checks.py
  run $PY scripts/leakage.py signals
  run $PY scripts/leakage.py s5
  run $PY scripts/leakage.py rule
  run $PY scripts/leakage.py final
  run $PY scripts/leakage.py sensitivity
  # CPU speed was measured on a MacBook: python scripts/speed_bench.py --clips speed/clips --models speed/models --out results/speed.json
}

stage_tables() {
  local t
  for t in make_results export_tables make_tl_tables make_human_tables; do run $PY scripts/$t.py; done
  run $PY scripts/make_examples.py
  run $PY scripts/make_figures.py
  [ -d review/keys ] && run $PY scripts/score_audits.py   # needs the filled audits + keys (not in git)
  return 0
}

ALL="data align baselines finetune quant lid rolew semantic transfer rerank repairs deploy ablation translit
     stress_synth stress_human analysis tables"
[ $# -ge 1 ] || { sed -n '2,14p' "$0"; exit 1; }
for s in "$@"; do
  [ "$s" = all ] && set -- $ALL
done
for s in "$@"; do
  declare -F "stage_$s" > /dev/null || { echo "unknown stage: $s"; exit 1; }
  echo "=== stage $s ($(date +'%d %b %H:%M'))"
  "stage_$s"
done
