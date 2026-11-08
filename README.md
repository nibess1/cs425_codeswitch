# Robust and Compact Recognition of Code-Switched Speech (CS425)

Whisper on mixed English–Mandarin speech: role-level error analysis, role-weighted and
semantic minimum-risk training, language-aware re-ranking, and how quantization changes
errors on embedded words. See `PLAN.md` for the full plan and status, and
`CS425 Proposal (Revised)` for the method.

## Setup (Linux or macOS, Python 3.11)
```bash
python3.11 -m venv .venv && source .venv/bin/activate   # or: micromamba create -n cs425 -c conda-forge python=3.11 pip ffmpeg
pip install -r requirements.txt   # direct dependencies, pinned; requirements-freeze.txt = full environment we ran in
pip install -e .
pytest                            # normalizer + scorer tests
```
Scoring and tables run on CPU. Training and 8/4-bit decoding need one CUDA GPU (`CUDA_VISIBLE_DEVICES=<free id>`);
`ffmpeg` is needed only to convert the phone recordings.

## Layout
| Path | What | Owner |
|---|---|---|
| `csasr/text/` | shared normalizer, unit splitter, language/role/critical labels | Rui Cong |
| `csasr/score/` | two-pass scorer (MER, role errors with block charging), confusion detection, sMER, CWER, bootstrap, R, Rec | Rui Cong |
| `csasr/data/` | ASCEND, NSC Part 4, FLEURS, stress-set loaders and manifests | Rui Cong / Nibras |
| `csasr/train/` | full fine-tuning, role-weighted and semantic MRT losses | Phoebe |
| `csasr/lid/`, `csasr/decode/` | forced alignment, language-ID head, N-best re-ranking, flags, segmented decoding | Bridgette |
| `csasr/quant/` | bitsandbytes simulation, CTranslate2 / whisper.cpp exports, sensitivity scan, QLoRA QAT | Nibras |
| `scripts/` | command-line entry points | |
| `results/` | tables and figures (every number comes from `scripts/score.py`) | |

Not in git: `data/`, `ckpt/`, `preds/`, `runs/`.

## Reproduce (pre-processing → training → testing → scoring)
Run from the repository root with the `cs425` environment active; pick an idle GPU with
`CUDA_VISIBLE_DEVICES=<id>`. The `scripts/queue_*.sh` files chain these steps exactly as we ran them.
Hyper-parameters are in `configs/*.yaml` (CLI flags override them) and in Table 2 of the report.

```bash
# 1. Pre-processing: 16 kHz mono FLAC clips + JSONL manifests in data/<corpus>/manifests/
huggingface-cli download CAiRE/ASCEND --repo-type dataset --local-dir data/ascend/hf
python scripts/prepare_ascend.py                        # official splits
python scripts/fetch_nsc_dropbox.py fetch data/nsc_part4 -j 2 --only cs-chn && python scripts/prepare_nsc.py
python scripts/prepare_fleurs.py                        # spliced clips for the language-ID head
python scripts/prepare_stress.py                        # human stress set: data/stress/audio/<reader>/<id>.wav
python scripts/align.py ascend_train nsc_train ascend_val nsc_val   # MMS word times

# 2. Training (checkpoints in ckpt/<run>/best)
python scripts/train.py configs/ft_base.yaml            # plain fine-tuning; also ft_tiny.yaml, ft_small.yaml
python scripts/train.py configs/ft_base.yaml --run full_base --alpha 0.5 --beta 0.5 --lambda_sem 0.1 --rho0 0.5
python scripts/train_lid.py --encoder ckpt/ft_base/best --run lid_ft_base
python scripts/train.py configs/ft_base.yaml --run qat_full_base --model ckpt/full_base/best \
       --lora true --lr 1e-4 --epochs 1 --warmup 100

# 3. Testing (predictions in preds/<model>__<precision>__<method>.jsonl)
python scripts/decode.py --model ckpt/full_base/best --model_tag full_base --precision nf4_bnb --method full
python scripts/deploy_eval.py --model ckpt/full_base/best --tag full_base --method full \
       --formats ct2_f32 ct2_int8 ggml_f16 ggml_q8_0 ggml_q4_0
python scripts/sensitivity.py --model ckpt/full_base/best --tag full_base       # mixed-precision blocks
python scripts/rerank.py --model ckpt/full_base/best --tag full_base --lid ckpt/lid_ft_base/head.pt --loop_filter

# 4. Scoring and tables (results/tables/*.csv, results/figures/)
python scripts/score.py full_base__fp16__full full_base__nf4_bnb__full      # every metric, 95% CIs
python scripts/score.py --fragility full_base__fp16__full full_base__nf4_bnb__full --set nsc_test --main en
python scripts/make_results.py && python scripts/export_tables.py && python scripts/make_figures.py
```

**Human stress recordings (tables 27–28, Nov 2026).** `GPU=<id> bash scripts/queue_h.sh` decodes every
system and format on set `stress_test` (method suffix `_human`); then `python scripts/make_human_tables.py`,
`python scripts/make_human_figure.py` and `python scripts/make_examples.py`. The recordings themselves are not
in this repository.

**Human label audits (table 20c).** Put the filled audits in `review/*_human.csv` and the answer keys in
`review/keys/` (both gitignored), then `python scripts/score_audits.py`.

**Number check.** `python scripts/check_paper_numbers.py <paper_dir> results/tables` lists every decimal
number in the report that is not in the tables its section draws on (expected misses: hyper-parameters,
thresholds, derived differences). In the results-heavy sections most random numbers also match something,
so it catches mistakes mainly in the smaller sections.

## Prediction format
Every decoder writes JSONL, one line per utterance:
```json
{"utt_id": "...", "corpus": "ascend|nsc_same|nsc_diff|stress", "split": "test", "model": "base",
 "precision": "fp16|int8_bnb|nf4_bnb|ct2_int8|ggml_q8_0|...", "method": "...", "ref": "...", "hyp": "..."}
```
`python scripts/score.py preds/<files>.jsonl` gives every metric with 95% paired-bootstrap CIs.

## Data
- **NSC Part 4**: `python scripts/fetch_nsc_dropbox.py list data/nsc_part4`, then `... fetch data/nsc_part4 -j 2 --only cs-chn`.
  Speaker sheet: `data/nsc_part4/speaker_metadata.xlsx` (it isn't in git; get it from the team drive).
- **ASCEND**: `CAiRE/ASCEND` on the Hugging Face Hub. **FLEURS**: `google/fleurs` (en_us, cmn_hans_cn).
- **Included small data** (`data/lex`, `data/llm_labels`, `data/leak`): CC-CEDICT (`data/lex/cedict_ts.u8`, CC-BY-SA 4.0,
  MDBG, from https://www.mdbg.net/chinese/dictionary?page=cc-cedict; see `licences/CC-CEDICT.txt`), the cached LaBSE
  similarities, the D1/D2 label files and the D3 leakage signals. Scoring needs them; the code stops with an error if
  CC-CEDICT is missing.
