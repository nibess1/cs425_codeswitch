# Robust and Compact Recognition of Code-Switched Speech (CS425)

Whisper on mixed English–Mandarin speech: role-level error analysis, role-weighted and
semantic minimum-risk training, language-aware re-ranking, and how quantization changes
errors on embedded words. See `PLAN.md` for the full plan and status, and
`CS425 Proposal (Revised)` for the method.

## Setup (smu2)
```bash
micromamba create -n cs425 -c conda-forge python=3.11 pip ffmpeg
micromamba activate cs425
pip install -r requirements.txt   # pinned
pip install -e .
export HF_HOME=~/hf-cache
pytest                            # normalizer + scorer tests
```
Use **one GPU** (`CUDA_VISIBLE_DEVICES=<free id>`), and run long jobs inside `tmux`.

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
- **CC-CEDICT**: `data/lex/cedict_ts.u8` from mdbg.net.
