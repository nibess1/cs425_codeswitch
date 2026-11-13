# Data sample (about 19 MB)

A fixed random subset of the two corpora, so the code can be tested without downloading them:
300 utterances, 1-10 s each, mixed and monolingual in each split's own proportion (seed 0,
`scripts/make_sample.py`). Same FLAC files and manifest format as `data/`.

| Set | Utterances | Use |
|---|---|---|
| `ascend_test`, `nsc_test`, `nsc_diff_test` | 60 each | decoding and scoring |
| `ascend_train`, `nsc_train` | 40 each | a few training steps (GPU) |
| `ascend_val`, `nsc_val` | 20 each | validation during training |

Set `CSASR_SAMPLE=1` and every `data/...` path resolves under `sample/`, so the normal scripts run
unchanged (from the repo root, after `pip install -r requirements.txt && pip install -e .`):

```bash
export CSASR_SAMPLE=1
python scripts/decode.py --model tiny --precision fp32 --method sample   # CPU works (49 s on a 256-thread server)
python scripts/score.py tiny__fp32__sample --no_labse                    # MER per set
python scripts/train.py configs/ft_tiny.yaml --run sample_tiny --epochs 1 --eval_every 0   # needs a GPU
```

Numbers on 60 utterances are noisy and are not comparable to the report, which uses the full test sets.
Our own stress-set recordings are not included (shared with course staff only).

## Licences
- **ASCEND** (`data/ascend/`): from `CAiRE/ASCEND` on the Hugging Face Hub (Lovenia et al., 2022),
  CC BY-SA 4.0 (https://creativecommons.org/licenses/by-sa/4.0/). Clips are unchanged apart from
  resampling to 16 kHz FLAC; this subset is shared under the same licence.
- **NSC Part 4** (`data/nsc/`): Contains information from the National Speech Corpus (Part 4) accessed
  on 14 October 2026 from the Infocomm Media Development Authority of Singapore (IMDA), which is made
  available under the terms of the Singapore Open Data Licence version 1.0
  (https://data.gov.sg/open-data-licence). Licence text: `licences/Singapore_Open_Data_Licence_v1.0.txt`.
