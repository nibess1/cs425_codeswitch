# Steps that need people (automatic stand-ins are in place until these are done)

Each item says what the pipeline currently uses instead, what to do, and how to re-run.

| # | Proposal step | Current stand-in | What to do | Re-run |
|---|---|---|---|---|
| 1 | Critical-term stress set, recorded by all four members (§3.1) | Sentences written (`stress/sentences.tsv`: 60 Chinese-main, 60 English-main, 187 auto-tagged critical words); **no audio**, so no stress-set rows in the results | Each member reads every line on a phone and saves `data/stress/audio/<name>/<id>.wav` (e.g. `data/stress/audio/phoebe/zh001.wav`) | `python scripts/prepare_stress.py`, then `scripts/decode.py ... --sets stress_test` for the systems you want |
| 2 | Hand-checked critical-word labels on ASCEND / NSC / stress test (§3.3) | Rule + term-list tagger (`csasr/text/critical.py`, `critical_terms.txt`) | Run `python scripts/export_for_review.py`; two members mark `review/critical_words.csv` and fix the term list | CWER is recomputed by `scripts/make_results.py` |
| 3 | τ for LaBSE translation detection, set on 100 hand-labelled blocks (§4.2) | τ = 0.70 (`CSASR_TAU`), and every CC-CEDICT match counts as a translation | Mark `review/translation_blocks.csv` (100 blocks with LaBSE similarity); choose τ with the best F1 | `CSASR_TAU=<τ> python scripts/make_results.py` (bump `STATS_VERSION`) |
| 4 | Listen to 20 NSC files for untranscribed partner speech (§3.1) | Utterances overlapping the partner's transcribed speech by > 25% are dropped (`partner_overlap` in `data/nsc/pool.parquet`; ~22% of intervals) | Listen to the 20 clips in `review/partner_leak.csv` | `scripts/prepare_nsc.py --...` (threshold `PARTNER_OVERLAP_MAX`) |
| 5 | Two members hand-check 30 forced-aligned utterances; report boundary error (§3.3) | MMS_FA alignments used as is (unit spans capped at a plausible length) | Fill the true times in `review/alignments.csv` (30 utterances) using Praat / Audacity | If the error is large, report LID results on the checked clips only |
| 6 | Report the NSC VERSION number (§3.1) | `data/nsc_part4/VERSION.txt` records the Dropbox manifest hash and download date | Confirm which IMDA release the Dropbox copy is | edit `VERSION.txt` |

## Other deviations from the proposal (decided while running)
- **NSC training data** is capped at 40 h (all-mixed first, up to 30 h, then English-only), because Same Room alone has ~310 h; val / test are random 3 h / 8 h samples of their speakers, Diff Room test is a 4 h sample.
- **ASCEND speaker overlap**: in the Hugging Face release the val/test `original_speaker_id`s do not appear in train (the proposal expected all 26 speakers in every split). Val and test have only 2–3 speakers each, so ASCEND CIs (bootstrap over utterances) are optimistic.
- **Semantic MRT candidates** are generated from the current model on a sub-batch every 4 steps instead of once per epoch.
- **Deployed (CPU) rows** are computed on a fixed seeded subset (all of ASCEND test, 4,000 NSC test, 2,000 Diff Room utterances); R compares the same utterances.
- **whisper.cpp** has no restricted language detection, so it runs with `-l auto` (all languages); CTranslate2 and the HF path restrict the language token to {zh, en}.
- **Decoding** is greedy for all main rows (re-ranking uses beam 4).
- **LID head** is trained once, on the plain fine-tuned base encoder, and used as a separate module for every re-ranked system.
