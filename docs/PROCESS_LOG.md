# Process log: Robust and Compact Recognition of Code-Switched Speech (CS425)

A dated record of everything done to implement the proposal. For each stage it covers what
was built, which decisions were made and why, what broke and how it was fixed, and the
results available at that point. Numbers are % unless stated; CIs are 95% paired bootstrap.
Final numbers live in `results/tables/*.csv` (also copied to the team's `group project/results`).

- Infrastructure: smu2 (8× L40S). **Only GPU 4 was used.** Long jobs ran in tmux queues
  (`scripts/queue_*.sh`).
- Code: `github.com/nibess1/cs425_codeswitch` (main).
- Speed tests: a MacBook Pro M2 Pro.

---

## 13 Oct: setup
- Created `~/cs425-codeswitch-asr` on smu2 and wrote `PLAN.md`, a step-by-step plan for all four workstreams.
- The team Dropbox share of NSC Part 4 is too large for Dropbox's zip download. Wrote
  `scripts/fetch_nsc_dropbox.py`, which lists the folder through Dropbox's web endpoint and
  downloads it file by file, resumably, with size checks.

## 14 Oct: data, shared code, first training

### Decisions confirmed with the team lead (Nibras)
- **Scope:** build all components end to end, so the other members' parts as well, with clear ownership in the README.
- **NSC training data** is capped at about 40 h (see Data). Same Room alone has about 310 h of speech, which is too much for one GPU.
- **Steps that need people** (stress-set recordings, hand checks) get automatic stand-ins, marked
  `PENDING_HUMAN` (see `PENDING_HUMAN.md`).
- **Git:** push to main after each working stage.

### Data
- **NSC Part 4, Mandarin speakers only.** The `cs-chn` files are 2,488 files (about 165 GB), downloaded with 0 failures after a retry pass at lower concurrency (Dropbox throttling).
- **Speaker sheet** (`NSC Part 4 Speaker Metadata.xlsx`): joined on Speaker ID. Session IDs in the sheet aren't zero-padded.
- **TextGrids** are UTF-16 or UTF-8 (detected by BOM). Mandarin is written as `<mandarin>汉字:pinyin</mandarin>`, and only the Han part is kept.
- **Filters** (counts in `data/nsc/pool.parquet`):
  - silence, unknown-word and empty intervals
  - duration outside 0.3–30 s
  - over 25% overlap with the partner's transcribed speech (stand-in for "listen for partner leakage"; drops about 22% of intervals)
  - intervals running past the end of the audio (7 utterances)
- **Splits:**
  - Same Room is split 70/10/20 by session, so partners stay together.
  - Diff Room (Zoom) is test-only.
  - Train is about 40 h: all mixed utterances up to 30 h, then English-only.
  - Val, test and Diff Room test are random 3, 8 and 4 h samples.
- **Audio:** 90 channels were 22.05, 44.1 or 48 kHz and were resampled to 16 kHz.
- **ASCEND:** taken from the Hugging Face `CAiRE/ASCEND` parquet with its official splits. **Finding:** in that release, val and test `original_speaker_id`s don't occur in train. Val and test have only 2–3 speakers each, so ASCEND CIs, which bootstrap over utterances, are optimistic.
- **FLEURS en_us + cmn_hans_cn train:** 5,848 utterances, 17.2 h, used only for the LID head.
- **Size check:** see `results/tables/01_data_counts.csv`. NSC test has 11,042 embedded units; ASCEND test has 772.

### Shared text and scoring code
- **Normalizer** (§3.2): marker stripping, OpenCC t2s, lowercase plus variant map, punctuation removal, Chinese numerals to digits.
- **Units:** one per Han character, Latin word or digit string.
- **Labels:** language/neutral, main language (jieba word counts), E/Mₛ/Mᵢ roles, switch points, and a rule-based critical-word tagger with a term list.
- **Two-pass scorer:**
  - Pass 1: Levenshtein, preferring substitution on ties.
  - Pass 2: error blocks, with insertions charged to the block's reference units.
  - An assertion that Σ role errors = S+D+I holds on every utterance.
- **Translation (language-confusion) test:** CC-CEDICT match or LaBSE ≥ τ (0.70).
- **Other metrics:** sMER, CWER, paired bootstrap, R with the "denominator CI must exclude 0" rule, and Rec.
- **Tests:** 19 pytest cases, including the proposal's worked examples.

### Bugs found and fixed
1. **Normalizer wasn't idempotent.** ASCEND writes `他们both alright的话` without spaces, so the variant map missed `alright`. Numeral runs could also form across removed punctuation (`周四，一份` → 41). The fix:
   - script-change spacing before variant mapping
   - numerals converted only for clear numbers (a magnitude character, a following counter, or 3+ digits) and only after punctuation removal
   - canonical Han spacing

   Every manifest was then re-derived, with an assertion that `label(target) == label(ref)`.
2. **Off-the-shelf HF decoding:** sped up bitsandbytes int8 decoding by raising the batch size to 160, since LLM.int8 has a large fixed cost per call.

### Training
- **Full fine-tuning:** bf16 autocast, the target starts with the main-language token, 128-token cap, 3 epochs, checkpoint chosen on val MER.
- **Token-to-unit roles** come from byte-level BPE byte spans. This is exact even when one Chinese character is split across two tokens.

| Model | Mean val MER | ASCEND test MER | NSC test MER |
|---|---|---|---|
| tiny | 31.1 | 26.1 | 34.5 |
| base | 24.9 | 21.3 | 28.4 |
| small | 18.8 | 14.8 | 21.4 |

Off-the-shelf small, by comparison, scores 26.3 on ASCEND and 55.1 on NSC.

### Forced alignment and LID
- **MMS forced alignment** (torchaudio MMS_FA plus uroman, one aligner word per unit, digits as `*`). It failed on 27 of 64,307 utterances. CTC lets words absorb silence, so labelled spans are capped at a plausible length.
- **LID head:** 2 layers on the frozen fine-tuned base encoder, trained on ASCEND+NSC frames plus 6,000 FLEURS word-piece splices (15.9 h).
- **Bug:** merging every segment under 200 ms also swallowed pauses between words and lowered frame accuracy from 87% to 78%. Short *language* runs are now merged over speech frames only.
- **Result:** 96% (ASCEND) and 95% (NSC) language accuracy on speech frames. Median switch-boundary error is 60 ms and 120 ms.

## 14–15 Oct: methods (tuned on base, selected on val with rules fixed beforehand)

### Role-weighted loss, A1 grid α ∈ {0.5, 1, 2} × β ∈ {0, 0.5}
- **Rule:** lowest embedded error among runs with val MER ≤ plain MER + 1 point.
- **Chosen: α=0.5, β=0.5.** Val MER 30.4 vs 31.4 plain; embedded 33.5 vs 36.1.
- The trainer's own 600-utterance val check had favoured α=2, but on the full val sets α=2 is worse.

### Semantic minimum-risk loss, A2 grid λ ∈ {0.1, 0.3} × ρ₀ ∈ {0.2, 0.5}, on top of α=β=0.5
- **Rule:** lowest val sMER among runs whose CWER doesn't rise.
- **Chosen: λ=0.1, ρ₀=0.5.** All four runs are close: val MER 29.4–29.6.
- **Deviation:** candidates come from the current model on a sub-batch every 4 steps, not once per epoch.
- **Candidate set:** beam-4 outputs under zh, en and free prefixes, plus dictionary-swap candidates.

### Test results on base (vs plain fine-tuning, points; * = CI excludes 0)

| System | NSC MER | NSC embedded | Diff Room MER | Diff Room embedded |
|---|---|---|---|---|
| Role-weighted + semantic (full) | −2.3* | −3.6* | −1.3* | −1.4* |
| Role-weighted only | −0.6 | −2.7* | −0.6 | −0.9* |
| Semantic only | −1.3* | −1.7 | −1.1* | −0.4 |

- On ASCEND every system improves, but the CIs include 0 (2 test speakers, and a few utterances with long insertion runs).
- On small, the best single method (role-weighted) is significantly better on ASCEND MER (−0.5*).
- The full system on small reached 18.2% val MER (plain: 18.8%).

### N-best re-ranking (method d): explanation for the report
**What it is.** Whisper normally outputs one transcript, its single best guess. Re-ranking first builds a shortlist of up to 12 candidates per clip: Whisper decodes three times (forced Mandarin `<|zh|>`, forced English `<|en|>`, and its own language choice), keeping its top 4 each time. It then scores every candidate with three signals and keeps the best:
1. **Whisper's own confidence:** log-probability per token.
2. **Acoustic match:** how well a separate speech model (MMS CTC) matches the romanized candidate to the audio. Audio of "assignment" fits "assignment" better than "zuo ye".
3. **Language agreement:** whether each word's language matches what the LID head hears in that stretch of audio.

The weights are tuned on validation. There are two variants: *with MMS* (an upper bound, since it adds a 300M-parameter model) and *on-device* (no MMS; word timings come from Whisper's cross-attention). Critical terms are flagged when the top candidates disagree on them.

**Why it was expected to help.** The correct embedded word is often present in some candidate even when Whisper's top guess drops or transliterates it.

**What happened: it hurt** (base, test MER vs greedy plain fine-tuning).

| Variant | ASCEND | NSC |
|---|---|---|
| Whisper score only | +16.8* | +32.5* |
| With MMS + LID | +0.2 | +2.6* |
| On-device | +10.0* | +8.9* |

**Why it hurt.** The proposal's score divides log-probability by the number of tokens. A repetition loop ("就只要就只要就只要…") is very predictable token by token, so it scores highly and wins. Forcing `<|zh|>` or `<|en|>` also produces confident wrong-language candidates.

**Loop filter (post-hoc check, tuned on val only, labelled `*_filt`).** Candidates whose gzip compression ratio is above 2.4 (Whisper's own repetition heuristic) are dropped before re-ranking. This recovers most of the damage: val NSC goes from 61.0 to 32.2 for the score-only pick, and ASCEND val reaches 23.1, better than greedy's 25.7. On test, though, all filtered variants are still significantly worse than greedy on NSC (+2.2 to +4.9 points) and neutral on ASCEND.

**Conclusion.** Re-ranking as specified doesn't help; greedy decoding with the model's own language choice is better. This is reported as a negative result. The team lead then chose to skip further re-ranking runs. The full-small re-ranking had already finished, so its results are reported too.

## 15 Oct: quantization (RQ3)
- **Simulated (GPU, bitsandbytes):** int8 (LLM.int8) and NF4, linear layers only; proj_out, convolutions, embeddings and layer norms stay 16-bit. Measured memory footprints are in `10_model_size_and_memory.csv`.
- **Deployed (CPU):** CTranslate2 int8 and whisper.cpp Q8_0/Q5_0/Q4_0, on a fixed seeded subset (all ASCEND test, 4,000 NSC test and 2,000 Diff Room utterances).
- **Finding / fix:** off-the-shelf HF greedy fp16 decoding loops far more than the CPU decoders: 8.2k vs 3.8k inserted units on the same 4,000 NSC utterances. That made CPU-quantized models look up to 20 points *better*. Deployed rows are therefore compared with a 16-bit run of **the same backend** (CTranslate2 float32, whisper.cpp F16), so only the weights differ.
- **Fine-tuned models barely change at 8-bit.** For most slices the matrix-error change is within noise, so R isn't reported (the proposal's own rule). At NF4, small's R estimates are 0.4–0.5 with CIs spanning 1, so there's no evidence that embedded words suffer more.
- **Sensitivity scan (A6)** on fine-tuned small, NF4: the most sensitive blocks are encoder layer 0 and decoder layer 11. Keeping those 2 blocks at 16-bit gives val MER 26.9 against 27.4 all-NF4 and 25.9 at fp16. The full-small scan has also finished.
- **QAT (QLoRA, rank 16, NF4 base):** fine-tuned small is done.
  - The full-small QAT ran out of memory (NF4 plus semantic-loss candidates) and was rerun at batch 8 × 4 accumulation.
  - Merged checkpoints lacked tokenizer files, so whisper.cpp conversion failed; `merge_adapter` now saves them.

## RQ1 findings
- **Same word, embedded vs matrix** (`11_*.csv`). Off-the-shelf small: English words +15 points worse when embedded [12, 19], Chinese characters +26 points. Fine-tuned small: English +7 [5, 10], Chinese +0.3 (not significant). Fine-tuning closes the gap for embedded Mandarin but not for embedded English.
- **Errors are rarely translations.** Only 0.4–3% of embedded units end up in a translation block. Off-the-shelf Whisper instead *transliterates* embedded English phonetically (deadline→得来, python→派森, hong kong→红矿). LaBSE separates the two cleanly: translations score 0.86–0.95, transliterations 0.26–0.49. τ stays PENDING_HUMAN.
- **Logistic regression (A5)** with a speaker-cluster bootstrap: see `12_rq1_logistic_regression.csv`.

## Speed (MacBook M2 Pro, 4 threads, 100 clips, 259 s of audio)
- Off-the-shelf models are measured. Examples: tiny CTranslate2 int8 RTF 0.14 / 376 MB; base whisper.cpp Q8_0 RTF 0.25 / 194 MB; small CTranslate2 int8 RTF 0.98 / 848 MB.
- One run (small Q5_0) was invalidated because the laptop slept; it will be re-run under `caffeinate`.
- Fine-tuned models: pending.

## Open items
See `PENDING_HUMAN.md`:
- stress-set recordings
- hand checks of critical words, translation blocks, alignments and partner leakage
- the IMDA release number
