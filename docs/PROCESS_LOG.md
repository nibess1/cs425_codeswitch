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

**First reading (15 Oct, superseded).** "Re-ranking as specified doesn't help." The team lead
then chose to skip further re-ranking; the full-small run had already finished, so all three
re-ranked systems are reported.

**Corrected reading (17 Oct).** The table above compares re-ranked systems with *plain*
fine-tuning, which mixes the training effect with the re-ranking effect. Compared with each
system's **own greedy output** (filtered variants; points, * = CI excludes 0):

| System | ASCEND MER | ASCEND embedded | NSC MER | Diff Room MER |
|---|---|---|---|---|
| plain FT base | −1.0 / −0.9 | −14.8 / −14.5 (CI incl. 0) | +2.4* / +2.3* | +4.9* / +4.7* |
| full base | **−0.9* / −0.7*** | **−4.2* / −4.3*** | +0.1 / −0.1 | +0.7* / +0.5* |
| full small | −0.3 / −0.4 | −1.9 / **−3.0*** | +0.8* / +0.8* | +1.4* / +1.2* |

(Each cell gives the score-only filtered and on-device filtered variants.) The unfiltered
with-MMS variant is +0.3* to +1.9* MER on all sets for the full systems.

**Conclusion for the report.**
- Re-ranking **helps embedded English inside Chinese-main speech** (ASCEND): embedded-word error falls by 3–4 points for the full systems, and MER falls slightly for full base.
- It **slightly hurts English-main speech** (NSC): MER rises by 0–1.4 points.
- Without the loop filter it is harmful, because length-normalised scores favour repetition loops.
- **On-device cost is high.** On the MacBook (CTranslate2 int8, fine-tuned base, 4 threads), plain decoding runs at RTF 0.29, adding the LID head gives 0.40, and on-device re-ranking gives 2.30, roughly 8× slower and slower than real time.

## 15 Oct: quantization (RQ3)
- **Simulated (GPU, bitsandbytes):** int8 (LLM.int8) and NF4, linear layers only; proj_out, convolutions, embeddings and layer norms stay 16-bit. Measured memory footprints are in `10_model_size_and_memory.csv`.
- **Deployed (CPU):** CTranslate2 int8 and whisper.cpp Q8_0/Q5_0/Q4_0, on a fixed seeded subset (all ASCEND test, 4,000 NSC test and 2,000 Diff Room utterances).
- **Finding / fix:** off-the-shelf HF greedy fp16 decoding loops far more than the CPU decoders: 8.2k vs 3.8k inserted units on the same 4,000 NSC utterances. That made CPU-quantized models look up to 20 points *better*. Deployed rows are therefore compared with a 16-bit run of **the same backend** (CTranslate2 float32, whisper.cpp F16), so only the weights differ.
- **8-bit is safe** in every backend (LLM.int8, CTranslate2 int8, whisper.cpp Q8_0) and every size: embedded and matrix error change by at most about ±2 points, so R isn't reportable (the matrix change is within noise, as the proposal anticipated).
- **4-bit hurts embedded units more in smaller models.** R is significantly above 1 in several slices (`06_fragility_ratio_R.csv`):

  | Model | Format | Slice | R |
  |---|---|---|---|
  | fine-tuned tiny | NF4 | Chinese-main ASCEND | 3.47 [1.85, 6.13] |
  | fine-tuned base | whisper.cpp Q4_0 | Chinese-main ASCEND | 4.4 [1.8, 12.2] |
  | full base | NF4 | English-main NSC | 1.75 [1.05, 3.03] |
  | full base | NF4 | NSC Diff Room | 1.64 [0.74, 3.29] |
  | fine-tuned tiny | whisper.cpp Q4_0 | English-main NSC | 1.4 [1.01, 1.86] |

  So both directions are affected: embedded English (ASCEND) and embedded Mandarin (NSC). This contradicts the off-the-shelf pilot (R ≈ 0.6) and supports the proposal's expectation that R > 1 for fine-tuned models.
- **Small is robust even at 4-bit.** Changes are about 1 point and R is about 1 or not reportable.
- **The fixes can't be judged on small.** Mixed precision and QAT were run on small, where the NF4 loss is already within noise, so the recovery rate Rec is undefined (`quant.md`). They should be re-run on base or tiny, where 4-bit loss is significant (future work).
- **Sensitivity scan (A6)** on fine-tuned small, NF4: the most sensitive blocks are encoder layer 0 and decoder layer 11. Keeping those 2 blocks at 16-bit gives val MER 26.9 against 27.4 all-NF4 and 25.9 at fp16. The full-small scan has also finished.
- **QAT (QLoRA, rank 16, NF4 base):** done for fine-tuned small and full small (val MER 19.3 for full small with adapters, against 18.2 at 16-bit).
  - The full-small QAT ran out of memory (NF4 plus semantic-loss candidates) and was rerun at batch 8 × 4 accumulation.
  - Merged checkpoints lacked tokenizer files, so whisper.cpp conversion failed; `merge_adapter` now saves them.

## RQ1 findings
- **Same word, embedded vs matrix** (`11_*.csv`). Off-the-shelf small: English words +15 points worse when embedded [12, 19], Chinese characters +26 points. Fine-tuned small: English +7 [5, 10], Chinese +0.3 (not significant). Fine-tuning closes the gap for embedded Mandarin but not for embedded English.
- **Errors are rarely translations.** Only 0.4–5.5% of embedded units end up in a translation block. Off-the-shelf Whisper instead *transliterates* embedded English phonetically (deadline→得来, python→派森, hong kong→红矿). LaBSE separates the two cleanly: translations score 0.86–0.95, transliterations 0.26–0.49. τ stays PENDING_HUMAN.
- **Logistic regression (A5)** with a speaker-cluster bootstrap: see `12_rq1_logistic_regression.csv`.

## Speed (MacBook M2 Pro, 4 threads, 100 clips, 259 s of audio)
- Off-the-shelf models are measured. Examples: tiny CTranslate2 int8 RTF 0.14 / 376 MB; base whisper.cpp Q8_0 RTF 0.25 / 194 MB; small CTranslate2 int8 RTF 0.98 / 848 MB.
- One run (small Q5_0) was invalidated because the laptop slept; it will be re-run under `caffeinate`.
- Fine-tuned models were measured on 17 Oct under `caffeinate`. The LID head adds about 40% to the decoding time (an extra encoder pass), and on-device re-ranking is about 8× slower. See `13_speed_macbook.csv`.

## Open items
See `PENDING_HUMAN.md`:
- stress-set recordings
- hand checks of critical words, translation blocks, alignments and partner leakage
- the IMDA release number

---

## 20 Oct: second run (final-report brief). Rules fixed BEFORE looking at any new test result
Compute for this run: **GPU 5** (quantization fixes, B), **GPU 7** (LID heads, TTS stress set,
Diff Room ablation: C, D, E), smu2 CPUs (exports, signal processing), MacBook (speed only).

### A1/A2 findings (data integrity, pool-level only, no test scores used)
- **Diff Room pool:** after the Same Room filters there are 207,969 utterances, **212.3 h**, 322 sessions and 637 speakers (Same Room: 211,499 / 193.2 h / 298 / 593).
- **Speaker overlap:** 4 speakers appear in both rooms (1782, 1783, 1786, 1787).
  - None of the Diff Room test speakers is in Same Room train or val; their Diff Room utterances were already dropped as `speaker_seen` (1,097 utterances).
  - 1786 and 1787 are in Same Room *test*, which is not training leakage.
  - **The existing Diff Room test is therefore speaker-disjoint from all training; no rebuild needed.**
- **626 vs 120 speakers is not a parsing bug.** The Diff Room test is a random 4 h *utterance* sample over all 322 Diff Room sessions (Diff Room was test-only), so it has many speakers with few utterances each (median 6). Same Room test is an 8 h sample from the 20% of sessions held out (median 73 per speaker). Speaker-cluster CIs (A3) account for this.

### Pre-registered decision rules
- **B1, mixed precision (k):** the same rule as for small. k ∈ {1, 2, ⌊n/8⌋, ⌊n/4⌋} blocks kept at 16-bit, chosen by the lowest pooled val embedded error (cs_E_emb), ties to the smaller k. Then run once on test.
- **B3, QAT:** QLoRA rank 16, 1 epoch, lr 1e-4, the same loss as the model's own training. The checkpoint is the one with the lowest mean val MER. No tuning on test.
- **B4, Rec and R:** reported only where the 16-bit→4-bit embedded change has a 95% CI excluding 0. The slices are the same as in `06_fragility_ratio_R.csv`.
- **C1, LID gate:** a clip is re-ranked only if the smoothed LID head output contains a language run of at least θ ms in the minority language. θ ∈ {0 (any switch), 200, 400, 800, ∞ = never}.
  - θ is chosen by the lowest pooled val MER of the gated system, using the re-ranking weights already chosen on val (filtered variants).
  - Ties go to the larger θ, which means fewer clips re-ranked.
- **C2, per-model LID heads:** the same recipe as `lid_ft_base` (2 epochs, lr 1e-3). Reported on val only, then used in C1 with the C1 rule.
- **C3, transliteration:** the phonetic-similarity threshold is chosen for the best F1 against my D1 labels (transliteration vs not), on the D1 pairs only.
- **D1, τ:** the best F1 of "CC-CEDICT match OR LaBSE ≥ τ" against my labels (translation vs not), over τ in steps of 0.05. A sensitivity table at τ ∈ {0.5, 0.6, 0.7, 0.8} is reported regardless.
- **D2, critical words:** CWER is reported under both the rule tagger and the adjudicated labels. No tuning.
- **D3, leakage:**
  - The combination rule is set on a pilot of NSC **val** clips only. It assigns clean / faint / partner-words-audible from thresholds on the four signals, written down before scoring test.
  - Test utterances are then flagged once.
  - The sensitivity re-score excludes flagged utterances.
- **E, Diff Room ablation:**
  - **Split:** Diff Room split by session 70/10/20 (seed 425). Sessions of the 4 dual-room speakers are excluded from E-train and E-val.
  - **New held-out Diff Room test:** a 4 h random sample from the E-test sessions (the existing Diff Room test spans all sessions and is kept as is for the main tables).
  - **Training:** each condition uses the `ft_base` recipe with no tuning. A = existing `ft_base`; B = 40 h Diff Room (all-mixed first, up to 30 h, then English-only, as for Same Room); C = 20 h Same Room (random half of the A training set) + 20 h Diff Room (random half of B).
  - **Evaluation:** once on the Same Room test and once on the E Diff Room test.

### 20 Oct: results of the first CPU stages
- **A3 (speaker-cluster bootstrap).** `csasr/score/bootstrap.speaker_arrays()` sums each speaker's counts, so resampling rows equals resampling whole speakers (every metric is a ratio of sums). A test checks that the speaker-level and utterance-level point estimates are identical. `report.table(..., speaker_ci=True)` and `report.compare(..., speaker_ci=True)` add `*_spk_lo/hi` for the NSC sets. ASCEND stays utterance-level, with CIs labelled optimistic.
- **A4 (version and licence).** The Dropbox share has only the 4 audio and script folders, with **no VERSION, UPDATE HISTORY or LICENCE file**, so the licence terms couldn't be confirmed (`data/nsc_part4/VERSION.txt`). Following the brief, section D labelling uses **ASCEND and the stress set only**. NSC is processed only automatically (signals and rules); I don't read or judge NSC transcripts. Logged in PENDING_HUMAN.
- **C4 (switching statistics)**, from `results/switch_stats.csv`. NSC train was selected code-switched-first, so the fair comparison is against the random Diff Room sample:

  | | ASCEND | NSC Diff Room |
  |---|---|---|
  | code-switched utterances | 26.8% | 29.1% |
  | switches per code-switched utterance | 2.12 | 2.85 |
  | median switches | 2 | 2 |
  | embedded span (units) | 2.0 | 2.8 |
  | main language | 69% Chinese-main | 80% English-main |

  NSC isn't more code-switched per utterance, but its switches are more frequent within an utterance and its embedded spans are longer.
- **E data fix.** The first split by session failed its own speaker-disjointness assertion, because one Diff Room speaker recorded 2 sessions. The split now uses connected speaker–session groups (union-find); the manifests from the failed run were deleted before use.
- **D1 (translation blocks).**
  - Items: every error block, across all systems, on ASCEND test and val, where an embedded run was replaced by other-script text. That gives 10,166 occurrences, deduplicated to **1,768 unique (reference, output) pairs** (`data/llm_labels/d1_pairs.csv`).
  - Labels (annotator 1, me), using `prompts/translation_rubric.md`: transliteration 1,157, other 498, translation 113 (`data/llm_labels/d1_labels_annotator1.csv`).
  - A second, independent subagent annotator, which sees only the rubric and the items, is running.
  - **τ by the pre-registered rule** (best unique-pair F1 of "CC-CEDICT or LaBSE ≥ τ" vs *translation*): **τ = 0.65** (P 0.61, R 0.71, F1 0.65). Occurrence-weighted F1 would favour 0.70 (0.88); the rule says unique pairs, so 0.65 is used. A sensitivity table at 0.5/0.6/0.7/0.8 will be reported.
- **C3 (transliteration detector, `csasr/score/translit.py`).** The Han side is romanized with pypinyin, both sides are encoded with Metaphone, and similarity = 1 − normalised Levenshtein. The threshold is the best F1 vs *transliteration* on the D1 pairs: **θ = 0.1** (P 0.75, R 0.91, F1 0.82; flat at 0.80–0.82 up to 0.25). The new field `N_tl_E` and metric `translit_rate` count embedded units in a cross-script block that isn't a translation and has similarity ≥ θ. `STATS_VERSION` = 6.

### 23 Oct: D1 finished (two annotators, adjudicated)
- **Annotator 2.** A fresh subagent that saw only `prompts/translation_rubric.md` and the 1,768 items, never annotator 1's labels. It labelled transliteration 945, other 713, translation 110 (`data/llm_labels/d1_labels_annotator2.csv`).
- **Agreement (Cohen's κ, annotator 1 vs 2):**

  | Comparison | Raw agreement | κ |
  |---|---|---|
  | 3-class | 0.798 | **0.62** |
  | translation vs rest | 0.981 | **0.84** |
  | transliteration vs rest | 0.813 | 0.62 |
  | other vs rest | 0.800 | 0.56 |

  Most disagreements (269 of 358) were pairs annotator 1 called transliteration and annotator 2 called other.
- **Adjudication** (annotator 1, with both reasons visible). Rules applied to all 358 disagreements:
  - **R1:** transliteration needs the sound-alike rendering in the *other* script relative to the reference. English sound-alike garble of an English word (complain→company) is ordinary misrecognition, so *other*. English sound-alikes of a Chinese reference (我们不能→"one more banana") *are* transliterations.
  - **R2:** a single-syllable partial match isn't enough.
  - **R3:** a multi-word reference counts as transliteration if at least half its syllables are rendered by sound.
  - **R4:** a filler or interjection equivalent counts as translation even with extra words.

  Result: transliteration 963, other 689, translation 116 (`d1_labels_adjudicated.csv`). The adjudication sided with annotator 2 in 252 cases, annotator 1 in 104, and neither in 2.
- **Thresholds re-selected on the adjudicated labels with the same rules:** τ = **0.65** (F1 0.66, P 0.62, R 0.71) and θ = **0.1** (F1 0.74, P 0.62, R 0.90), both unchanged.
- **Audit:** `review/translation_audit.csv` (30 random pairs).

### D2 in progress
- **Candidates:** 335 unique words from ASCEND test + the stress sentences, combining the rule tagger with a broad finder (digits/numerals, Latin words outside the top-5k English list, fuzzy term matches, jieba nr/ns/nt/nz).
- **Annotator 1:** critical 222, not critical 113.
- **Annotator 1 vs the rule tagger:** agreement 0.773, κ **0.55**. The rule tagger flags 10 words annotator 1 rejects (function words via substring matches, `块`, `report`). It misses 66 that annotator 1 accepts: almost all names, places, brands, exams and course terms (深圳, toefl, mcgrady, alipay, semester…), because it has no named-entity component.
- **Finder recall:** reading 500 random ASCEND test utterances in full found **34 critical words the finder missed**. The main gaps are `ai` (×8), `hong kong` written as two words (×4), product names (`mac`, `iphone`, `windows`), school years (`大二`, `大四`) and durations (`一个月`, `one year`).
- **Annotator 2:** a fresh subagent with only the rubric and items is running.

### 23 Oct: D2 finished (two annotators, adjudicated)
- **Annotator 2.** A fresh subagent that saw only `prompts/critical_rubric.md` and the 335 items, without annotator 1's labels or the rule-tagger column. It labelled critical 232, not critical 103.
- **Agreement:**

  | Comparison | Raw agreement | κ |
  |---|---|---|
  | annotator 1 vs 2 | 0.899 | **0.77** |
  | annotator 1 vs rule tagger | 0.773 | 0.55 |
  | annotator 2 vs rule tagger | 0.731 | 0.47 |
- **Adjudication** of the 34 disagreements (sided with annotator 2 in 27, annotator 1 in 7):
  - *Critical:* names, places, brands and their fragments (博, 中国式家长, 回大陆, 福建人), language names (a fact about the conversation), degree levels, technical terms, time parts (9 点, p m), and the World Cup.
  - *Not critical:* `有 1 点` ("a bit"), generic roles (tutor, supervisor), and everyday objects and actions (parcel, email, booking, slides, bug, 门卡, 面试).

  Result: critical 229, not critical 106. Adding the 20 extra types found by reading 500 utterances gives a **critical-word list of 249 types** (`data/llm_labels/d2_critical_words.txt`).
- **New CWER mode (b).** `CSASR_CRITICAL_LABELS=<list>` makes the tagger mark adjudicated word types (English words, multi-word Latin types, jieba words over Chinese runs) plus every digit string. Stats are cached separately (`*_critD2.pkl`).
- **CWER on ASCEND test, (a) rule tagger vs (b) adjudicated labels:**

  | System | (a) 81 critical words | (b) 298 critical words |
  |---|---|---|
  | off-the-shelf small | 35.8 | 36.2 |
  | ft tiny | 33.3 | 45.0 |
  | ft base | 24.7 | 38.9 |
  | ft small | 19.8 | 29.9 |
  | full base | 21.0 | 36.6 |
  | full small | 17.3 | 27.5 |
  | ft small NF4 | 12.3 | 26.5 |
  | ft tiny NF4 | 46.9 | 53.0 |

  The rule tagger finds only about a quarter of the critical words, mostly numbers and dates, which fine-tuning handles well. With names, places, brands and course terms included, **fine-tuning's critical-word gain shrinks to about a third** (small: −6.3 points rather than −16.0). Both CWERs will be reported.
- **Audit:** `review/critical_audit.csv` (50 random disagreements with rule-tagger, annotator and adjudicated labels).
- **NSC transcripts** were not hand-labelled (A4), so NSC CWER stays rule-tagger only.

### 23 Oct: D3 partner leakage (NSC licence confirmed by the team lead, so NSC transcripts may be read)
- **Environment incident.** `pip install speechbrain` upgraded torch to 2.14 (too new for the GPU driver) and pulled in CUDA 13 libraries. Fixed by force-reinstalling torch/torchaudio 2.5.1+cu124 with their CUDA 12 dependencies and using speechbrain 1.0.3; `pip check` is clean and the 21 tests pass.
- **Signals** (`scripts/leak_signals.py`, `leak_s5.py`; 16,328 clips: NSC val as pilot, Same Room test, Diff Room test):
  - **s1:** overlap with the partner's transcribed speech.
  - **s2:** envelope correlation with the partner's channel. *Not directional:* in Same Room both phones record whoever is talking (median r = 0.74 vs 0.01 in Diff Room), so it measures acoustic coupling, not leakage, and isn't used in the rule.
  - **s3:** ECAPA (spkrec-ecapa-voxceleb), the max over 1.5 s windows of cos(partner) − cos(self), against enrollment from up to 8 clean utterances per channel (816 channels).
  - **s4:** inserted units matching the partner's ±2 s transcript.
  - **s5** (added after the pilot showed that leaked speech is often *substituted*, not inserted): output units absent from the reference but present in the partner's ±2 s transcript, excluding common words.
- **Pilot** (val only). I judged the 103 val clips with s5 ≥ 1 or s3 > 0.2 against the partner transcript: 30 partner-words-audible, 33 faint, 40 clean. Rule fixed from the pilot, before test:
  - **partner_words_audible** if (s5 ≥ 2 and s3 > −0.15) or (s5 ≥ 1 and s3 > 0.1) or (s3 > 0.2 and ref_recall < 0.3)
  - **faint** if otherwise s3 > 0.2
  - **clean** otherwise

  The rule agrees with my pilot judgement on 82.5% of those hard cases.
- **Test.** The rule is applied once. The 158 borderline test clips were judged by me from text and numbers only (`data/llm_labels/d3_test_judgements.csv`); there the rule agreed with my judgement 63% of the time, mostly "rule faint, judged clean" where the output matches the reference. Final labels:

  | Test set | Clean | Partner words audible | Faint | Flagged |
  |---|---|---|---|---|
  | Same Room | 98.25% | 0.88% | 0.87% | **1.75%** |
  | Diff Room | 99.85% | 0.07% | 0.07% | 0.15% |

  The Diff Room rate is consistent with separate rooms.
- **Sensitivity** (flagged clips excluded). Same Room MER falls by about 1 point for fine-tuned models: ft_small 21.37 → 20.32, full_small 20.65 → 19.71, ft_base 28.39 → 27.25, full_base 26.07 → 25.17. The off-the-shelf small falls 55.07 → 54.40, and Diff Room changes by ≤ 0.3. Embedded-unit error changes by ≤ 1.3 points. **No conclusion changes.**
- **Audit:** `review/leak_audit.csv` (10 flagged + 10 cleared clips, with audio paths and signals).
- **Single annotator.** I made all D3 judgements. There was no second D3 annotator, because judging a clip needs the partner transcript and the signal values, so the labels are single-annotator; the audit above is the human check.
