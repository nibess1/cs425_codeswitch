# Results report: Robust and Compact Recognition of Code-Switched Speech

CS425 group project (Rui Cong, Phoebe, Nibras, Bridgette). This report covers every experiment in the revised proposal (13 Oct 2026) plus the second run (20–25 Oct: data integrity, quantization fixes, method follow-ups, automated labelling, Diff Room ablation).

Compute:
- First run (14–17 Oct): one NVIDIA L40S GPU on smu2 (GPU 4).
- Second run: GPUs 5 and 7 on smu2.
- CPU evaluation: smu2's CPUs.
- Speed: a MacBook Pro (Apple M2 Pro).

Results were frozen on 25 Oct (tag `results-final`).

- **Data behind every number:** `results/tables/*.csv` (column definitions in `results/tables/README.txt`).
- **Figures:** `results/figures/fig1–7` (PNG + SVG, each with the CSV it plots).
- **Slide numbers:** `results/KEY_NUMBERS.md`.
- **How each step was done, every decision rule (written before test results were looked at), and every bug:** `docs/PROCESS_LOG.md`.
- **What still needs people:** `PENDING_HUMAN.md`.

**Conventions**
- All numbers are on **test** sets unless stated.
- A unit is one Chinese character, English word or digit string.
- **MER** = (S+D+I)/N.
- Role errors (embedded E, matrix-at-switch Mₛ, other matrix Mᵢ) come from Pass 2, which charges each error block's insertions to its reference units, so they can exceed 100%.
- Brackets are 95% bootstrap CIs. They are paired for differences, and * means the CI of a difference excludes 0.
- NSC claims also have **speaker-cluster CIs** (resampling whole speakers; table 16), which are wider and are the ones to quote.
- ASCEND CIs resample utterances and are optimistic, because ASCEND test has only 2 speakers.
- Decoding is greedy with a 128-token cap unless stated.

---

## 1. Data
| Set | Utterances | Hours | Code-switched | Embedded units | Speakers |
|---|---|---|---|---|---|
| ASCEND train / val / test | 9,869 / 1,130 / 1,315 | 8.8 / 0.9 / 0.9 | 28% / 18% / 28% | 7,081 / 471 / 772 | 18 / 3 / 2 |
| NSC Part 4 Same Room train (Mandarin speakers) | 35,647 | 40.0 | 62% | 105,493 | 412 |
| NSC Same Room val / test | 3,367 / 8,876 | 3.0 / 8.0 | 25% / 29% | 4,099 / 11,042 | 60 / 120 |
| NSC Different Room test (Zoom) | 4,085 | 4.0 | 29% | 5,648 | 626 |
| Stress set, **SYNTHETIC** (TTS, 4 voices × 120 sentences, clean + phone channel) | 480 + 480 | 0.6 + 0.6 | 100% | 908 | 4 voices |

**NSC sampling**
- NSC train is capped at 40 h, out of about 193 h of usable Same Room speech: all code-switched utterances up to 30 h, then English-only ones.
- NSC val and test are random samples within a session-level speaker split.
- The Diff Room pool is 212 h, 322 sessions and 637 speakers.
- **Integrity checks (A1/A2):**
  - No Diff Room test speaker appears in any training split. Four speakers appear in both rooms; their Diff Room utterances are excluded.
  - The Diff Room test has many speakers with few utterances each (626 speakers, median 6 utterances), because it is a random utterance sample over all sessions. This is not a parsing bug.

**Switching statistics (C4, table 22)**

| | Code-switched utterances | Switches per code-switched utterance | Mean embedded span |
|---|---|---|---|
| ASCEND (spontaneous) | 26.8% | 2.12 | 2.0 units |
| NSC Diff Room (random sample, asked to switch) | 29.1% | 2.85 | 2.8 units |

NSC Same Room train is 53.5% code-switched only because it was selected mixed-first. NSC speakers switch about as often per utterance as ASCEND speakers, but more times within a code-switched utterance and in longer spans. This is the "deliberate switching" caveat in §8.

**Licence.** NSC is used under the Singapore Open Data Licence v1.0, confirmed by the team. The attribution sentence must be copied from the licence page (`PENDING_HUMAN.md` item 2). The Dropbox share has no VERSION file; `data/nsc_part4/VERSION.txt` records the manifest hash and download date.

## 2. Baselines and fine-tuning (table 02, figure 1)
| System | ASCEND MER | ASCEND E | NSC Same MER | NSC Same E | NSC Diff MER | NSC Diff E |
|---|---|---|---|---|---|---|
| small, off-the-shelf (model picks language) | 26.3 [24.4, 28.1] | 56.8 | 55.1 [53.5, 56.7] | 76.7 | 48.8 [46.4, 51.3] | 72.7 |
| tiny, fine-tuned | 26.1 [24.6, 27.5] | 53.2 | 34.5 [33.3, 35.8] | 36.2 | 26.8 [25.4, 28.4] | 29.8 |
| base, fine-tuned | 21.3 [19.3, 23.6] | 55.3 | 28.4 [27.2, 29.8] | 31.0 | 22.4 [21.0, 23.9] | 24.0 |
| small, fine-tuned | 14.8 [13.7, 15.8] | 30.6 | 21.4 [20.6, 22.2] | 23.8 | 19.0 [17.6, 20.6] | 22.9 |
| **base, full system** | 20.3 [19.0, 21.4] | 39.2 | 26.1 [25.5, 26.6] | 27.4 | 21.1 [20.2, 22.0] | 22.5 |
| **small, full system** | **14.6** [13.6, 15.7] | **29.8** | **20.6** [20.1, 21.2] | **21.3** | **17.1** [16.2, 18.1] | **18.7** |

Speaker-cluster CIs for NSC (table 16) are wider. For example, full small is 20.6 [18.6, 23.4] on NSC Same Room and 17.1 [15.7, 18.6] on Diff Room.

- **Fine-tuning on ASCEND + NSC makes the biggest difference**, most of all on NSC: 55.1 → 21.4 for small.
- **Forcing the wrong language token is catastrophic.** Forced `<|zh|>` on NSC gives 104.5% MER because of insertions. Prompting with both tokens doesn't beat letting the model choose (`02_*.csv`, `offshelf_*` rows).

## 3. RQ1: where does Whisper fail, and is it role or language?
1. **Embedded units have the highest error** in almost every system and corpus (E > Mₛ > Mᵢ; figure 2). Off-the-shelf small: ASCEND E 56.8 vs Mᵢ 23.4; NSC E 76.7 vs Mᵢ 48.6.
2. **Same word, embedded vs matrix** (table 11): the same word type has a higher error when embedded than when part of the main language.

   | System | English words | Chinese characters |
   |---|---|---|
   | Off-the-shelf small | +15.4 pts [12.1, 18.6] | +26.0 pts [23.9, 28.0] |
   | Fine-tuned small | +7.4 [5.2, 10.1] | +0.3 (n.s.) |

3. **Logistic regression (A5, table 12)** controls for language, training frequency, spoken length and corpus, with speaker-bootstrap CIs.
   - Off-the-shelf: the embedded coefficient is b₁ = 0.94 [0.78, 1.15] (small) and 1.18 [1.02, 1.37] (base) log-odds, a role effect over and above language.
   - After fine-tuning, b₁ for embedded *Mandarin* is ≈ 0 (small −0.07 [−0.22, 0.17]), but the embedded-*English* interaction stays large (+0.79 [0.42, 1.11]).
   - **Fine-tuning removes the role penalty for embedded Mandarin but not for embedded English.**
4. **Errors are rarely translations; they are transliterations** (C3, D1).
   - After the D1 labelling, translations account for only 0.5–1.9% of embedded units in fine-tuned systems (3.4–4.0% off-the-shelf).
   - Phonetic *transliterations* into the other script (deadline→得来, python→派森, tutor→跳特) account for 2.3–10.9%. ASCEND: fine-tuned tiny 10.9%, ft small 4.4%, full small 3.8%, off-the-shelf small 8.8%. NSC: off-the-shelf small 10.7%, full small 2.5%.
   - **The proposal's hypothesis that many embedded errors are translations is not supported.** Transliteration is about 3–5× more common.
   - Detectors:
     - Translation (CC-CEDICT, or LaBSE ≥ τ = 0.65 chosen on the D1 labels): pair-level F1 0.66, occurrence-weighted F1 0.87.
     - Transliteration (pypinyin + Metaphone similarity ≥ θ = 0.1): F1 0.74.
     - Changing τ between 0.6 and 0.7 changes the translation rate by about ±20% (table 23); the conclusion holds for every τ ≤ 0.7.

## 4. RQ2: do the proposed methods reduce these errors at 16-bit?
Methods were tuned on base, with hyperparameters chosen on validation by rules fixed in advance (table 04):
- role weights **α = 0.5, β = 0.5**
- semantic loss **λ = 0.1, ρ₀ = 0.5**

Paired differences vs plain fine-tuning of the same size (table 03, figure 3), in MER and embedded error E (points; negative = better):

| System | ASCEND MER / E | NSC Same MER / E | NSC Diff MER / E |
|---|---|---|---|
| base + role-weighted | −0.7 / −14.6 | −0.6 / **−2.7*** | −0.6 / **−0.9*** |
| base + semantic only | −1.2 / −14.6 | **−1.3*** / −1.7 | **−1.1*** / −0.4 |
| **base full (role-weighted + semantic)** | −1.0 / −16.1 | **−2.3*** / **−3.6*** | **−1.3*** / **−1.4*** |
| small + role-weighted (best single method) | **−0.5*** / −2.3 | +0.1 / −1.1 | −0.9 / −3.3 |
| **small full** | −0.1 / −0.8 | **−0.7*** / **−2.4*** | **−1.9*** / **−4.2*** |

- **The role-weighted loss lowers embedded-unit error, and combining it with the semantic loss works best.** The full system is significantly better on both NSC test sets for base and small, with no rise in critical-word error.
- **ASCEND changes are mostly within noise.** Its test set is small (2 speakers) and dominated by a few utterances with long insertion runs; that is why the ASCEND embedded-error CIs for base span about 45 points.
- **Semantic MER ≈ MER**, because so few errors are translations (§3.4). The semantic loss therefore acts mostly as a sequence-level minimum-risk loss.

### N-best re-ranking (method d)
**What it is.** Re-ranking builds a shortlist of up to 12 transcripts per clip: beam 4 under forced `<|zh|>`, forced `<|en|>` and the model's own language choice. It then picks one by combining:
- Whisper's per-token log-probability
- an MMS CTC acoustic-match score
- agreement between each word's language and the LID head

The weights are tuned on validation. The *on-device* variant drops MMS and takes word timings from Whisper's cross-attention.

**Always re-ranking (loop-filtered) vs each system's own greedy output**, MER / embedded error E (points). The first value in each cell is on-device, the second is with MMS:

| System | ASCEND MER | ASCEND E | NSC Same MER | NSC Diff MER |
|---|---|---|---|---|
| plain FT base | −0.9 / +0.2 (n.s.) | −14.5 / −16.4 (n.s.) | **+2.3* / +2.2*** | **+4.7* / +4.9*** |
| full base | **−0.7*** / **+1.0*** | **−4.3* / −5.1*** | −0.1 / +0.3 (n.s.) | **+0.5* / +1.0*** |
| full small | −0.4 / **+1.1*** | **−3.0* / −4.7*** | **+0.8* / +1.1*** | **+1.2* / +1.5*** |

- **Re-ranking consistently lowers embedded-unit error in Chinese-main speech** (ASCEND E −3 to −5 points for the full systems).
- **It raises MER on English-main NSC speech.**
- **The MMS variant is not an upper bound in practice.** Its acoustic score favours candidates that over-insert, so its MER is worse than on-device even though its embedded error is lower.
- **Unfiltered, it is harmful.** The proposal's length-normalised score lets repetition loops win: val NSC MER was 61% vs 27% greedy for plain base. A post-hoc loop filter (drop candidates with gzip compression ratio > 2.4, Whisper's own heuristic, tuned on val only) removes most of that.
- **Critical-term flags** (raised when the top-3 candidates disagree on a critical word) are a weak safety net. Full small with MMS: precision 55% / recall 13% on NSC test, 39% / 33% on ASCEND (`methods.md`).

### LID-gated re-ranking (C1) and per-model LID heads (C2) (table 15, figure 7)
**How the gate works.** A clip is re-ranked only when the LID head finds a minority-language run of at least θ ms; otherwise the greedy output is kept. The θ rule was fixed before testing: θ ∈ {0, 200, 400, 800, ∞}, lowest val MER, ties to the larger θ.

- **Gate setting:** θ = 200 ms was chosen for every model and head. The gate re-ranks 25% of ASCEND test, 28% of NSC Same Room and 26% of NSC Diff Room clips.

**Change vs each system's own greedy output** (on-device variant; MMS variant in table 15):

| System | ASCEND MER / E | NSC Same MER / E | NSC Diff MER / E |
|---|---|---|---|
| full base, shared head | **−0.7 [−1.0, −0.3]*** / **−5.4 [−8.6, −2.3]*** | **−0.4 [−0.5, −0.2]*** / −0.4 (n.s.) | **−0.3 [−0.6, −0.2]*** / −0.7 (n.s.) |
| full base, own head | **−0.7*** / **−5.0*** | **−0.4*** / −0.4 | **−0.3*** / −0.6 |
| full small, shared head | **−0.5 [−0.8, −0.2]*** / **−3.9 [−6.9, −1.2]*** | −0.1 (n.s.) / +0.2 | +0.1 (n.s.) / −0.1 |
| full small, own head | **−0.5*** / −2.0 (n.s.) | −0.2 (n.s.) / +0.1 | 0.0 / 0.0 |

- **Gating removes the harm on English-main speech.** Always-on re-ranking made NSC worse (full base Diff Room +0.5*, full small +0.8 to +1.5*). Gated re-ranking is better or neutral on every test set, and for full base it is significantly better on all three.
- **The ASCEND embedded-error gain survives** (−4 to −7 points for base, MMS variant −6.7*).
- **Own vs shared LID head (C2):** the model's own head gives the same accuracy. Its benefit is speed, because it reuses the encoder pass.
- **Estimated cost, full base on the MacBook.** RTF ≈ 0.30 (ASR + LID) + share × 1.47 (re-ranking overhead) ≈ **0.66–0.70**, vs 0.21 greedy and 1.76 always-on. Re-ranking speed for small was not measured.
- **Bug note:** the first gated run (20 Oct) had an ASCEND val/test id collision. These are the corrected `*_gatedv2` runs (PROCESS_LOG, 25 Oct).

### Language-ID head (method c, C2; table 09)
| Head (frozen encoder) | Speech-frame language accuracy (ASCEND / NSC val) | Median switch-boundary error | Switches detected |
|---|---|---|---|
| shared (fine-tuned base encoder) | 96.0% / 95.4% | 60 / 120 ms | 75% / 85% |
| own head, full base | 95.9% / 95.3% | 60 / 100 ms | 77% / 84% |
| own head, full small | **97.7% / 96.4%** | **20 / 80 ms** | 78% / 85% |

The model's own head reuses that model's encoder pass. The LID head adds about 0.09 RTF on the MacBook (base: 0.21 → 0.30, §7).

## 5. RQ3: does quantization hurt embedded units more, and do the fixes help?
**Fragility.** R = Δembedded / Δmatrix from 16-bit to quantized, on code-switched utterances, compared with the *same backend's* 16-bit run (table 06, figure 4). R is reported only when the matrix change CI excludes 0.
- **8-bit is safe** in every backend (bitsandbytes LLM.int8, CTranslate2 int8, whisper.cpp Q8_0) and every size: embedded and matrix error move by at most about ±2 points, and R is not reportable.
- **4-bit hurts embedded units more for the smaller fine-tuned models** (R > 1, CI excludes 1):

  | Model | Format | Slice (embedded language) | Δ embedded | Δ matrix | R |
  |---|---|---|---|---|---|
  | fine-tuned tiny | NF4 | ASCEND, Chinese-main (English) | +16.2 | +4.7 | **3.47 [1.85, 6.13]** |
  | fine-tuned base | whisper.cpp Q4_0 | ASCEND, Chinese-main (English) | +8.2 | +1.9 | **4.40 [1.78, 12.17]** |
  | full base | whisper.cpp Q4_0 | ASCEND, Chinese-main (English) | +7.1 | +2.3 | **3.07 [1.07, 6.96]** |
  | full base | NF4 | NSC, English-main (Mandarin) | +3.0 | +1.7 | **1.75 [1.05, 3.03]** |
  | fine-tuned tiny | whisper.cpp Q4_0 | NSC, English-main (Mandarin) | +8.9 | +6.4 | **1.40 [1.01, 1.86]** |

  - The off-the-shelf pilot (R ≈ 0.6) doesn't carry over to fine-tuned models, which fits the proposal's hypothesis. Both embedded English and embedded Mandarin are affected.
- **Small is robust at 4-bit:** changes are about 1 point, and R is about 1 or not reportable.
- **Deployed rows are compared with a 16-bit run of the *same* CPU backend.** Off-the-shelf HF fp16 greedy decoding loops far more than the CPU decoders, so comparing across backends would credit quantization with a decoder difference.

**Fixes (B1–B4; table 14, figure 5).**
- Rec = (Eq_emb − Eq,fix_emb) / (Eq_emb − E16_emb): 1 = the fix removes the whole 4-bit loss, 0 = no help.
- It is reported only where the 4-bit embedded loss has a CI excluding 0. Bold = Rec CI excludes 0.
- k (blocks kept at 16-bit) and the QAT checkpoint were chosen on validation.

| Fix | Model | Slice | 4-bit loss (pts) | Rec |
|---|---|---|---|---|
| Mixed precision, 2 blocks at 16-bit (NF4) | ft tiny | ASCEND zh-main | +16.2 | **0.65 [0.29, 0.91]** |
| | ft tiny | NSC en-main | +11.2 | **0.77 [0.40, 1.05]** |
| | ft tiny | NSC Diff en-main | +6.4 | **0.80 [0.28, 1.02]** |
| Mixed precision, 3 blocks (NF4) | full base | NSC en-main | +3.0 | **0.70 [0.38, 1.04]** |
| | full base | NSC Diff en-main | +2.2 | **0.48 [0.03, 0.90]** |
| Mixed precision, 1 block (NF4) | ft base | NSC en-main | +2.5 | **0.86 [0.26, 2.99]** |
| QLoRA QAT (NF4 + 16-bit adapters) | ft tiny | ASCEND zh-main | +16.2 | **0.83 [0.57, 1.26]** |
| | ft tiny | NSC en-main | +11.2 | **0.72 [0.46, 0.98]** |
| | full base | NSC en-main | +3.0 | **0.77 [0.54, 1.10]** |
| | ft base | NSC en-main | +2.5 | −0.30 [−3.64, 2.08] (no help) |
| QAT merged → NF4 | ft tiny | ASCEND zh-main | +16.2 | **0.88 [0.60, 1.30]** |
| QAT merged → **whisper.cpp Q4_0** | ft tiny | NSC en-main | +8.9 | **−0.12 [−0.42, 0.10]** |
| | full base | NSC en-main | +3.3 | 0.06 [−0.38, 0.39] |
| | ft base | ASCEND zh-main | +8.2 | −0.03 [−0.78, 0.43] |

- **Both fixes recover most of the NF4 embedded loss on tiny and full base** (Rec ≈ 0.5–0.9).
- **QAT gains do not transfer to whisper.cpp Q4_0** (Rec ≈ 0 on every slice). The adapters were trained against NF4's quantization grid, while Q4_0 uses a different block-absmax grid, so the deployed format needs its own fix. This is a negative result.
- **Deployed mixed precision (B2) is not possible in whisper.cpp:**
  - A patched quantizer can keep chosen tensors at F16, but the model loader allocates every weight at the file's global type and rejects the file.
  - Mixed precision is therefore reported for NF4 only.
  - The practical answer is Q8_0: it is as fast as Q4_0 on the M2 and causes no embedded-unit loss (§7).
- **Sensitivity scans (A6):** the most sensitive blocks are always early encoder layers (layer 0 first) and the last decoder layer (`07_*.csv`).

## 6. Synthetic stress set (D5; table 17) — SYNTHETIC, not human speech
**How it was made.** 120 code-switched sentences rich in critical terms (names, numbers, course codes) were synthesised by CosyVoice 2-0.5B (Apache-2.0) zero-shot, in 4 voices prompted by FLEURS clips (2 English, 2 Mandarin speakers; CC-BY 4.0). They were also passed through a simulated phone channel: 300–3400 Hz, 8 kHz round trip, synthetic room impulse response, 20 dB noise.

**Results (clean / phone):**

| System (16-bit) | MER | Embedded error | CWER (rule tagger) |
|---|---|---|---|
| full small | 22.3 / 36.8 | 54.6 / 74.5 | 26.7 / 41.2 |
| ft small | 22.3 / 36.6 | 54.6 / 73.9 | 27.0 / 38.8 |
| full base | 32.3 / 57.6 | 69.4 / 98.4 | 36.1 / 59.1 |
| ft base | 32.6 / 58.9 | 70.4 / 101.1 | 37.4 / 58.8 |
| ft tiny | 39.5 / 74.5 | 85.3 / 106.3 | 47.7 / 72.2 |
| off-the-shelf small | 46.4 / 52.6 | 76.7 / 92.6 | 59.9 / 66.3 |

- **The phone channel roughly doubles errors**, and embedded words suffer most.
- **Sizes rank as on real data.** 4-bit (NF4, Q4_0) hurts tiny and base most: ft tiny MER 39.5 → 47.9 at NF4. 8-bit changes are within noise.
- **Absolute embedded error is very high even for clean TTS.** This suggests the zero-shot voices often mispronounce embedded words, so this set is useful only for *relative* comparisons, and the human recordings (`PENDING_HUMAN.md` item 1) are still needed.

## 7. Efficiency (MacBook Pro M2 Pro, 4 threads, 100 test clips / 259 s; table 13, figure 6)
| Model | Format | File MB | Peak RSS MB | RTF | NSC Same MER (CPU subset) |
|---|---|---|---|---|---|
| ft tiny | whisper.cpp Q8_0 | 42 | 135 | 0.086 | 31.6 |
| ft tiny | whisper.cpp Q4_0 | 24 | 118 | 0.086 | 39.6 |
| full base | whisper.cpp Q8_0 | 78 | 196 | 0.176 | 25.8 |
| full base | whisper.cpp Q4_0 | 44 | 161 | 0.172 | 28.7 |
| full base | CTranslate2 int8 | 74 | 494 | 0.210 | 26.5 |
| full base | whisper.cpp F16 | 141 | 261 | 0.336 | 25.7 |
| full base | CT2 int8 + LID head | 74 | 508 | 0.296 | – |
| full base | CT2 int8 + LID + on-device re-ranking | 74 | 508 | 1.761 | – |
| **full small** | **whisper.cpp Q8_0** | 252 | 427 | **0.573** | **20.3** |
| full small | whisper.cpp Q4_0 | 139 | 315 | 0.584 | 21.0 |
| full small | CTranslate2 int8 | 238 | 839 | 0.742 | 21.1 |
| full small | whisper.cpp F16 | 465 | 650 | 1.299 | 20.3 |

- **Q8_0 is the sweet spot** on this machine: as fast as Q4_0, F16 accuracy, half the F16 size.
- **Q5_0 is about 2.4× slower than Q4_0.** The Apple build has no fast Q5 kernels.
- **QAT-merged models run at exactly the speed of their originals.**
- **Always-on re-ranking is about 8× slower than greedy.** Gating brings the estimate down to about 0.7 RTF for full base (§4).
- **Re-measurement:** the first speed batch was taken with Low Power Mode on or on battery, and the first ft_tiny / ft_small / ft_base batch was inconsistent. All of these were re-measured, and the originals are kept in `results/speed_first_batch_*.json`.

## 8. Further checks and ablations
**Diff Room training ablation (E; table 21)**

**Setup.** The Diff Room data was split by session 70/10/20, keeping E-test sessions out of all training. Each condition uses the ft_base recipe with no tuning.

| Training data (40 h NSC + ASCEND) | Same Room test MER | E Diff Room test MER | Δ vs A on Diff Room |
|---|---|---|---|
| A: 40 h Same Room (= ft_base) | 28.4 [27.2, 29.7] | **21.6** [20.5, 22.8] | – |
| B: 40 h Diff Room | 32.3 [30.7, 34.1] | 23.6 [21.9, 25.5] | **+2.0 [0.4, 3.8]*** |
| C: 20 h Same + 20 h Diff Room | 28.3 [27.1, 29.6] | 22.2 [20.8, 23.7] | +0.6 [−0.5, 1.7] |

**Negative result:** Diff Room training doesn't help on Diff Room speech. Same Room training is as good or better on both rooms, so the Same/Diff Room MER gap in §2 is *not* a training-domain mismatch.

**Partner-speech leakage in NSC (D3; table 18)**
- **Signals:** partner-transcript overlap, ECAPA speaker similarity, and hypothesis words that appear in the partner's transcript. The rule was fixed on a val pilot.
- **Flag rates:** 1.75% of Same Room test (0.88% partner words audible, 0.87% faint) vs 0.15% of Diff Room test.
- **Effect:** excluding flagged clips lowers Same Room MER by about 1 point for every fine-tuned model (full small 20.65 → 19.71, ft small 21.37 → 20.32). Diff Room changes by ≤ 0.3. No conclusion changes.

**Forced-alignment quality (D4; table 19)**
- MMS boundaries at *known* switch times in FLEURS splices: median **10 ms** (87% within 50 ms).
- MMS vs Whisper cross-attention at real switches: median 73 ms.
- MMS first-word start vs TextGrid interval edges: median 64 ms (Same Room) and 80 ms (Diff Room).
- The LID training labels are therefore accurate to tens of milliseconds.

**Critical-word definition (D2; table 24)**
- CWER on ASCEND was recomputed with the adjudicated list: 249 types plus all digit strings, 298 critical tokens vs 81 from the rule tagger.
- **Systems keep their ranking:** off-the-shelf small 35.8 → 36.2, ft small 19.8 → 29.9, full small 17.3 → 27.5.
- **The fine-tuned systems' advantage shrinks,** because the rule tagger mostly found digits and dates, which fine-tuning fixes, while the labelled list adds names, places and brands, which stay hard.

## 9. Limitations
- **NSC speakers were asked to switch.** Their switching is denser and in longer spans than ASCEND's (§1), so it may be more deliberate than natural code-switching.
- **ASCEND test has 2 speakers**, so its CIs (utterance bootstrap) are optimistic. NSC claims use speaker-cluster CIs (table 16).
- **What is automated, what was audited, and how well the annotators agree (table 20):**

  | Check | How it was done | Agreement | Human audit |
  |---|---|---|---|
  | D1 translation / transliteration labels (1,768 ASCEND pairs) → τ, θ | 2 independent LLM annotators (the second a fresh agent seeing only the rubric), adjudicated | κ 0.62 (3-class), **0.84** (translation vs rest) | `review/translation_audit.csv`, 30 pairs, **not yet done** |
  | D2 critical words (335 types) | 2 LLM annotators, adjudicated, + 500-utterance read-through for recall | κ **0.77**; vs rule tagger 0.55 / 0.47 | `review/critical_audit.csv`, 50 words, **not yet done** |
  | D3 partner leakage (NSC) | signal rule piloted on val; 158 borderline test clips judged by one annotator from text and signals | rule vs annotator κ 0.29 (63% agreement on hard cases) | `review/leak_audit.csv`, 20 clips, **not yet done** |
  | D4 alignment quality | synthetic splices with known switches + cross-checks | – | optional (`PENDING_HUMAN.md`) |
  | D5 stress set | **SYNTHETIC** TTS | – | human recordings **pending** |

  **No label in this report was checked by a person.** The LLM annotators are a stand-in, and the audit files are ready for the team.
- **Deployed accuracy** was measured on a fixed seeded subset of about 7.3k test utterances. whisper.cpp detects the language among all languages rather than zh/en only.
- **Semantic-loss candidates** were generated during training (every 4 steps) rather than once per epoch.
- **Re-ranking loop filter:** the post-hoc version is reported alongside the specified one.
- **Bugs found and fixed during the final rebuild (PROCESS_LOG, 25 Oct):**
  1. The first gated re-ranking run had an ASCEND val/test id collision. It was rerun as `*_gatedv2`, and only those runs are reported.
  2. Stale critical-word caches made the label-based CWER inconsistent. They were recomputed.
- **Single-machine speed.** The MacBook numbers are from one machine (Apple M2 Pro). Relative order between formats may differ on x86 or phone CPUs.

## 10. Third run (26 Oct): transliteration-penalised MRT (tables 25, 25b, 25c) — negative result
This run is separate from the frozen `results-final` numbers above, which it does not change; it is tagged `results-translit`. The rules were fixed in PROCESS_LOG before training.

**Change.** The cost of each MRT candidate becomes c_sem(y) + γ · N_tl(y). Here N_tl counts embedded reference units that the candidate renders as a phonetic transliteration, using the C3 detector `translit.py` unchanged (θ = 0.1). A transliterated unit therefore costs 1 + γ instead of 1, and γ = 0 is exactly the old loss (unit-tested).
- Base: γ ∈ {0.5, 1}, everything else as full_base.
- Noise control: a γ = 0 retrain with the same settings and seed. GPU training is not bit-reproducible.

**Selection on val.** The rule: MER ≤ full_base + 0.5 points, CWER ≤ full_base, then lowest transliteration rate. **Neither γ qualified**, because both had val CWER about 1 point above full_base (24.8 vs 23.8). The γ = 0 control, which uses the unchanged loss, was also 0.8 points above (24.6). The CWER failure is therefore within retraining noise, but by the rule it is a negative result, so small was not trained.

| Val, mean of ASCEND + NSC (%) | MER | CWER | Embedded error | Transliteration |
|---|---|---|---|---|
| full_base | 27.13 | 23.81 | 38.85 | 5.66 |
| γ = 0 retrain (control) | 27.35 | 24.60 | 39.30 | 5.86 |
| γ = 0.5 | 27.21 | 24.81 | 38.30 | 5.64 |
| γ = 1 | 27.01 | 24.79 | 39.09 | 5.82 |

**Test (decoded after the selection was recorded).** These are paired differences vs full_base in points; negative = better; * = the CI excludes 0. The ASCEND CIs are optimistic (2 test speakers). On NSC the speaker-cluster CIs agree with the utterance CIs.

| | ASCEND MER | NSC Same MER | NSC Diff MER | Transliteration rate (ASCEND / Same / Diff) |
|---|---|---|---|---|
| γ = 0 control | −0.14 [−0.56, 0.27] | −0.09 [−0.29, 0.14] | −0.08 [−0.28, 0.11] | +0.91 / −0.07 / +0.21 (all n.s.) |
| γ = 0.5 | −0.27 [−0.78, 0.22] | +0.17 [−0.13, 0.46] | **+0.31 [0.00, 0.62]*** | +0.13 / +0.05 / −0.39 (all n.s.) |
| γ = 1 | −0.37 [−0.81, 0.02] | −0.04 [−0.21, 0.16] | −0.02 [−0.26, 0.23] | −0.65 / −0.25 / −0.05 (all n.s.) |

- **Embedded error E** is not significant anywhere; the changes range from −0.8 to +0.6 points.
- **Translation rate** is not significant anywhere.
- **CWER:**
  - γ = 0.5 on NSC Same with the adjudicated list: +1.5 [0.1, 3.1]* (utterance CI). The speaker CI just touches 0: [0.0, 3.2].
  - With the rule tagger it is +1.1 (n.s.).
  - γ = 1 is n.s. everywhere.

**Reading.**
- The penalty does not measurably reduce transliteration on test. γ = 1 points the right way on all three sets (−0.65 / −0.25 / −0.05 points), but every CI includes 0, and the size is no bigger than the γ = 0 control's run-to-run spread (up to 0.9 points on ASCEND).
- The penalty does not reduce MER or E either.
- γ = 0.5 is slightly worse on NSC Diff MER (+0.3) and on NSC Same labelled-list CWER (+1.5).
- **Likely reason:** the penalty rarely applies. Only 3–8% of MRT candidates contain a transliterated embedded unit (training log `tl_cand_share`), MRT runs on 8 clips every 4th step, and transliteration is already only 3.5–6% of embedded units in full_base.
- **Caveat:** the same detector defines the training penalty and the evaluation rate, so even a translit_rate drop alone would not be strong evidence. MER and E, which the claim rests on, do not move.
- **Bug during the run:** parallel trainers raced on the shared LaBSE cache file. The fix changed only how the file is written. The crashed γ = 0 run was retrained under a new id, `fulltl_base_g0r`.

## 11. Fourth run (27 Oct): why the penalty did nothing, and a λ retune (tables 26, 26b, 26c) — negative result
**Diagnosis: loss scale.** `scripts/loss_scale_probe.py` measures gradient L2 norms on 40 training batches from full_base, without updating weights; output in `results/loss_scale_probe.json`. The transliteration part is measured as ‖∇(sem + γ·tl) − ∇sem‖, which is exact because the MRT gradient is linear in the cost.

| Term | Gradient norm | vs role CE, on an MRT step | Averaged over steps (MRT every 4th step) |
|---|---|---|---|
| Role-weighted CE | 5.66 | 1 | 1 |
| Semantic MRT, λ = 0.1 | 1.43 | 0.25 | 0.06 |
| Transliteration part, λ = 0.1, γ = 1 | 0.17 | 0.03 | ≈ 0.0075 |

- The third-run penalty was about 1/130 of the main loss.
- It is non-zero on 27.5% of MRT sub-batches, and its direction is unrelated to the role gradient (cosine 0.004).
- Logged loss values hide this: the MRT value is mean-subtracted, so it averages near 0.

**Retune.** λ ∈ {0.4, 1} × γ ∈ {0, 3} on base.
- λ = 0.4 puts the semantic gradient at about 1× role on MRT steps, λ = 1 at about 2.5×.
- γ = 3 puts the transliteration part at 0.36× (λ = 0.4) or 0.9× (λ = 1).
- The γ = 0 runs are controls.
- Rule, fixed before training: val MER ≤ full_base + 0.5, val CWER ≤ full_base + 1.0 (the margin is the third run's retrain noise), then lowest val translit_rate.
- Only λ = 1, γ = 3 qualified, so it was chosen, even though it had the highest val translit_rate (6.13% vs 5.66%).

**Test** (points; negative = better; * = the CI excludes 0; ASCEND CIs are optimistic; NSC speaker CIs agree):

| Comparison | MER: ASCEND / NSC Same / NSC Diff | Transliteration rate | Other significant changes |
|---|---|---|---|
| λ 1, γ 3 (chosen) vs full_base | +0.12 / +0.26 / +0.19 (n.s.) | +0.26 / −0.04 / −0.27 (n.s.) | none |
| λ 1, γ 3 vs λ 1, γ 0 | +0.24 / **+0.32*** / +0.03 | −0.13 / +0.27 / −0.21 (n.s.) | none |
| λ 0.4, γ 3 vs λ 0.4, γ 0 | +0.34 / +0.10 / **+0.35*** | −1.04 / −0.11 / −0.16 (n.s.) | none |
| λ 1, γ 0 vs full_base | −0.12 / −0.06 / +0.16 (n.s.) | n.s. | E +0.69* (NSC Same); CWER +6.2* (ASCEND) |
| λ 0.4, γ 0 vs full_base | −0.30 / +0.01 / −0.09 (n.s.) | n.s. | none |

**Reading.**
- Even at about 0.9× the role gradient, the penalty does not reduce transliteration, and it costs up to +0.35 MER. A stronger semantic MRT (λ = 1) on its own also hurts E and CWER.
- So weight was not the only problem.
- Hypothesis, not tested: MRT only reweights the candidates it generates. The correct embedded word is often missing from them, and the dictionary-swap candidates are translations (only half-penalised), so the penalty has no correct alternative to move toward.
- Val translation rate rose for the chosen run (1.99% vs 1.65%), which fits, but test is mixed.
- Next test: add the reference itself as a candidate.
- The same-detector caveat from §10 applies.
