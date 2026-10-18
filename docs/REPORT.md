# Results report: Robust and Compact Recognition of Code-Switched Speech

CS425 group project (Rui Cong, Phoebe, Nibras, Bridgette). This report covers every experiment
in the revised proposal (13 Oct 2026). Experiments were run 14–17 Oct on one NVIDIA L40S GPU
(smu2, GPU 4), smu2's CPUs, and a MacBook Pro M2 Pro.

- **Data behind every number:** `results/tables/*.csv`, with column definitions in `results/tables/README.txt`.
- **How each step was done, and every decision and bug:** `docs/PROCESS_LOG.md`.
- **Human checks still outstanding:** `PENDING_HUMAN.md`.

All numbers are on **test** sets unless stated. Units are one Chinese character, English word or
digit string. **MER** = (S+D+I)/N. Role errors (embedded E, matrix-at-switch Mₛ, other matrix Mᵢ)
come from Pass 2, which charges each error block's insertions to its reference units, so they can
exceed 100%. Brackets are 95% paired-bootstrap CIs (1,000 utterance resamples); * means the CI
of a difference excludes 0. Decoding is greedy with a 128-token cap unless stated.

---

## 1. Data
| Set | Utterances | Hours | Code-switched | Embedded units | Speakers |
|---|---|---|---|---|---|
| ASCEND train / val / test | 9,869 / 1,130 / 1,315 | 8.8 / 0.9 / 0.9 | 28% / 18% / 28% | 7,081 / 471 / 772 | 18 / 3 / 2 |
| NSC Part 4 Same Room train (Mandarin speakers) | 35,647 | 40.0 | 62% | 105,493 | 412 |
| NSC Same Room val / test | 3,367 / 8,876 | 3.0 / 8.0 | 25% / 29% | 4,099 / 11,042 | 60 / 120 |
| NSC Different Room test (Zoom) | 4,085 | 4.0 | 29% | 5,648 | 626 |

- **NSC sampling.** NSC train is capped at 40 h, out of about 310 h of Same Room speech: all code-switched utterances up to 30 h, topped up with English-only ones. NSC val and test are random samples within a session-level split, and Diff Room is test-only.
- **ASCEND speakers.** In the Hugging Face release, ASCEND val and test speakers don't appear in train, but those splits have only 2–3 speakers each, so ASCEND CIs are optimistic.
- **Main language.** ASCEND is mostly Chinese-main (embedded English). NSC is mostly English-main (embedded Mandarin) but has a sizeable Chinese-main share.

## 2. Baselines and fine-tuning
| System | ASCEND MER | ASCEND E | NSC MER | NSC E | Diff Room MER |
|---|---|---|---|---|---|
| small, off-the-shelf, model picks language | 26.3 [24.4, 28.1] | 56.8 | 55.1 [53.5, 56.7] | 76.7 | – |
| small, forced `<|zh|>` | 35.5 | 59.8 | 104.5 | 78.0 | – |
| small, forced `<|en|>` | 89.2 | 58.7 | 57.9 | 94.1 | – |
| small, forced `<|zh|><|en|>` | 30.4 | 88.5 | 64.5 | 80.8 | – |
| tiny, fine-tuned | 26.1 | 53.2 | 34.5 | 36.2 | – |
| base, fine-tuned | 21.3 | 55.3 | 28.4 | 31.0 | – |
| **small, fine-tuned** | **14.8 [13.7, 15.8]** | **30.6** | **21.4 [20.6, 22.2]** | **23.7** | – |

The Diff Room column and the remaining tiny/base rows are in `02_accuracy_16bit_all_systems.csv`.

- **Fine-tuning** on ASCEND + NSC halves the error, most of all on NSC (55.1 → 21.4 for small).
- **Forcing the wrong language token is catastrophic.** MER goes above 100% because of insertions. Prompting with both tokens doesn't beat letting the model choose.

## 3. RQ1: where does Whisper fail, and is it role or language?
1. **Embedded units have the highest error** in every system and corpus (E > Mₛ > Mᵢ). Off-the-shelf small: ASCEND E 56.8 vs Mᵢ 23.4; NSC E 76.7 vs Mᵢ 48.6.
2. **Same word, embedded vs matrix** (`11_*.csv`): the same word type has a higher error when embedded than when part of the main language.

   | System | English words | Chinese characters |
   |---|---|---|
   | Off-the-shelf small | +15.4 pts [12.1, 18.6] | +26.0 pts |
   | Fine-tuned small | +7.4 [5.2, 10.1] | +0.3 (n.s.) |

   This is a role effect, not just word type. **Fine-tuning closes the gap for embedded Mandarin but not for embedded English.**
3. **Logistic regression (A5)** controls for language, training frequency, spoken length and corpus, with speaker-cluster bootstrap CIs. The embedded coefficient is in `12_rq1_logistic_regression.csv`.
4. **Errors are rarely translations.** Only 0.4–5.5% of embedded units fall in a translation block. Instead, Whisper *transliterates* embedded English phonetically (deadline→得来, python→派森, tutor→跳特, hong kong→红矿). LaBSE separates the two cleanly: translations 0.86–0.95, transliterations 0.26–0.49. So **the proposal's hypothesis "many embedded errors are translations" is not supported.** τ = 0.70 is pending a hand check.

## 4. RQ2: do the proposed methods reduce these errors at 16-bit?
Methods were tuned on base, with hyperparameters chosen on validation by rules fixed in advance (`04_*.csv`):
- role weights **α = 0.5, β = 0.5**
- semantic loss **λ = 0.1, ρ₀ = 0.5**

| System (vs plain fine-tuning, same size) | ASCEND MER | ASCEND E | NSC MER | NSC E | Diff Room MER | Diff Room E |
|---|---|---|---|---|---|---|
| base + role-weighted | −0.7 | −14.6 | −0.6 | **−2.7*** | −0.6 | **−0.9*** |
| base + semantic only | −1.2 | −14.6 | **−1.3*** | −1.7 | **−1.1*** | −0.4 |
| **base + role-weighted + semantic (full)** | −1.0 | −16.1 | **−2.3*** | **−3.6*** | **−1.3*** | **−1.4*** |
| small + role-weighted (best subset) | **−0.5*** | −2.3 | +0.1 | −1.1 | −0.9 | −3.3 |
| **small full** | −0.2 | −0.8 | **−0.7*** | **−2.4*** | **−1.9*** | **−4.2*** |

- **The role-weighted loss lowers embedded-unit error.** The two losses combine best: the full system is significantly better on both NSC test sets for base and small, with no rise in critical-word error.
- ASCEND changes are mostly within noise, because its test set is small (2 speakers) and dominated by a few utterances with long insertion runs.
- **Semantic MER ≈ MER**, because so few errors are translations (§3.4), so the semantic loss acts mostly as a sequence-level minimum-risk loss.

### N-best re-ranking (method d)
**What it is.** Re-ranking builds a shortlist of up to 12 transcripts per clip: beam 4 under forced `<|zh|>`, forced `<|en|>` and the model's own language choice. It then picks one by combining:
- Whisper's per-token log-probability
- an MMS CTC acoustic-match score
- agreement between each word's language and the LID head

The weights are tuned on validation. The *on-device* variant drops MMS and takes word timings from Whisper's cross-attention.

**Result against each system's own greedy output** (filtered variants):

| System | ASCEND MER | ASCEND E | NSC MER | Diff Room MER |
|---|---|---|---|---|
| plain FT base | −1.0 | −14.8 (n.s.) | **+2.4*** | **+4.9*** |
| full base | **−0.9*** | **−4.2*** | +0.1 | **+0.7*** |
| full small | −0.3 | −1.9 (on-device **−3.0***) | **+0.8*** | **+1.4*** |

- **Helps embedded English inside Chinese-main speech:** embedded error −3 to −4 points for the full systems.
- **Slightly hurts English-main speech.**
- **Unfiltered, it is harmful.** The proposal's length-normalised score lets repetition loops win: val NSC MER was 61% vs 27% greedy for plain base. A post-hoc loop filter (drop candidates with gzip compression ratio > 2.4, Whisper's own heuristic, tuned on val only) removes most of that.
- **Too slow for on-device use.** On the MacBook, on-device re-ranking is about 8× slower than plain decoding (§6).
- **Critical-term flags** (raised when the top-3 candidates disagree on a critical word) are about half right but catch few errors. Full small, with-MMS: precision 55% / recall 13% on NSC test, 39% / 33% on ASCEND. On-device: 49% / 15% on NSC. So they are a weak safety net, not a reliable one (`methods.md`).

### Language-ID head (method c)
- **Accuracy:** language accuracy on speech frames is 96% (ASCEND val) and 95% (NSC val).
- **Switch boundaries:** median error 60 ms and 120 ms; 75–85% of switches detected.
- **Training data:** a frozen fine-tuned base encoder, ASCEND+NSC frames labelled by MMS forced alignment, and 6,000 FLEURS word-piece splices.

## 5. RQ3: does quantization hurt embedded units more, and do the fixes help?
- **8-bit is safe** in every backend (bitsandbytes LLM.int8, CTranslate2 int8, whisper.cpp Q8_0) and every size: embedded and matrix error move by at most about ±2 points. R isn't reportable because the matrix change is within noise, as the proposal's statistics plan anticipated.
- **4-bit hurts embedded units more for the smaller fine-tuned models** (R > 1, CI excludes 1):

  | Model | Format | Slice (embedded language) | Δ embedded | Δ matrix | R |
  |---|---|---|---|---|---|
  | fine-tuned tiny | NF4 | ASCEND, Chinese-main (English) | +16.2 | +4.7 | **3.47 [1.85, 6.13]** |
  | fine-tuned base | whisper.cpp Q4_0 | ASCEND, Chinese-main (English) | +8.2 | +1.9 | **4.4 [1.8, 12.2]** |
  | full base | NF4 | NSC, English-main (Mandarin) | +3.0 | +1.7 | **1.75 [1.05, 3.03]** |
  | fine-tuned tiny | whisper.cpp Q4_0 | NSC, English-main (Mandarin) | +8.9 | +6.4 | **1.40 [1.01, 1.86]** |

  The off-the-shelf pilot (R ≈ 0.6) doesn't carry over to fine-tuned models, which fits the proposal's hypothesis. Both embedded English and embedded Mandarin are affected.
- **Small is robust at 4-bit.** Changes are about 1 point and R is about 1 or not reportable.
- **Deployed rows** are compared with a 16-bit run of the *same* CPU backend. Off-the-shelf HF fp16 greedy decoding loops far more than the CPU decoders (8.2k vs 3.8k inserted units on 4,000 NSC utterances), so comparing across backends would credit quantization with a decoder difference.
- **Sensitivity scan (A6, NF4)** on fine-tuned and full small: the most sensitive blocks are encoder layer 0, decoder layer 11 and encoder layer 7. The chosen mixed-precision configurations are k = 2 and k = 6 blocks kept at 16-bit.
- **Recovery is undefined on small.** Mixed precision and QLoRA QAT were run on small, where the NF4 loss is already within noise, so Rec can't be estimated. The fixes should be re-run on tiny or base, where 4-bit loss is significant.

## 6. Efficiency (MacBook Pro M2 Pro, 4 threads, 100 clips / 259 s)
SPEED_TABLE

GPU memory footprints of the simulated formats are in `10_model_size_and_memory.csv`.

## 7. Limitations
- **NSC speakers were asked to switch**, which may be more deliberate than natural code-switching.
- **ASCEND test has 2 speakers**, so its CIs are optimistic.
- **Automatic stand-ins still in place:**
  - critical-word labels
  - τ for translation detection
  - partner-leakage filter
  - alignment quality

  The stress set isn't recorded yet (`PENDING_HUMAN.md`).
- **Deployed accuracy** was measured on a fixed seeded subset of about 7.3k test utterances.
- **whisper.cpp** detects the language among all languages rather than zh/en only.
- **Semantic-loss candidates** were generated during training rather than once per epoch.
- **Re-ranking loop filter:** the post-hoc version is reported alongside the specified one.
