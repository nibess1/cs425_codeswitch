# Key numbers for the slides (frozen 25 Oct 2026, tag `results-final`)

**How to read this file**
- Every number is a percentage on the **test** set unless stated.
- Brackets are 95% bootstrap CIs; * means the CI of a difference excludes 0.
- NSC = NSC Part 4 Mandarin speakers: Same Room 120 speakers, Diff Room (Zoom) 626 speakers. ASCEND test has 2 speakers, so its CIs are optimistic.
- Source = `results/tables/<file>`, with row = system id / slice.

## Data
| Number | Value | Source |
|---|---|---|
| NSC train (Same Room, capped) | 40.0 h, 35,647 utts, 62% code-switched | 01_data_counts.csv |
| Test sets | ASCEND 1,315 utts · NSC Same 8,876 · NSC Diff 4,085 | 01_data_counts.csv |
| Code-switched utterances, ASCEND vs NSC Diff | 26.8% vs 29.1% | 22_switching_statistics.csv |
| Switches per code-switched utterance, ASCEND vs NSC Diff | 2.12 vs 2.85 | 22_switching_statistics.csv |

## Baselines → full system (16-bit, greedy)
| Number | Value | Source |
|---|---|---|
| Off-the-shelf small, NSC Same MER | 55.1 [53.5, 56.7] | 02, small__fp16__offshelf_free |
| Fine-tuned small, NSC Same MER | 21.4 [20.6, 22.2] | 02, ft_small__fp16__ft |
| **Full small, NSC Same MER** | **20.6 [20.1, 21.2]** (speaker CI [18.6, 23.4]) | 02 / 16, full_small__fp16__full |
| Full small, NSC Diff MER | 17.1 [16.2, 18.1] (speaker CI [15.7, 18.6]) | 02 / 16 |
| Full small, ASCEND MER | 14.6 [13.6, 15.7] | 02 |
| Embedded-unit error, off-the-shelf small → full small (NSC Same) | 76.7 → 21.3 | 02 |

## RQ1: role, not just language
| Number | Value | Source |
|---|---|---|
| Same English word, embedded vs matrix error, off-the-shelf small | +15.4 pts [12.1, 18.6] | 11 |
| Same English word, embedded vs matrix error, fine-tuned small | +7.4 pts [5.2, 10.1] | 11 |
| Same Chinese character, embedded vs matrix error, fine-tuned small | +0.3 (n.s.) | 11 |
| Embedded log-odds b₁, off-the-shelf small | 0.94 [0.78, 1.15] | 12, term emb |
| Embedded-English interaction, fine-tuned small | +0.79 [0.42, 1.11] | 12, term emb_en |
| Translation share of embedded units (fine-tuned) | 0.5–1.9% | 02, translation_rate_pct |
| **Transliteration share of embedded units** | ft tiny ASCEND 10.9%, off-the-shelf small NSC 10.7%, full small 2.5–3.8% | 02, transliteration_rate_pct |

## RQ2: methods vs plain fine-tuning (paired, points)
| Number | Value | Source |
|---|---|---|
| Full base, NSC Same: MER / embedded | −2.3 [−3.5, −1.4]* / −3.6 [−6.5, −1.5]* | 03 |
| Full base, NSC Diff: MER / embedded | −1.3 [−2.3, −0.4]* / −1.4 [−2.3, −0.6]* | 03 |
| Full small, NSC Same: MER / embedded | −0.7 [−1.3, −0.1]* / −2.4 [−4.8, −0.8]* | 03 |
| Full small, NSC Diff: MER / embedded | −1.9 [−3.1, −0.9]* / −4.2 [−8.9, −1.1]* | 03 |
| Base + role-weighted only, NSC Same embedded | −2.7 [−5.6, −0.6]* | 03 |
| Always re-ranking (on-device), full base vs greedy | ASCEND embedded −4.3 [−7.7, −1.0]*, but NSC Diff MER +0.5 [0.1, 0.8]* | score.py --compare |
| **LID-gated re-ranking, full base (on-device), MER vs greedy** | ASCEND −0.7 [−1.0, −0.3]* · NSC Same −0.4 [−0.5, −0.2]* · NSC Diff −0.3 [−0.6, −0.2]* | 15, full_base__fp16__rerank_ondevice_filt_gatedv2 |
| LID-gated re-ranking, full base, ASCEND embedded error | −5.4 [−8.6, −2.3]* | 15 |
| LID-gated re-ranking, full small, ASCEND MER | −0.5 [−0.8, −0.2]* (NSC n.s.) | 15 |
| Share of clips re-ranked (θ = 200 ms) | 25% ASCEND · 28% NSC Same · 26% NSC Diff | 15 |

## RQ3: quantization
| Number | Value | Source |
|---|---|---|
| 8-bit (any backend) | embedded and matrix error change ≤ ~2 pts; R not reportable | 06 |
| R, ft tiny NF4, ASCEND (embedded English) | **3.47 [1.85, 6.13]** | 06 |
| R, ft base whisper.cpp Q4_0, ASCEND | **4.40 [1.78, 12.17]** | 06 |
| R, full base NF4, NSC (embedded Mandarin) | **1.75 [1.05, 3.03]** | 06 |
| Small at 4-bit | changes ≈ 1 pt, R ≈ 1 / not reportable | 06 |
| Rec, mixed precision (2 blocks), ft tiny NF4, ASCEND | **0.65 [0.29, 0.91]** | 14 |
| Rec, QLoRA QAT, ft tiny NF4, ASCEND / NSC | **0.83 [0.57, 1.26] / 0.72 [0.46, 0.98]** | 14 |
| Rec, QLoRA QAT, full base NF4, NSC en-main | **0.77 [0.54, 1.10]** | 14 |
| Rec, QAT merged → whisper.cpp Q4_0, ft tiny NSC | **−0.12 [−0.42, 0.10]** (no transfer) | 14 |
| whisper.cpp mixed precision | not possible (loader rejects per-tensor types) | PROCESS_LOG |

## Efficiency (MacBook Pro M2 Pro, 4 threads)
| Number | Value | Source |
|---|---|---|
| Full small, whisper.cpp Q8_0 | RTF 0.57, 252 MB file, 427 MB peak, NSC MER 20.3 | 13 + fig6 CSV |
| Full base, whisper.cpp Q8_0 | RTF 0.18, 78 MB, NSC MER 25.8 | 13 + fig6 CSV |
| Ft tiny, whisper.cpp Q8_0 | RTF 0.086, 42 MB, NSC MER 31.6 | 13 + fig6 CSV |
| Q5_0 vs Q4_0 speed | ~2.4× slower | 13 |
| LID head overhead (base) | RTF 0.21 → 0.30 | 13 |
| Always-on re-ranking (full base, on-device) | RTF 1.76 (~8× greedy) | 13 |
| Gated re-ranking, estimated RTF (full base) | ≈ 0.66–0.70 (0.30 + share × 1.47) | 13 + 15 |

## Other checks
| Number | Value | Source |
|---|---|---|
| Diff Room training (40 h) vs Same Room, on Diff Room test | +2.0 [0.4, 3.8]* MER (worse; negative result) | 21 |
| 20 h + 20 h mix vs Same Room, on Diff Room test | +0.6 [−0.5, 1.7] (n.s.) | 21 |
| Partner leakage flagged, Same / Diff Room | 1.75% / 0.15%; excluding flagged → about −1 pt MER (Same) | 18, 18b |
| MMS boundary error at known switch times | median 10 ms, 87% within 50 ms | 19 |
| Label agreement (Cohen's κ) | D1 0.62 (3-class), 0.84 (translation vs rest); D2 0.77; D3 rule vs annotator 0.29 | 20 |
| CWER ASCEND, rule tagger vs adjudicated labels, full small | 17.3 → 27.5 | 24 |
| SYNTHETIC stress set, full small, clean vs phone MER | 22.3 vs 36.8 | 17 |

## NSC attribution statement (for slides and report)
> Contains information from the National Speech Corpus (Part 4) accessed from the Infocomm Media Development Authority of Singapore (IMDA), which is made available under the terms of the Singapore Open Data Licence version 1.0.

This follows the licence's standard template. **Check it against the exact wording on the licence page before submission** (`PENDING_HUMAN.md` item 2).
