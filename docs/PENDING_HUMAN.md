# Steps that need people (rewritten 25 Oct, D6)

Since 20 Oct, most of the original human steps have **automated stand-ins**: LLM double annotation with adjudication, signal rules, and synthetic checks. Every stand-in is documented in `docs/PROCESS_LOG.md` and reported in the results as a proxy.

What is left for people is (a) things no proxy can replace, and (b) short audits that tell us whether to trust the proxies. Each audit file is a small random sample, so a few minutes of checking is enough.

## A. Still required (no automated substitute)

| # | Task | Why it matters | Files | After it is done |
|---|---|---|---|---|
| 1 | **Record the human stress set.** Each member reads their 30 assigned sentences on a phone (quiet room, normal speed). | The stress-set rows in the report are currently **SYNTHETIC** (CosyVoice 2 TTS, 4 voices, clean + simulated phone channel). Real read speech is needed before claiming anything about critical words in people's speech. | `stress/assignments.tsv` (member → 30 sentence ids); save to `data/stress/audio/<member>/<id>.wav` | `python scripts/prepare_stress.py`, then decode with `--sets stress_test` and re-run `scripts/make_results.py` |
| 2 | **Copy the exact attribution sentence** from the Singapore Open Data Licence v1.0 page into the report and `data/nsc_part4/VERSION.txt`. | Licence confirmed by the team, but the Dropbox share has no LICENCE file, so the wording is not in the repo. | `data/nsc_part4/VERSION.txt`, `docs/REPORT.md` (Data section), `results/KEY_NUMBERS.md` | none |
| 3 | **Confirm the NSC release/version** the Dropbox copy corresponds to. | Reproducibility; the share has no VERSION or UPDATE HISTORY file. | `data/nsc_part4/VERSION.txt` (manifest hash + download date recorded) | edit `VERSION.txt` |

## B. Audits of automated labels (recommended, about 1 h in total)
These are optional. If an audit disagrees a lot with the automated labels, say so in the report's limitations. Each line gives the size of the effect it could change.

| # | Automated step (proxy) | Audit file | What to mark | Effect if the proxy is wrong |
|---|---|---|---|---|
| 4 | **D1: translation / transliteration labels.** 1,768 ASCEND output pairs, 2 LLM annotators (κ 0.62 for 3 classes, 0.84 for translation vs rest), adjudicated. Sets τ = 0.65 and θ = 0.1. | `review/translation_audit.csv` (30 pairs) | `human_label` ∈ {translation, transliteration, other} | conf_rate changes by about ±20% for τ ∈ [0.6, 0.7] (`results/d1_tau_sensitivity.csv`). It stays ≤ 2% of embedded units either way. |
| 5 | **D2: critical-word list.** 249 types from 2 LLM annotators (κ 0.77) plus a 500-utterance read-through. Used for CWER mode (b). | `review/critical_audit.csv` (50 words) | `human_critical` ∈ {0, 1} | Mode (b) vs mode (a) CWER: both are reported. |
| 6 | **D3: partner-speech leakage in NSC.** Signal rule (s1–s5); the rule was piloted on val; flags 1.75% of Same Room and 0.15% of Diff Room test utterances. | `review/leak_audit.csv` (20 clips: flagged and unflagged) | `human_leak` ∈ {yes, no} | Excluding flagged utterances lowers Same Room MER by about 1 point. All conclusions unchanged. |
| 7 | **D4: forced-alignment quality** (MMS). Checked automatically on FLEURS splices with known switches (median 10 ms), against Whisper cross-attention (73 ms), and at TextGrid edges. | none needed; optional: `review/alignments.csv` | true switch times (Praat) | LID boundary metrics assume about 10–70 ms alignment error. |

## C. Decided while running (deviations from the proposal, for the report)
- **NSC training data** is capped at 40 h (all-mixed first, up to 30 h, then English-only). Val/test are random 3 h / 8 h samples of their speakers; the Diff Room test is a 4 h sample.
- **ASCEND:** val/test speakers don't appear in train, and there are only 2–3 speakers per split, so ASCEND CIs (bootstrapped over utterances) are optimistic. NSC CIs are speaker-cluster bootstraps (A3).
- **Semantic MRT candidates** are generated on a sub-batch every 4 steps instead of once per epoch.
- **Deployed (CPU) rows** use a fixed seeded subset (all of ASCEND test, 4,000 NSC test, 2,000 Diff Room utterances). R and Rec compare against the same-backend 16-bit reference (ct2_f32, ggml_f16).
- **whisper.cpp** runs with `-l auto`. CT2 and HF restrict the language token to {zh, en}.
- **whisper.cpp mixed precision (B2) is not possible**: the loader rejects per-tensor types. Mixed precision is reported for bitsandbytes NF4 only.
- **Decoding** is greedy for main rows. Re-ranking uses beam 4 and the gated variant (C1, θ chosen on val).
- **Speed** is measured on the MacBook (plugged in, 4 threads). The first ft_small/ft_base batch was re-measured after Low Power Mode was found on.
- **Synthetic stress set (D5)** uses CosyVoice 2-0.5B zero-shot (code and model card: Apache-2.0) with FLEURS prompts (CC-BY 4.0). It is labelled SYNTHETIC everywhere and is not a substitute for item 1.
- **LLM labels** are not human gold. Agreement statistics are reported with every table that uses them.
- **Diff Room training ablation (E):** training on Diff Room audio did not help on Diff Room test (a negative result; see PROCESS_LOG 25 Oct).
