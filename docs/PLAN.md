# CS425 — Robust and Compact Recognition of Code-Switched Speech
## Implementation plan (derived from the revised proposal, 13 Oct 2026)

Team: Rui Cong (Data & Eval) · Phoebe (Fine-tuning) · Nibras (Quantization + NSC) · Bridgette (LID & post-processing)
Hard dates: **data frozen 19 Oct** · **plain FT done 22 Oct** · **presentation 10 Nov (>90% results)** · **report + code 27 Nov**

Priority if time runs short (drop from the bottom): role-weighted loss → quantized FT models + R → sensitivity scan → QAT → re-ranking → semantic loss → segmented decoding.

---

## 0. Infrastructure (Day 1, 13–14 Oct) — Nibras sets up, everyone uses

### 0.1 Repo
Shared GitHub repo, cloned on smu2 at `~/cs425-codeswitch-asr`. Data, audio, checkpoints and predictions are **never** committed.

```
cs425-codeswitch-asr/
├── PLAN.md
├── README.md                  # setup + how to run each stage
├── environment.yml            # one pinned env for everyone
├── .gitignore                 # data/ ckpt/ preds/ runs/ *.wav *.pt *.bin *.gguf
├── configs/                   # one YAML per experiment (model, data, loss, α, β, λ, ρ0, precision)
├── csasr/                     # importable package
│   ├── text/normalize.py      # §3.2 shared normalizer (Rui Cong)  ← everything depends on this
│   ├── text/units.py          # unit splitter, script-change spacing
│   ├── text/labels.py         # language / neutral / main-language / role / critical tags
│   ├── score/align.py         # Pass 1 (MER) + Pass 2 (error blocks, A7 charging)
│   ├── score/confusion.py     # CC-CEDICT + LaBSE translation test
│   ├── score/bootstrap.py     # paired bootstrap, R, Rec, intervals
│   ├── data/ascend.py  data/nsc.py  data/fleurs.py  data/stress.py
│   ├── train/finetune.py      # HF Seq2SeqTrainer subclass; losses plug in
│   ├── train/losses.py        # role-weighted (A1), semantic MRT (A2)
│   ├── lid/                   # frame labels, head, training, smoothing (A3)
│   ├── decode/                # baseline prompts, N-best pooling, re-ranking (A4), segmented
│   └── quant/                 # bnb sim, CT2 + whisper.cpp export, sensitivity (A6), QLoRA QAT
├── scripts/                   # thin CLIs: prepare_*.py, train.py, decode.py, score.py, quantize.py
├── tests/                     # unit tests for normalizer + scorer (worked examples from the proposal)
└── notebooks/                 # analysis only, never pipeline logic
```
Out-of-repo (on smu2, gitignored): `data/` (raw + prepared manifests), `ckpt/`, `preds/`, `runs/` (logs).

### 0.2 Environment
- micromamba env `cs425` (Python 3.10/3.11): `torch` (CUDA), `transformers`, `datasets`, `accelerate`, `evaluate`, `bitsandbytes`, `peft`, `ctranslate2`, `faster-whisper`, `opencc-python-reimplemented`, `cn2an`, `jieba`, `textgrid`/`praatio`, `soundfile`, `librosa`, `torchaudio` (MMS forced aligner), `uroman`, `sentence-transformers` (LaBSE), `jiwer` (sanity check only), `statsmodels`, `pandas`, `wandb` or `tensorboard`.
- whisper.cpp built from source (CPU build for Mac speed tests; GPU build optional).
- Pin the versions in `environment.yml`, commit it, and have everyone install from it.
- HF cache → `~/hf-cache` (already exists); set `HF_HOME` in `.bashrc`.

### 0.3 smu2 compute etiquette
- 8× L40S 46 GB. GPUs 0–1 are already partly used by others, so check `nvidia-smi` and pin jobs with `CUDA_VISIBLE_DEVICES`.
- Run long jobs inside `tmux` (or with `nohup`). The SSH connection drops when you leave the campus network/VPN.
- `/home` is 95% full (≈730 GB free, shared). Budget: NSC raw ≤ ~250 GB, clipped 16 kHz audio ~30–60 GB, checkpoints ~50 GB. Delete intermediate checkpoints; keep best + last.

### 0.4 One prediction format (frozen by 17 Oct)
JSONL, one line per utterance:
```json
{"utt_id": "...", "corpus": "ascend|nsc_same|nsc_diff|stress", "split": "test",
 "model": "base", "precision": "fp16|int8_bnb|nf4_bnb|ct2_int8|ggml_q8_0|ggml_q5_0|ggml_q4_0",
 "method": "offshelf_zh|offshelf_en|offshelf_both|ft|ft_role_a1b0.5|...", "ref": "...", "hyp": "...",
 "lang_token": "zh", "nbest": [...optional...]}
```
`scripts/score.py preds/*.jsonl` → all metrics table + bootstrap CIs. **Nobody computes metrics any other way.**

---

## Phase 1 — Foundations (13–17 Oct, parallel)

### 1A. Normalizer, labels and scorer — Rui Cong (critical path: everyone is blocked on this)
1. `normalize()`: OpenCC t2s → cn2an numerals→digits → lowercase EN + variant map → strip punctuation (incl. ，。？) → strip corpus markers (`(um)` deleted; `[oh]`, `!wah!`, `#name#`, speaker tags stripped, word kept).
2. `units()`: each Han char, Latin word, digit string = 1 unit. `training_target()`: inserts a space at every script change.
3. Labels: per-unit language (zh/en/neutral: digits, spelled letters, lah/leh/lor/meh/hor/sia); main language via jieba word counts; balanced on ties; roles E / Mₛ / Mᵢ; switch-point flag.
4. Critical-word tagger: regex (numbers, dates, amounts) + NER + term lists (drug names, tech terms). Hand-checking of the test sets happens in Phase 2.
5. Scorer Pass 1: Levenshtein over units, tie-break *substitution over ins+del* → S, D, I, N, MER.
6. Pass 2: group consecutive errors between correct units into blocks; charge per A7; **assert Σ Nᵣ·Eᵣ == S+D+I** on every run.
7. Language-confusion test: CC-CEDICT lookup ∨ LaBSE cosine ≥ τ (τ tuned later on 100 hand-labelled blocks).
8. sMER (ρ0 = 0.5, fixed now), CWER, paired bootstrap (1000 resamples, utterance-level; speaker-level for the A5 regression), R with denominator-CI check, Rec.
9. `tests/`: the proposal's examples ("我明天要交 assignment 啦", assignment→作业 block) as unit tests.
**Done when:** the tests pass and running the scorer on a dummy JSONL prints the full table.

### 1B. NSC Part 4 + off-the-shelf quantized runs — Nibras
1. Download NSC Part 4 from the team Dropbox to `data/nsc_part4/raw` (`scripts/fetch_nsc_dropbox.py`). It contains Same Room Audio/Scripts and Diff Room Audio/Scripts. **Fetch only `*cs-chn*` (Mandarin mother tongue) files** to save disk. Record the corpus VERSION.
2. Find Mandarin speakers: filename tag `cs-chn`, cross-checked against the speaker sheet, with Han characters in transcripts as a fallback.
3. Parse the TextGrids → utterance clips ≤ 30 s (merge adjacent intervals up to 30 s, cut at silences) → 16 kHz mono WAV + manifest (`utt_id, speaker, room, path, start, end, ref_raw, ref_norm`).
4. Partner leakage: listen to 20 files, drop segments with audible untranscribed partner speech, and document the rule used.
5. Speaker split 70/10/20 (Same Room). Diff Room is test-only (robustness).
6. Off-the-shelf Whisper tiny/base/small on ASCEND + NSC test at fp16 / bnb int8 / bnb NF4 with the 3 baseline prompts → JSONL. Record the list of quantized modules + measured GPU memory.
7. Export pipeline: HF checkpoint → CTranslate2 int8; HF → ggml → whisper.cpp Q8_0/Q5_0/Q4_0. Verify each one's output matches the HF model on 20 clips.
**Done when:** NSC manifests exist for each split, the counts table is drafted, and off-the-shelf JSONLs are in `preds/`.

### 1C. Forced alignment + speed harness — Bridgette
1. uroman-romanize the normalized transcripts; run the torchaudio MMS forced aligner on ASCEND, NSC and FLEURS (en_us, cmn_hans_cn).
2. Unit start/end times → 20 ms frame labels {en, zh, non-speech}, stored as `.npy` per utterance.
3. Two members hand-check 30 utterances and report the boundary error. If it's large, LID results are limited to the hand-checked clips.
4. CPU speed harness (MacBook, 4 threads): RTF, peak RSS, file size. One script for CT2 and whisper.cpp.

### 1D. Fine-tuning pipeline + baselines — Phoebe
1. ASCEND loader (HF `CAiRE/ASCEND`, official splits; report the speaker overlap).
2. `Seq2SeqTrainer` subclass: fp16/bf16 full fine-tuning, target prefix = `<|sot|><|main-lang|><|transcribe|><|notimestamps|>` + spaced target; at test time the model predicts the language token itself; `max_new_tokens=128`.
3. Hooks for a custom loss: `compute_loss` receives per-token weights (A1) and an optional MRT term.
4. Token↔unit span mapping by prefix decoding (needed for A1); unit-test it on mixed sentences.
5. Checkpoint every 500 steps (needed on Colab; harmless on smu2).
6. Smoke test: 1 epoch of tiny on 500 ASCEND clips end-to-end → JSONL → scorer.

---

## Phase 2 — Data frozen (by 19 Oct) — All
1. Freeze: copy final manifests to `data/frozen_v1/`, chmod read-only, and record NSC VERSION + git hash.
2. **Size table** per split: utterances, mixed utterances, embedded units, critical words (per corpus and main language). Check this against the "±4 pts at 400 embedded units" detectability bar.
3. **Fallback decision:** if NSC Mandarin isn't ready, NSC becomes test-only and FT uses ASCEND only.
4. Stress set: write ~120 sentences (60 Chinese-main, 60 English-main) with drug names, dates, deadlines, amounts and tech terms. All 4 members record them on phones (16 kHz). That gives 480 clips.
5. Hand-check critical-word labels on the ASCEND/NSC/stress test sets (2 members each, resolve disagreements).
6. Hand-label 100 blocks to set τ for LaBSE.

## Phase 3 — Baselines and RQ1 (20–26 Oct)
1. **Plain FT** of tiny/base/small on ASCEND+NSC train (by 22 Oct). Select checkpoints on val MER. Rough budget on L40S: tiny < 1 h, base ~1–2 h, small ~3–5 h.
2. Evaluate off-the-shelf (3 prompts) and plain FT on all test sets.
3. **RQ1** (Rui Cong): role-level error tables; language-confusion rate; same-word embedded-vs-matrix comparison (EN words in ASCEND-zh vs NSC-en sentences); A5 logistic regression with a speaker-bootstrap CI on b₁.
4. **Role-weighted loss** (Phoebe): grid on base α∈{0.5,1,2} × β∈{0,0.5} = 6 runs; pick on val.
5. **Quantize the FT models** (Nibras): bnb int8 / NF4 for all 3 sizes; compute R separately for Chinese-main (ASCEND) and English-main (NSC), with CIs; report the raw differences when the denominator CI includes 0.
6. **LID head** (Bridgette): 2-layer MLP on frozen encoder frames; train on ASCEND+NSC aligned frames + FLEURS word-sized splices (incl. same-language cross-speaker splices, speed perturb 0.9/1.1); mask padding; 5-frame median + merge segments < 200 ms. Report frame accuracy + boundary error.

## Phase 4 — Methods, compression and ablations (27 Oct–4 Nov)
1. **Semantic MRT loss** (Phoebe): candidate set per clip = beam-4 under `<|zh|>`, `<|en|>` and a free token + one dictionary-swap candidate per embedded word; dedup after normalization; regenerate each epoch; skip clips where all costs are equal. ℒ = ℒrole + λℒsem; grid λ∈{0.1,0.3} × ρ0∈{0.2,0.5} on base (4 runs). Judge by sMER + CWER.
2. **Re-ranking + flags** (Bridgette): pool the N-best from 3 prefixes; score = Whisper logp/len + γ·MMS CTC logp/char + η·LID agreement; tune γ, η on val. Report two variants: with MMS (upper bound) and the on-device variant (γ=0, Whisper word timestamps). Flag critical-term disagreement; report flag precision/recall.
3. **Sensitivity scan** (Nibras): for each encoder/decoder block g, quantize only g and compute the A6 KL on embedded tokens (teacher-forced, val). Confirm the top 5 with embedded error. Keep top-k at fp16, k ≤ ¼ of blocks. Compute Rec.
4. **QAT via QLoRA** (Nibras): freeze the NF4 FT model + rank-16 LoRA, same loss; evaluate with adapters at 16-bit; then merge → re-quantize → CT2/whisper.cpp and report how much of the gain survives. STE-QAT is a stretch goal.
5. **Deployed rows** (Nibras + Bridgette): CT2 int8 + whisper.cpp Q8/Q5/Q4 for off-the-shelf, plain FT and the full system; accuracy re-measured; RTF / peak memory / size on the Mac, LID head included as a separate module.
6. **Ablations on base**: each method added alone to plain FT; each removed from the full system; segmented decoding (0.5 s padding, previous-segment prompt).
7. **Transfer to small**: full system + best subset, run once with the base-tuned settings (2 runs).

Training-run budget ≈ 18: 3 plain FT, 6 α/β, 4 λ/ρ0, 2 on small, 1 LID head, 2 QAT.

## Phase 5 — Write-up
- **5–9 Nov**: freeze results tables (all from `score.py`), make figures, build slides. **10 Nov presentation.**
- **11–27 Nov**: fill remaining gaps, write the report (limitations: deliberate switching in NSC, ASCEND speaker overlap, alignment quality), clean the repo, write a README with reproduction commands. **27 Nov submit.**

---

## Reproducibility rules
- Every run = one YAML in `configs/` + seed + git hash written into the run directory and the JSONL.
- Tune on val only; touch test once per final configuration.
- Hyperparameters are tuned on **base only**, then transferred unchanged to small.

## Open items to confirm with the team
- [ ] GitHub repo owner/name; add all 4 members.
- [ ] Do teammates have smu2 accounts? If not, who runs their GPU jobs (or do they use SCIS/Colab)?
- [ ] NSC Part 4 licence/VERSION: confirm the Dropbox copy is the version we cite.
- [ ] Who owns the stress-set script text, and the recording deadline.
