# Testing plan: human stress-set recordings (7 Nov 2026)

Written before any decode. Rules below are fixed before looking at any result on the new recordings.
Tag for everything here: `results-human`. Nothing in `results-final`, `results-translit` or `results-lambda` is changed.

## 1. What the recordings are
- The **human version of the D5 stress set** (`PENDING_HUMAN.md` A1). 120 code-switched sentences rich in critical words
  (`stress/sentences.tsv`), split 40 per reader: Rui Cong, Phoebe, Bridgette. Each reader has 20 English-main
  (`en…`, embedded Mandarin) and 20 Mandarin-main (`zh…`, embedded English) sentences. One take per sentence.
- Recorded on each reader's phone (quiet room, 20–30 cm, built-in mic, 1 s silence before and after), following
  `team_tasks/1_recordings/HOW_TO_RECORD.md`.
- All 120 file IDs match `stress/assignments.tsv`. Total 10.7 min (207 / 208 / 225 s per reader; clips 3.8–7.2 s).
- **References are the prompt sentences**, not verbatim transcripts (decided with Nibras, 7 Nov). The text in
  `stress/sentences.tsv` is identical to the team's sentence lists (checked 120/120).

## 2. Pipeline (identical to the earlier experiments)
Same as the synthetic stress set (D5, `scripts/queue_d5.sh`) and the main test sets:
- `scripts/prepare_stress.py` (unchanged): mean to mono, librosa resample to 16 kHz, FLAC clips, manifest with
  `normalize()` references and `label()` main language. Set name `stress_test` (already in `csasr/data/manifest.py`).
- No chunking or VAD (every clip < 30 s, one Whisper window), greedy decoding, `--lang free` (HF/CT2 restrict to
  {zh, en}; whisper.cpp `-l auto`), as for every main row.
- Scoring with `scripts/score.py` / `csasr/score/report.py` (unchanged): MER, sMER, CWER (a) rule tagger and
  (b) adjudicated D2 list (`CSASR_CRITICAL_LABELS`), embedded error E, matrix-at-switch Mₛ, other matrix Mᵢ,
  switch-point error, translation and transliteration rates; 1,000-resample utterance bootstrap.
- R = `R.fragility` (16-bit → quantized, same backend reference for CPU formats), Rec = `R.recovery`, with the
  pre-registered reporting rule: R only where the matrix-change CI excludes 0; Rec only where the 4-bit embedded
  loss CI excludes 0.

## 3. Runs (set `stress_test`, 120 utterances; method suffix `_human`)
| # | Systems | Precisions | Where | What it shows |
|---|---|---|---|---|
| H1 | `small` off-the-shelf | fp16 | GPU | baseline; does the off-the-shelf embedded-word gap hold on real voices |
| H2 | `ft_tiny`, `ft_base`, `ft_small`, `full_base`, `full_small` | fp16, int8_bnb, nf4_bnb | GPU | 16-bit accuracy by model size; simulated 8/4-bit and R |
| H3 | same five | ct2_f32, ct2_int8, ggml_f16, ggml_q8_0, ggml_q4_0 | smu2 CPU | deployed formats; R against the same backend's 16/32-bit run |
| H4 (secondary) | ft_tiny, full_base repairs (mixed precision, QLoRA NF4, QLoRA merged → Q4_0) | as in table 14 | GPU / CPU | whether the NF4 repairs carry over to real speech |

That is 41 main decodes (+ H4). Each is 120 short clips, minutes per run.

**GPU use.** `nvidia-smi` before launch and before each batch; only GPUs with no process and < 2 GB used
(on 7 Nov: GPUs 5 and 7). Queue `scripts/queue_h.sh` in tmux session `h_human`, log `logs/queue_h_20261107.log`.

## 4. Outputs (all new files)
| What | Where (smu2 `~/cs425-codeswitch-asr`) |
|---|---|
| original m4a | `data/stress/audio_m4a/<reader>/` |
| converted wav (48 kHz PCM, ffmpeg) | `data/stress/audio/<reader>/<id>.wav` |
| 16 kHz clips + manifest | `data/stress/clips/`, `data/stress/manifests/test.jsonl` |
| predictions | `preds/<model>__<precision>__<method>_human.jsonl` (synthetic ones are `*_stress`, untouched) |
| raw scores | `results/d5h_*.csv` |
| tables | `results/tables/27_stress_set_HUMAN.csv`, `27b_human_fragility_R.csv`, `27c_human_vs_synthetic.csv` |
| figure | `results/figures/fig8_human_stress.{png,svg,csv}` |

Copied to the team folder `results/` and `paper/figures/`.

## 5. Analysis
1. 16-bit, every system: human vs synthetic clean / phone vs ASCEND and NSC test.
2. By language: Mandarin-main sentences (embedded English, like ASCEND) vs English-main (embedded Mandarin, like NSC).
3. Around switch points: Mₛ vs Mᵢ, switch-point error.
4. Quantization: ΔE, ΔM and R per model × format × slice. Does "8-bit safe, 4-bit hurts embedded words in tiny/base,
   small robust" hold on real voices?
5. Per reader (descriptive only) and a list of clips where every 16-bit model agrees on the same non-reference word
   (possible reading deviation; for a teammate to listen to).

## 6. Things that don't fit the existing setup
1. **Format.** `.m4a`, not `.wav`: Phoebe and Bridgette AAC stereo 48 kHz (lossy); Rui Cong ALAC mono 48 kHz
   (lossless). `prepare_stress.py` reads `*.wav` only, so files are converted with ffmpeg first. Codec and phone are
   confounded with the reader.
2. **Folder names** (`bridge-`, `phoebe-`, `ruicong-cs425-recordings`) are mapped to `bridgette`, `phoebe`, `rui_cong`.
3. **Silence.** About 1 s of silence at each end, by instruction. Not trimmed (the pipeline doesn't trim other sets);
   insertions are reported since Whisper can hallucinate on silence.
4. **References are prompts**, not transcripts of what was said. A reader's slip counts as a model error.
5. **Design.** 3 readers, each reading different sentences, one take: reader and sentence are confounded. The
   synthetic set had 4 voices reading all 120. A speaker-cluster CI is meaningless with 3 readers, so CIs are
   utterance bootstraps and are optimistic (as for ASCEND). About a quarter as many embedded units as the
   synthetic set, so intervals are wider and R will often not be reportable.
6. **Numbers.** Sentences write numbers as English words ("twelve", "three p m"); the normalizer turns Chinese
   numerals into digits but leaves English number words. A model writing "12" is an error. Same text as the
   synthetic set, so comparisons are fair; the normalizer is not changed.
7. **Mandarin tokenization** is unchanged (one unit per Han character, jieba only for main-language labels); the
   sentences are Simplified, so OpenCC does nothing.
8. **Reader metadata** (L1, accent, phone model) is not in the repo; it must come from the readers.
