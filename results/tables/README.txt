CS425 results tables (exported 2026-10-25 18:52)
All *_pct columns are percentages (21.35 = 21.35%). Test sets unless the name says val.
MER = (substitutions + deletions + insertions) / reference units; one unit = one Chinese
character, English word or digit string. Role rates charge insertions to error blocks, so
they can exceed 100%. Speed on the MacBook is in 13_speed_macbook.csv.
SYNTHETIC = text-to-speech audio, not human speech.

01_data_counts.csv
  Size of every split (proposal 3.1 size check).
    sentence_main_language: all / zh / en / balanced (tie in word counts)

02_accuracy_16bit_all_systems.csv
  Test accuracy of every system at 16-bit (greedy decoding unless re-ranked).
    system_id: <model>__<precision>__<method> (name of the prediction file)
    n_*: counts in the reference transcripts of that test set
    MER_pct: mixed error rate over all units
    embedded_unit_error_pct: error on embedded units (E)
    matrix_at_switch_error_pct: error on matrix units next to a switch (Ms)
    matrix_other_error_pct: error on other matrix units (Mi)
    switch_point_error_pct: error on all units next to a language change
    matrix_unit_error_codeswitched_pct: matrix (Ms+Mi) error on code-switched utterances
    translation_rate_pct: share of embedded units replaced by a translation
    semantic_MER_pct: MER with partial credit for correct translations (rho0=0.5)
    critical_word_error_pct: share of critical words not reproduced exactly
    transliteration_rate_pct: share of embedded units replaced by a sound-alike in the other script (C3, theta=0.1)
    *_ci95_low_pct / *_ci95_high_pct: 95% bootstrap confidence interval (1000 utterance resamples)

03_methods_vs_plain_finetuning.csv
  Paired bootstrap difference (method minus plain fine-tuning) on the same utterances. Negative = better.
    *_change_points: difference in percentage points
    *_change_significant: True if the 95% CI of the difference excludes 0

04_hyperparameter_grids_validation.csv
  Grid search on base, pooled ASCEND + NSC validation.
    rule (role-weighted): lowest embedded error among runs with MER <= reference MER + 1 point
    rule (semantic): lowest semantic MER among runs with critical-word error <= reference

05_accuracy_all_precisions.csv
  Test accuracy at every precision (CPU formats are on a fixed subset: all ASCEND test, 4,000 NSC test, 2,000 Diff Room utterances; see n_utterances).
    system_id: <model>__<precision>__<method> (name of the prediction file)
    n_*: counts in the reference transcripts of that test set
    MER_pct: mixed error rate over all units
    embedded_unit_error_pct: error on embedded units (E)
    matrix_at_switch_error_pct: error on matrix units next to a switch (Ms)
    matrix_other_error_pct: error on other matrix units (Mi)
    switch_point_error_pct: error on all units next to a language change
    matrix_unit_error_codeswitched_pct: matrix (Ms+Mi) error on code-switched utterances
    translation_rate_pct: share of embedded units replaced by a translation
    semantic_MER_pct: MER with partial credit for correct translations (rho0=0.5)
    critical_word_error_pct: share of critical words not reproduced exactly
    transliteration_rate_pct: share of embedded units replaced by a sound-alike in the other script (C3, theta=0.1)
    *_ci95_low_pct / *_ci95_high_pct: 95% bootstrap confidence interval (1000 utterance resamples)

06_fragility_ratio_R.csv
  R = (embedded error change) / (matrix error change) from 16-bit to quantized, code-switched utterances only.
    R > 1: quantization hurts embedded units more; < 1 less
    R_reported: False when the matrix change's CI includes 0 (R undefined/unstable); use the two changes instead
    compared_with_16bit: GPU rows vs HF 16-bit; CPU rows vs the same CPU backend at 16/32-bit

07_sensitivity_scan_per_block.csv
  A6: KL divergence on embedded-word tokens when only one block is NF4.
    rank: 1 = most sensitive
    val_*_only_this_block_4bit: confirmation decode for the top 5

08_mixed_precision_validation.csv
  NF4 with the top-k most sensitive blocks kept at 16-bit (validation).
    chosen: k picked on validation embedded error, then run once on test

09_language_id_head.csv
  Frame-level language ID on the frozen fine-tuned base encoder (20 ms frames).
    3class: en / zh / non-speech
    switch_boundary_error: distance from each true switch to the nearest predicted one

10_model_size_and_memory.csv
  Measured GPU memory footprint (bitsandbytes) and CPU model file sizes.

11_rq1_same_word_embedded_vs_matrix.csv
  Same word type: error when embedded vs when part of the main language.

12_rq1_logistic_regression.csv
  A5: logit P(unit error) on role, language, frequency, length, corpus.
    emb: embedded-unit effect (b1) with language etc. held fixed
    en: English unit
    emb_en: interaction
    logf: log training frequency
    dur: spoken length (s)
    nsc: NSC corpus
    ci95: bootstrap over speakers

13_speed_macbook.csv
  CPU speed on the MacBook (plugged in, Low Power Mode off), 100 fixed test clips, each configuration in a fresh process after a warm-up run.
    real_time_factor: processing time / audio length (lower is faster; < 1 = faster than real time)
    peak_memory_RSS_MB: peak resident memory of the process
    note: the first-batch ft_tiny / ft_small / ft_base rows were re-measured (first batch kept in speed_first_batch_*.json, not used)

14_recovery_Rec.csv
  Rec = (Eq_emb - Eq,fix_emb) / (Eq_emb - E16_emb), code-switched utterances.
    Rec: 1 = fix removes the whole quantization loss, 0 = no help, < 0 = worse
    Rec_reportable: False when the quantization loss CI includes 0 (Rec is then not meaningful)
    fix: mixed precision (bitsandbytes NF4, k blocks at 16-bit chosen on val), QLoRA QAT with 16-bit adapters, or QAT merged then re-quantized (deployed rows use the same-backend 16-bit reference)

15_gated_reranking.csv
  C1/C2: re-ranking applied only when the LID head finds a minority-language run >= theta ms; theta in {0,200,400,800,inf} chosen on val (lowest MER, ties to larger).
    change_vs_greedy: paired difference vs the same model with greedy decoding (negative = better)
    lid_head: lid_ft_base = shared head; lid_full_* = the model's own head (C2)

16_speaker_cluster_ci.csv
  A3: NSC CIs from resampling whole speakers vs resampling utterances.
    speaker_ci95: use these for NSC claims; utterance CIs are too narrow when speakers repeat

17_stress_set_SYNTHETIC.csv
  D5: SYNTHETIC stress set - 120 sentences x 4 CosyVoice 2 zero-shot voices (FLEURS prompts, 2 en + 2 zh speakers). NOT human speech; the human recordings are still pending.
    channel: phone = 300-3400 Hz band-limit, 8 kHz round trip, synthetic RIR (RT60 0.3 s), 20 dB white noise
    CI: utterance bootstrap; the 4 voices are not independent speakers

18_partner_leakage_sensitivity.csv
  D3: MER with and without utterances flagged for audible partner speech.

18b_partner_leakage_rates.csv
  D3: % of test utterances per leakage label (rule piloted on val, borderline test clips judged by annotator 1).

19_alignment_checks.csv
  D4: forced-alignment quality without human labels.
    median_ms / mean_ms: absolute timing difference
    within_*: % of switches within 50/100 ms

20_label_agreement.csv
  D1-D3: agreement of the automated annotators (LLM annotators were fresh agents that saw only the rubric and the items).
    cohen_kappa: chance-corrected agreement

21_diff_room_training_ablation.csv
  E: does training on Diff Room audio help on Diff Room speech? Same ft_base recipe, no tuning; negative change = better than A.

22_switching_statistics.csv
  C4: how often and how long speakers switch, per corpus.

23_tau_sensitivity.csv
  D1: translation detector (CC-CEDICT or LaBSE >= tau) vs adjudicated labels.
    pair_*: each output pair counted once
    occ_*: weighted by occurrences across all systems
    flagged_occ_rel_to_tau0.65: approximate multiplier on conf_rate relative to the chosen tau

24_cwer_rule_vs_labels.csv
  D2: CWER with (a) the rule tagger vs (b) the adjudicated critical-word list (249 types + every digit string).

