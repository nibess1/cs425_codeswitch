CS425 results tables (exported 2026-10-15 10:05)
All *_pct columns are percentages (21.35 = 21.35%). Test sets unless the name says val.
MER = (substitutions + deletions + insertions) / reference units; one unit = one Chinese
character, English word or digit string. Role rates charge insertions to error blocks, so
they can exceed 100%. Speed on the MacBook is in 13_speed_macbook.csv.

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

