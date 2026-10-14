# Quantization (RQ3)

## Accuracy by precision (test sets; cs_ = code-switched utterances only)

| model   | method        | precision   | set           | MER                  | cs_E_emb             | cs_E_mat             |   CWER |   size_mb |
|:--------|:--------------|:------------|:--------------|:---------------------|:---------------------|:---------------------|-------:|----------:|
| base    | offshelf_both | fp16        | ascend_test   | 101.7 [95.6, 108.2]  | 83.9 [64.8, 117.7]   | 109.4 [100.9, 119.1] |   85.2 |     138.5 |
| base    | offshelf_both | fp16        | nsc_test      | 80.0 [77.7, 82.4]    | 115.0 [108.8, 121.4] | 88.4 [83.4, 93.2]    |   86.7 |     138.5 |
| base    | offshelf_both | fp16        | nsc_diff_test | 77.7 [74.1, 81.9]    | 120.5 [112.3, 130.0] | 93.5 [86.4, 101.8]   |   78.2 |     138.5 |
| base    | offshelf_en   | fp16        | ascend_test   | 94.1 [90.7, 98.1]    | 66.4 [60.1, 73.0]    | 104.8 [98.9, 112.1]  |   84   |     138.5 |
| base    | offshelf_en   | fp16        | nsc_test      | 70.2 [68.7, 71.9]    | 103.2 [99.8, 107.2]  | 75.9 [72.8, 79.3]    |   84.8 |     138.5 |
| base    | offshelf_en   | fp16        | nsc_diff_test | 67.0 [64.5, 69.8]    | 106.1 [100.0, 112.2] | 76.9 [72.5, 81.7]    |   73.6 |     138.5 |
| base    | offshelf_free | fp16        | ascend_test   | 35.8 [34.0, 37.8]    | 83.1 [71.2, 98.3]    | 29.6 [27.1, 32.6]    |   53.1 |     138.5 |
| base    | offshelf_free | fp16        | nsc_test      | 71.1 [68.9, 73.4]    | 101.8 [96.1, 107.5]  | 75.5 [71.2, 80.1]    |   78.6 |     138.5 |
| base    | offshelf_free | fp16        | nsc_diff_test | 64.2 [60.9, 67.7]    | 103.1 [90.6, 115.7]  | 65.2 [60.1, 70.7]    |   69.2 |     138.5 |
| base    | offshelf_free | ct2_int8    | ascend_test   | 37.3 [35.0, 39.9]    | 83.6 [74.3, 93.4]    | 30.7 [27.7, 34.3]    |   49.4 |      73.9 |
| base    | offshelf_free | ct2_int8    | nsc_test      | 64.0 [62.3, 65.9]    | 89.9 [85.8, 93.9]    | 63.1 [59.2, 66.8]    |   81   |      73.9 |
| base    | offshelf_free | ct2_int8    | nsc_diff_test | 56.8 [54.5, 59.3]    | 83.6 [77.9, 90.3]    | 57.3 [53.0, 61.5]    |   72.5 |      73.9 |
| base    | offshelf_free | ggml_q8_0   | ascend_test   | 36.2 [34.6, 37.9]    | 78.2 [70.0, 86.4]    | 29.4 [26.9, 32.4]    |   49.4 |      78   |
| base    | offshelf_free | ggml_q8_0   | nsc_test      | 81.6 [80.4, 82.8]    | 89.0 [86.2, 92.0]    | 74.6 [71.6, 77.1]    |   87.4 |      78   |
| base    | offshelf_free | ggml_q8_0   | nsc_diff_test | 76.6 [74.7, 78.4]    | 83.8 [79.5, 88.0]    | 70.7 [67.1, 74.5]    |   87.3 |      78   |
| base    | offshelf_free | ggml_q5_0   | ascend_test   | 37.1 [35.4, 39.0]    | 81.3 [72.3, 90.1]    | 30.3 [27.6, 33.4]    |   58   |      52.7 |
| base    | offshelf_free | ggml_q5_0   | nsc_test      | 79.8 [78.6, 81.0]    | 87.5 [84.4, 90.5]    | 73.1 [70.3, 75.7]    |   86.7 |      52.7 |
| base    | offshelf_free | ggml_q5_0   | nsc_diff_test | 73.9 [71.9, 75.8]    | 83.6 [79.4, 88.3]    | 66.5 [62.4, 70.7]    |   84.3 |      52.7 |
| base    | offshelf_free | ggml_q4_0   | ascend_test   | 39.7 [37.9, 41.7]    | 92.4 [84.2, 100.9]   | 33.2 [30.3, 36.6]    |   54.3 |      44.3 |
| base    | offshelf_free | ggml_q4_0   | nsc_test      | 82.6 [81.2, 84.2]    | 95.7 [90.1, 104.3]   | 77.0 [74.0, 80.1]    |   89.1 |      44.3 |
| base    | offshelf_free | ggml_q4_0   | nsc_diff_test | 77.5 [75.2, 80.0]    | 89.2 [84.5, 93.9]    | 71.6 [66.8, 76.6]    |   87   |      44.3 |
| base    | offshelf_zh   | fp16        | ascend_test   | 49.9 [44.0, 56.7]    | 101.9 [80.2, 127.8]  | 41.6 [30.3, 56.0]    |   59.3 |     138.5 |
| base    | offshelf_zh   | fp16        | nsc_test      | 140.4 [136.6, 144.4] | 117.0 [109.4, 124.4] | 119.9 [113.6, 126.5] |   88.5 |     138.5 |
| base    | offshelf_zh   | fp16        | nsc_diff_test | 134.3 [128.7, 140.3] | 116.2 [101.4, 132.2] | 106.5 [98.1, 115.3]  |   87.7 |     138.5 |
| tiny    | offshelf_both | fp16        | ascend_test   | 65.2 [60.3, 70.8]    | 108.4 [88.5, 131.4]  | 60.7 [53.4, 69.3]    |   72.8 |      72   |
| tiny    | offshelf_both | fp16        | nsc_test      | 81.4 [79.5, 83.4]    | 106.4 [102.1, 111.2] | 87.4 [83.3, 91.5]    |   87.3 |      72   |
| tiny    | offshelf_both | fp16        | nsc_diff_test | 75.9 [73.0, 79.2]    | 106.3 [99.2, 115.1]  | 83.9 [78.3, 90.3]    |   78.7 |      72   |
| tiny    | offshelf_en   | fp16        | ascend_test   | 99.8 [94.4, 105.5]   | 84.0 [71.0, 101.7]   | 103.6 [98.1, 110.7]  |   88.9 |      72   |
| tiny    | offshelf_en   | fp16        | nsc_test      | 78.4 [76.8, 80.0]    | 106.0 [102.8, 109.5] | 82.6 [79.7, 85.8]    |   87.4 |      72   |
| tiny    | offshelf_en   | fp16        | nsc_diff_test | 72.9 [70.4, 75.6]    | 103.3 [99.1, 108.3]  | 84.1 [79.2, 89.5]    |   80.9 |      72   |
| tiny    | offshelf_zh   | fp16        | ascend_test   | 82.0 [70.1, 95.7]    | 127.9 [114.8, 144.6] | 58.2 [42.2, 77.8]    |   66.7 |      72   |
| tiny    | offshelf_zh   | fp16        | nsc_test      | 184.6 [178.9, 190.4] | 157.5 [145.5, 168.8] | 156.0 [148.0, 163.6] |   93.9 |      72   |
| tiny    | offshelf_zh   | fp16        | nsc_diff_test | 197.1 [188.0, 205.2] | 158.7 [142.2, 176.4] | 153.7 [141.3, 165.8] |   92.9 |      72   |

## Fragility ratio R = (Eq_emb - E16_emb) / (Eq_mat - E16_mat)

d_emb / d_mat: change in embedded / matrix error (points) vs the same model at 16-bit, 95% CIs. R is reported only when the CI of d_mat excludes 0.

| model   | method        | precision   | set           | main   | d_emb                | d_mat               | R                                         |
|:--------|:--------------|:------------|:--------------|:-------|:---------------------|:--------------------|:------------------------------------------|
| base    | offshelf_free | ct2_int8    | ascend_test   | zh     | -1.9 [-17.6, 9.5]    | 1.0 [-0.6, 3.2]     | not reported (matrix change within noise) |
| base    | offshelf_free | ct2_int8    | nsc_test      | en     | -20.9 [-35.1, -8.8]  | -15.4 [-22.8, -8.7] | 1.35 [0.59, 2.52]                         |
| base    | offshelf_free | ct2_int8    | nsc_test      | zh     | -1.5 [-5.1, 1.9]     | -9.8 [-22.4, -0.4]  | 0.15 [-0.60, 1.10]                        |
| base    | offshelf_free | ct2_int8    | nsc_diff_test | en     | -19.4 [-41.6, -4.2]  | -6.0 [-11.7, -0.9]  | 3.23 [0.71, 15.76]                        |
| base    | offshelf_free | ggml_q8_0   | ascend_test   | zh     | -6.1 [-20.3, 2.2]    | -0.1 [-1.1, 1.0]    | not reported (matrix change within noise) |
| base    | offshelf_free | ggml_q8_0   | nsc_test      | en     | -21.6 [-36.9, -8.6]  | 3.1 [-8.1, 12.6]    | not reported (matrix change within noise) |
| base    | offshelf_free | ggml_q8_0   | nsc_test      | zh     | -2.7 [-6.4, 0.5]     | -10.4 [-22.5, -0.4] | 0.26 [-0.19, 1.89]                        |
| base    | offshelf_free | ggml_q8_0   | nsc_diff_test | en     | -19.2 [-43.5, -1.4]  | 15.3 [5.0, 23.7]    | -1.25 [-5.13, -0.02]                      |
| base    | offshelf_free | ggml_q5_0   | ascend_test   | zh     | -2.7 [-16.3, 6.2]    | 0.5 [-0.8, 1.8]     | not reported (matrix change within noise) |
| base    | offshelf_free | ggml_q5_0   | nsc_test      | en     | -23.7 [-38.7, -10.1] | 0.8 [-10.0, 10.1]   | not reported (matrix change within noise) |
| base    | offshelf_free | ggml_q5_0   | nsc_test      | zh     | -2.8 [-7.2, 1.1]     | -10.6 [-23.1, -0.5] | 0.26 [-0.27, 1.62]                        |
| base    | offshelf_free | ggml_q5_0   | nsc_diff_test | en     | -19.0 [-40.5, -2.9]  | 8.5 [-1.5, 17.0]    | not reported (matrix change within noise) |
| base    | offshelf_free | ggml_q4_0   | ascend_test   | zh     | 5.8 [-8.7, 17.4]     | 3.3 [1.5, 5.1]      | 1.79 [-3.12, 7.34]                        |
| base    | offshelf_free | ggml_q4_0   | nsc_test      | en     | -17.3 [-32.5, -4.0]  | 2.6 [-8.9, 12.6]    | not reported (matrix change within noise) |
| base    | offshelf_free | ggml_q4_0   | nsc_test      | zh     | 10.4 [-5.2, 35.9]    | -3.0 [-15.5, 6.9]   | not reported (matrix change within noise) |
| base    | offshelf_free | ggml_q4_0   | nsc_diff_test | en     | -12.7 [-34.0, 4.0]   | 12.6 [1.2, 22.5]    | -1.01 [-7.50, 0.49]                       |

## Recovery Rec = (Eq_emb - Eq,fix_emb) / (Eq_emb - E16_emb)

loss: embedded-error increase from quantization (points). Rec is reported with a CI only when the loss CI excludes 0.

_not run yet_

## Sensitivity scan (A6)

_not run yet_
