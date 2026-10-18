# RQ1: role or language?

## small__fp16__offshelf_free

### Unit error (substitution/deletion) by role, code-switched utterances

| set         | main   | role   |   err_% |   units |
|:------------|:-------|:-------|--------:|--------:|
| ascend_test | en     | E      |    55.7 |     149 |
| ascend_test | en     | Mi     |    36.3 |     190 |
| ascend_test | en     | Ms     |    49.1 |      57 |
| ascend_test | zh     | E      |    42.5 |     623 |
| ascend_test | zh     | Mi     |    15.5 |    3600 |
| ascend_test | zh     | Ms     |    19.6 |     690 |
| nsc_test    | en     | E      |    55.6 |    7935 |
| nsc_test    | en     | Mi     |    59.2 |   15694 |
| nsc_test    | en     | Ms     |    71.7 |    3413 |
| nsc_test    | zh     | E      |    85.1 |    3107 |
| nsc_test    | zh     | Mi     |    31.5 |    8786 |
| nsc_test    | zh     | Ms     |    37.7 |    2672 |

### Same word, embedded vs matrix

| word_language   |   word_types |   embedded_tokens |   matrix_tokens |   err_embedded |   err_matrix | diff              |
|:----------------|-------------:|------------------:|----------------:|---------------:|-------------:|:------------------|
| en              |          725 |              3094 |           15637 |           81.5 |         66.1 | 15.4 [12.1, 18.6] |
| zh              |          559 |              7793 |           14992 |           54.5 |         28.6 | 26.0 [23.9, 27.8] |

### Logistic regression (A5), 46916 units, 122 speakers, speaker-bootstrap CIs (B=175); covariates: emb, en, emb_en, logf, dur, nsc

| term   |   coef | ci               |
|:-------|-------:|:-----------------|
| const  | -1.504 | [-1.687, -1.262] |
| emb    |  0.935 | [0.780, 1.152]   |
| en     |  1.175 | [0.949, 1.425]   |
| emb_en |  0.09  | [-0.273, 0.362]  |
| logf   | -0.027 | [-0.044, -0.013] |
| dur    | -0.488 | [-0.739, -0.302] |
| nsc    |  1.073 | [0.875, 1.226]   |

## base__fp16__offshelf_free

### Unit error (substitution/deletion) by role, code-switched utterances

| set         | main   | role   |   err_% |   units |
|:------------|:-------|:-------|--------:|--------:|
| ascend_test | en     | E      |    63.8 |     149 |
| ascend_test | en     | Mi     |    34.7 |     190 |
| ascend_test | en     | Ms     |    54.4 |      57 |
| ascend_test | zh     | E      |    59.9 |     623 |
| ascend_test | zh     | Mi     |    25.2 |    3600 |
| ascend_test | zh     | Ms     |    31.2 |     690 |
| nsc_test    | en     | E      |    72.1 |    7935 |
| nsc_test    | en     | Mi     |    60.1 |   15694 |
| nsc_test    | en     | Ms     |    73.5 |    3413 |
| nsc_test    | zh     | E      |    90.5 |    3107 |
| nsc_test    | zh     | Mi     |    42.2 |    8786 |
| nsc_test    | zh     | Ms     |    49.4 |    2672 |

### Same word, embedded vs matrix

| word_language   |   word_types |   embedded_tokens |   matrix_tokens |   err_embedded |   err_matrix | diff              |
|:----------------|-------------:|------------------:|----------------:|---------------:|-------------:|:------------------|
| en              |          725 |              3094 |           15637 |           86.9 |         67.3 | 19.7 [16.7, 22.5] |
| zh              |          559 |              7793 |           14992 |           71.1 |         39.1 | 32.0 [30.5, 33.4] |

### Logistic regression (A5), 46916 units, 122 speakers, speaker-bootstrap CIs (B=175); covariates: emb, en, emb_en, logf, dur, nsc

| term   |   coef | ci               |
|:-------|-------:|:-----------------|
| const  | -0.574 | [-0.728, -0.351] |
| emb    |  1.185 | [1.024, 1.373]   |
| en     |  0.717 | [0.490, 0.968]   |
| emb_en |  0.232 | [-0.131, 0.596]  |
| logf   | -0.07  | [-0.086, -0.057] |
| dur    | -0.483 | [-0.714, -0.282] |
| nsc    |  0.925 | [0.719, 1.075]   |

## ft_base__fp16__ft

### Unit error (substitution/deletion) by role, code-switched utterances

| set         | main   | role   |   err_% |   units |
|:------------|:-------|:-------|--------:|--------:|
| ascend_test | en     | E      |    23.5 |     149 |
| ascend_test | en     | Mi     |    21.1 |     190 |
| ascend_test | en     | Ms     |    28.1 |      57 |
| ascend_test | zh     | E      |    28.9 |     623 |
| ascend_test | zh     | Mi     |    12   |    3600 |
| ascend_test | zh     | Ms     |    14.3 |     690 |
| nsc_test    | en     | E      |    21.4 |    7935 |
| nsc_test    | en     | Mi     |    17.6 |   15694 |
| nsc_test    | en     | Ms     |    23.2 |    3413 |
| nsc_test    | zh     | E      |    31.2 |    3107 |
| nsc_test    | zh     | Mi     |    21.5 |    8786 |
| nsc_test    | zh     | Ms     |    22.4 |    2672 |

### Same word, embedded vs matrix

| word_language   |   word_types |   embedded_tokens |   matrix_tokens |   err_embedded |   err_matrix | diff            |
|:----------------|-------------:|------------------:|----------------:|---------------:|-------------:|:----------------|
| en              |          725 |              3094 |           15637 |           26.7 |         18.8 | 7.9 [5.9, 10.6] |
| zh              |          559 |              7793 |           14992 |           19.6 |         18.3 | 1.3 [0.2, 2.5]  |

### Logistic regression (A5), 46916 units, 122 speakers, speaker-bootstrap CIs (B=175); covariates: emb, en, emb_en, logf, dur, nsc

| term   |   coef | ci               |
|:-------|-------:|:-----------------|
| const  |  0.428 | [0.177, 0.778]   |
| emb    |  0.01  | [-0.144, 0.214]  |
| en     | -0.213 | [-0.409, 0.038]  |
| emb_en |  0.647 | [0.349, 0.939]   |
| logf   | -0.285 | [-0.314, -0.262] |
| dur    | -2.45  | [-3.096, -1.840] |
| nsc    |  0.569 | [0.194, 0.847]   |

## ft_small__fp16__ft

### Unit error (substitution/deletion) by role, code-switched utterances

| set         | main   | role   |   err_% |   units |
|:------------|:-------|:-------|--------:|--------:|
| ascend_test | en     | E      |    20.1 |     149 |
| ascend_test | en     | Mi     |    17.4 |     190 |
| ascend_test | en     | Ms     |    17.5 |      57 |
| ascend_test | zh     | E      |    22.6 |     623 |
| ascend_test | zh     | Mi     |     8.2 |    3600 |
| ascend_test | zh     | Ms     |    10.7 |     690 |
| nsc_test    | en     | E      |    16.5 |    7935 |
| nsc_test    | en     | Mi     |    13.7 |   15694 |
| nsc_test    | en     | Ms     |    18.8 |    3413 |
| nsc_test    | zh     | E      |    26.3 |    3107 |
| nsc_test    | zh     | Mi     |    17.9 |    8786 |
| nsc_test    | zh     | Ms     |    18.6 |    2672 |

### Same word, embedded vs matrix

| word_language   |   word_types |   embedded_tokens |   matrix_tokens |   err_embedded |   err_matrix | diff            |
|:----------------|-------------:|------------------:|----------------:|---------------:|-------------:|:----------------|
| en              |          725 |              3094 |           15637 |           23.1 |         15.7 | 7.4 [5.2, 10.1] |
| zh              |          559 |              7793 |           14992 |           15.2 |         14.9 | 0.3 [-0.8, 1.3] |

### Logistic regression (A5), 46916 units, 122 speakers, speaker-bootstrap CIs (B=175); covariates: emb, en, emb_en, logf, dur, nsc

| term   |   coef | ci               |
|:-------|-------:|:-----------------|
| const  | -0.18  | [-0.514, 0.200]  |
| emb    | -0.07  | [-0.219, 0.171]  |
| en     | -0.228 | [-0.454, 0.091]  |
| emb_en |  0.793 | [0.417, 1.111]   |
| logf   | -0.248 | [-0.271, -0.225] |
| dur    | -2.443 | [-3.042, -1.834] |
| nsc    |  0.672 | [0.263, 0.967]   |
