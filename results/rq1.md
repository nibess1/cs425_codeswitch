# RQ1: role or language?

## tiny__fp16__offshelf_zh

### Unit error (substitution/deletion) by role, code-switched utterances

| set         | main   | role   |   err_% |   units |
|:------------|:-------|:-------|--------:|--------:|
| ascend_test | en     | E      |    55   |     149 |
| ascend_test | en     | Mi     |    95.8 |     190 |
| ascend_test | en     | Ms     |    91.2 |      57 |
| ascend_test | zh     | E      |    83.6 |     623 |
| ascend_test | zh     | Mi     |    32   |    3600 |
| ascend_test | zh     | Ms     |    36.1 |     690 |
| nsc_test    | en     | E      |    70.7 |    7935 |
| nsc_test    | en     | Mi     |    98.3 |   15694 |
| nsc_test    | en     | Ms     |    97.8 |    3413 |
| nsc_test    | zh     | E      |    97.4 |    3107 |
| nsc_test    | zh     | Mi     |    53.6 |    8786 |
| nsc_test    | zh     | Ms     |    56.1 |    2672 |

### Same word, embedded vs matrix

| word_language   |   word_types |   embedded_tokens |   matrix_tokens |   err_embedded |   err_matrix | diff              |
|:----------------|-------------:|------------------:|----------------:|---------------:|-------------:|:------------------|
| en              |          725 |              3094 |           15637 |           96   |         98   | -2.0 [-3.0, -1.0] |
| zh              |          559 |              7793 |           14992 |           69.2 |         48.3 | 20.9 [19.0, 23.0] |

### Logistic regression (A5), 46916 units, 122 speakers, speaker-bootstrap CIs (B=175); covariates: emb, en, emb_en, logf, nsc

| term   |   coef | ci               |
|:-------|-------:|:-----------------|
| const  |  0.338 | [0.100, 0.731]   |
| emb    |  0.709 | [0.558, 0.898]   |
| en     |  3.772 | [3.497, 4.141]   |
| emb_en | -1.691 | [-2.357, -1.035] |
| logf   | -0.155 | [-0.176, -0.138] |
| nsc    |  0.992 | [0.659, 1.228]   |
