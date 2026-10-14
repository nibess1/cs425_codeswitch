# Training and decoding methods

## Validation (checkpoint selection, best step by mean val MER)

| run               | model   |   alpha |   beta |   lambda_sem |   rho0 |   best_step |   val_mean_MER |   ascend_val_MER |   nsc_val_MER |   ascend_val_E_emb |   nsc_val_E_emb |
|:------------------|:--------|--------:|-------:|-------------:|-------:|------------:|---------------:|-----------------:|--------------:|-------------------:|----------------:|
| ft_base           | base    |     0   |    0   |            0 |    0.5 |        4269 |           24.9 |             22.8 |          26.9 |               46.3 |            34.4 |
| ft_small          | small   |     0   |    0   |            0 |    0.5 |        2000 |           18.8 |             15.4 |          22.1 |               38.1 |            28.7 |
| ft_tiny           | tiny    |     0   |    0   |            0 |    0.5 |        4269 |           31.1 |             28.5 |          33.7 |               54.4 |            39.5 |
| rw_base_a0.5_b0.5 | base    |     0.5 |    0.5 |            0 |    0.5 |           0 |           49.5 |             35.9 |          63.1 |               87   |            97   |
| rw_base_a0.5_b0   | base    |     0.5 |    0   |            0 |    0.5 |        4269 |           25.6 |             22.7 |          28.5 |               44.3 |            41.1 |

## Test: paired bootstrap difference vs plain fine-tuning of the same size (percentage points; * = 95% CI excludes 0; negative = better)

_not run yet_

## Critical-term flags (precision / recall against real critical-word errors)

_not run yet_
