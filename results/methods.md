# Training and decoding methods

## Validation (checkpoint selection, best step by mean val MER)

| run                | model   |   alpha |   beta |   lambda_sem |   rho0 |   best_step |   val_mean_MER |   ascend_val_MER |   nsc_val_MER |   ascend_val_E_emb |   nsc_val_E_emb |
|:-------------------|:--------|--------:|-------:|-------------:|-------:|------------:|---------------:|-----------------:|--------------:|-------------------:|----------------:|
| abl_base_semonly   | base    |     0   |    0   |          0.1 |    0.5 |        4269 |           24   |             21.1 |          26.8 |               44.5 |            34.5 |
| ft_base            | base    |     0   |    0   |          0   |    0.5 |        4269 |           24.9 |             22.8 |          26.9 |               46.3 |            34.4 |
| ft_small           | small   |     0   |    0   |          0   |    0.5 |        2000 |           18.8 |             15.4 |          22.1 |               38.1 |            28.7 |
| ft_tiny            | tiny    |     0   |    0   |          0   |    0.5 |        4269 |           31.1 |             28.5 |          33.7 |               54.4 |            39.5 |
| full_small         | small   |     0.5 |    0.5 |          0.1 |    0.5 |        3000 |           18.2 |             14.6 |          21.8 |               38.8 |            25   |
| rw_base_a0.5_b0.5  | base    |     0.5 |    0.5 |          0   |    0.5 |        4000 |           25   |             23   |          27   |               46   |            32.5 |
| rw_base_a0.5_b0    | base    |     0.5 |    0   |          0   |    0.5 |        4269 |           25.6 |             22.7 |          28.5 |               44.3 |            41.1 |
| rw_base_a1_b0.5    | base    |     1   |    0.5 |          0   |    0.5 |        4269 |           25.6 |             22.6 |          28.6 |               45   |            34   |
| rw_base_a1_b0      | base    |     1   |    0   |          0   |    0.5 |        3000 |           25.6 |             21.5 |          29.6 |               43.3 |            35.2 |
| rw_base_a2_b0.5    | base    |     2   |    0.5 |          0   |    0.5 |        4269 |           25.1 |             23   |          27.3 |               40.8 |            32.1 |
| rw_base_a2_b0      | base    |     2   |    0   |          0   |    0.5 |        4269 |           25   |             22.9 |          27.1 |               42.9 |            33   |
| sem_base_l0.1_r0.2 | base    |     0.5 |    0.5 |          0.1 |    0.2 |        4269 |           23.8 |             21   |          26.7 |               44.3 |            32.1 |
| sem_base_l0.1_r0.5 | base    |     0.5 |    0.5 |          0.1 |    0.5 |        4269 |           24   |             21.1 |          26.9 |               43.8 |            32.3 |
| sem_base_l0.3_r0.2 | base    |     0.5 |    0.5 |          0.3 |    0.2 |        3000 |           23.8 |             21.1 |          26.5 |               45.5 |            30.4 |
| sem_base_l0.3_r0.5 | base    |     0.5 |    0.5 |          0.3 |    0.5 |        4269 |           23.8 |             20.8 |          26.8 |               41.7 |            32.2 |

## Test: paired bootstrap difference vs plain fine-tuning of the same size (percentage points; * = 95% CI excludes 0; negative = better)

| system                                   | vs       | set           | MER                      | E_E                    | cs_E_emb               | conf_rate           | sMER                     | CWER                   |
|:-----------------------------------------|:---------|:--------------|:-------------------------|:-----------------------|:-----------------------|:--------------------|:-------------------------|:-----------------------|
| ediff_base eabl                          | ft_base  | ascend_test   | -1.1 [-3.3, +0.5]        | -12.3 [-43.4, +5.5]    | -12.3 [-43.4, +5.5]    | -0.1 [-0.6, +0.3]   | -1.1 [-3.2, +0.5]        | -2.5 [-9.1, +3.8]      |
| ediff_base eabl                          | ft_base  | nsc_test      | +3.9 [+2.0, +5.8] *      | +4.7 [-0.2, +10.2]     | +4.7 [-0.2, +10.2]     | +0.1 [-0.1, +0.3]   | +3.9 [+2.0, +5.8] *      | +4.1 [+2.2, +6.0] *    |
| emix_base eabl                           | ft_base  | ascend_test   | -1.0 [-3.2, +0.4]        | -12.8 [-43.7, +5.0]    | -12.8 [-43.7, +5.0]    | +0.3 [+0.0, +0.7]   | -1.0 [-3.2, +0.4]        | -1.2 [-9.0, +5.9]      |
| emix_base eabl                           | ft_base  | nsc_test      | -0.1 [-1.2, +1.0]        | +0.0 [-3.4, +3.6]      | +0.0 [-3.4, +3.6]      | -0.0 [-0.2, +0.1]   | -0.1 [-1.2, +1.0]        | +1.4 [-0.1, +3.0]      |
| esame_base eabl                          | ft_base  | ascend_test   | +0.0 [+0.0, +0.0]        | +0.0 [+0.0, +0.0]      | +0.0 [+0.0, +0.0]      | +0.0 [+0.0, +0.0]   | +0.0 [+0.0, +0.0]        | +0.0 [+0.0, +0.0]      |
| esame_base eabl                          | ft_base  | nsc_test      | +0.0 [+0.0, +0.0]        | +0.0 [+0.0, +0.0]      | +0.0 [+0.0, +0.0]      | +0.0 [+0.0, +0.0]   | +0.0 [+0.0, +0.0]        | +0.0 [+0.0, +0.0]      |
| full_base full                           | ft_base  | ascend_test   | -1.0 [-3.2, +0.4]        | -16.1 [-46.1, +0.4]    | -16.1 [-46.1, +0.4]    | -0.1 [-0.4, +0.0]   | -1.0 [-3.2, +0.4]        | -3.7 [-9.6, +1.2]      |
| full_base full                           | ft_base  | nsc_test      | -2.3 [-3.5, -1.4] *      | -3.6 [-6.5, -1.5] *    | -3.6 [-6.5, -1.5] *    | -0.0 [-0.1, +0.1]   | -2.3 [-3.5, -1.4] *      | +0.2 [-1.1, +1.4]      |
| full_base full                           | ft_base  | nsc_diff_test | -1.3 [-2.3, -0.4] *      | -1.4 [-2.3, -0.6] *    | -1.4 [-2.3, -0.6] *    | -0.1 [-0.3, +0.1]   | -1.3 [-2.3, -0.4] *      | -1.8 [-3.7, -0.1] *    |
| full_base rerank_lp                      | ft_base  | ascend_test   | -1.7 [-4.0, -0.1] *      | -19.1 [-49.9, -1.4] *  | -19.1 [-49.9, -1.4] *  | +0.6 [-0.3, +1.6]   | -1.7 [-4.0, -0.1] *      | -4.9 [-10.3, -1.1] *   |
| full_base rerank_lp                      | ft_base  | nsc_test      | -2.1 [-3.3, -1.1] *      | -3.8 [-6.9, -1.5] *    | -3.8 [-6.9, -1.5] *    | +0.1 [-0.1, +0.2]   | -2.1 [-3.3, -1.1] *      | +0.2 [-1.2, +1.7]      |
| full_base rerank_lp                      | ft_base  | nsc_diff_test | -0.6 [-1.7, +0.4]        | -2.2 [-3.2, -1.3] *    | -2.2 [-3.2, -1.3] *    | -0.1 [-0.3, +0.2]   | -0.6 [-1.7, +0.4]        | -2.4 [-4.5, -0.3] *    |
| full_base rerank_lp_filt                 | ft_base  | ascend_test   | -1.9 [-4.2, -0.5] *      | -20.3 [-51.2, -3.2] *  | -20.3 [-51.2, -3.2] *  | +0.6 [-0.3, +1.6]   | -1.9 [-4.2, -0.5] *      | -4.9 [-10.3, -1.1] *   |
| full_base rerank_lp_filt                 | ft_base  | nsc_test      | -2.2 [-3.4, -1.2] *      | -3.8 [-6.9, -1.5] *    | -3.8 [-6.9, -1.5] *    | +0.1 [-0.1, +0.2]   | -2.2 [-3.4, -1.2] *      | +0.2 [-1.2, +1.7]      |
| full_base rerank_lp_filt                 | ft_base  | nsc_diff_test | -0.6 [-1.7, +0.4]        | -2.2 [-3.2, -1.3] *    | -2.2 [-3.2, -1.3] *    | -0.1 [-0.3, +0.2]   | -0.6 [-1.7, +0.4]        | -2.4 [-4.5, -0.3] *    |
| full_base rerank_mms                     | ft_base  | ascend_test   | -0.0 [-2.4, +1.6]        | -21.2 [-52.5, -3.7] *  | -21.2 [-52.5, -3.7] *  | +0.6 [-0.4, +1.8]   | -0.0 [-2.4, +1.6]        | -7.4 [-13.7, -2.4] *   |
| full_base rerank_mms                     | ft_base  | nsc_test      | -2.1 [-3.2, -1.0] *      | -3.7 [-6.7, -1.5] *    | -3.7 [-6.7, -1.5] *    | +0.0 [-0.1, +0.2]   | -2.1 [-3.2, -1.0] *      | +0.3 [-0.9, +1.8]      |
| full_base rerank_mms                     | ft_base  | nsc_diff_test | -0.2 [-1.3, +0.8]        | -1.6 [-2.7, -0.5] *    | -1.6 [-2.7, -0.5] *    | +0.0 [-0.2, +0.2]   | -0.2 [-1.3, +0.8]        | -2.1 [-4.6, +0.3]      |
| full_base rerank_mms_filt                | ft_base  | ascend_test   | -0.0 [-2.4, +1.6]        | -21.2 [-52.5, -3.7] *  | -21.2 [-52.5, -3.7] *  | +0.6 [-0.4, +1.8]   | -0.0 [-2.4, +1.6]        | -7.4 [-13.7, -2.4] *   |
| full_base rerank_mms_filt                | ft_base  | nsc_test      | -2.1 [-3.3, -1.0] *      | -3.7 [-6.7, -1.4] *    | -3.7 [-6.7, -1.4] *    | +0.0 [-0.1, +0.2]   | -2.1 [-3.3, -1.0] *      | +0.3 [-0.9, +1.8]      |
| full_base rerank_mms_filt                | ft_base  | nsc_diff_test | -0.3 [-1.4, +0.8]        | -1.6 [-2.7, -0.5] *    | -1.6 [-2.7, -0.5] *    | +0.0 [-0.2, +0.2]   | -0.3 [-1.4, +0.8]        | -2.1 [-4.6, +0.3]      |
| full_base rerank_mms_filt_gated          | ft_base  | ascend_test   | +97.0 [+90.6, +103.9] *  | +46.5 [+14.1, +70.1] * | +46.5 [+14.1, +70.1] * | -1.3 [-2.2, -0.5] * | +97.1 [+90.7, +104.0] *  | +58.0 [+44.9, +71.1] * |
| full_base rerank_mms_filt_gated          | ft_base  | nsc_test      | -2.6 [-3.8, -1.6] *      | -3.9 [-6.9, -1.7] *    | -3.9 [-6.9, -1.7] *    | +0.0 [-0.1, +0.2]   | -2.7 [-3.8, -1.7] *      | +0.2 [-1.2, +1.5]      |
| full_base rerank_mms_filt_gated          | ft_base  | nsc_diff_test | -1.6 [-2.6, -0.7] *      | -2.3 [-3.4, -1.3] *    | -2.3 [-3.4, -1.3] *    | -0.1 [-0.3, +0.1]   | -1.6 [-2.6, -0.7] *      | -2.7 [-4.8, -0.4] *    |
| full_base rerank_ondevice                | ft_base  | ascend_test   | -1.4 [-3.7, +0.2]        | -19.1 [-49.5, -1.5] *  | -19.1 [-49.5, -1.5] *  | +0.9 [+0.0, +1.8]   | -1.4 [-3.8, +0.1]        | -4.9 [-10.3, -1.1] *   |
| full_base rerank_ondevice                | ft_base  | nsc_test      | -2.4 [-3.6, -1.4] *      | -3.9 [-7.0, -1.6] *    | -3.9 [-7.0, -1.6] *    | +0.1 [-0.1, +0.2]   | -2.4 [-3.6, -1.4] *      | +0.7 [-0.6, +2.0]      |
| full_base rerank_ondevice                | ft_base  | nsc_diff_test | -0.8 [-2.0, +0.2]        | -1.9 [-3.0, -1.0] *    | -1.9 [-3.0, -1.0] *    | -0.1 [-0.3, +0.2]   | -0.8 [-2.0, +0.2]        | -2.0 [-4.2, +0.2]      |
| full_base rerank_ondevice_filt           | ft_base  | ascend_test   | -1.7 [-3.9, -0.3] *      | -20.4 [-51.3, -3.4] *  | -20.4 [-51.3, -3.4] *  | +0.9 [+0.0, +1.8]   | -1.7 [-3.9, -0.3] *      | -4.9 [-10.3, -1.1] *   |
| full_base rerank_ondevice_filt           | ft_base  | nsc_test      | -2.4 [-3.6, -1.4] *      | -3.9 [-7.0, -1.6] *    | -3.9 [-7.0, -1.6] *    | +0.1 [-0.1, +0.2]   | -2.4 [-3.6, -1.4] *      | +0.7 [-0.6, +2.0]      |
| full_base rerank_ondevice_filt           | ft_base  | nsc_diff_test | -0.8 [-2.0, +0.2]        | -1.9 [-3.0, -1.0] *    | -1.9 [-3.0, -1.0] *    | -0.1 [-0.3, +0.2]   | -0.8 [-2.0, +0.2]        | -2.0 [-4.2, +0.2]      |
| full_base rerank_ondevice_filt_gated     | ft_base  | ascend_test   | +97.0 [+90.7, +103.8] *  | +47.2 [+15.0, +70.6] * | +47.2 [+15.0, +70.6] * | -1.3 [-2.2, -0.5] * | +97.1 [+90.7, +103.9] *  | +59.3 [+46.5, +71.8] * |
| full_base rerank_ondevice_filt_gated     | ft_base  | nsc_test      | -2.7 [-3.9, -1.7] *      | -4.0 [-7.0, -1.8] *    | -4.0 [-7.0, -1.8] *    | +0.1 [-0.1, +0.2]   | -2.7 [-3.9, -1.7] *      | +0.0 [-1.3, +1.3]      |
| full_base rerank_ondevice_filt_gated     | ft_base  | nsc_diff_test | -1.6 [-2.6, -0.7] *      | -2.2 [-3.2, -1.2] *    | -2.2 [-3.2, -1.2] *    | -0.0 [-0.2, +0.2]   | -1.6 [-2.6, -0.7] *      | -2.3 [-4.3, -0.2] *    |
| full_small full                          | ft_small | ascend_test   | -0.1 [-0.6, +0.3]        | -0.8 [-3.4, +2.2]      | -0.8 [-3.4, +2.2]      | -0.3 [-0.7, +0.0]   | -0.1 [-0.6, +0.3]        | -2.5 [-10.3, +4.5]     |
| full_small full                          | ft_small | nsc_test      | -0.7 [-1.3, -0.1] *      | -2.4 [-4.8, -0.7] *    | -2.4 [-4.8, -0.7] *    | -0.2 [-0.4, -0.0] * | -0.7 [-1.3, -0.1] *      | +0.0 [-1.2, +1.3]      |
| full_small full                          | ft_small | nsc_diff_test | -1.9 [-3.1, -0.9] *      | -4.2 [-8.9, -1.1] *    | -4.2 [-8.9, -1.1] *    | +0.1 [-0.2, +0.3]   | -1.9 [-3.1, -0.9] *      | +0.0 [-1.8, +1.7]      |
| full_small rerank_lp                     | ft_small | ascend_test   | +8.6 [+3.0, +15.3] *     | -2.6 [-4.9, -0.1] *    | -2.6 [-4.9, -0.1] *    | +0.1 [-0.4, +0.7]   | +8.6 [+3.0, +15.3] *     | -6.2 [-14.0, +1.4]     |
| full_small rerank_lp                     | ft_small | nsc_test      | +10.5 [+8.2, +12.9] *    | -1.7 [-4.4, +0.7]      | -1.7 [-4.4, +0.7]      | +0.1 [-0.1, +0.3]   | +10.5 [+8.2, +12.8] *    | +0.4 [-0.8, +1.7]      |
| full_small rerank_lp                     | ft_small | nsc_diff_test | +8.3 [+5.2, +11.1] *     | -3.5 [-8.2, -0.3] *    | -3.5 [-8.2, -0.3] *    | +0.2 [-0.1, +0.5]   | +8.3 [+5.2, +11.1] *     | +0.3 [-1.7, +2.2]      |
| full_small rerank_lp_filt                | ft_small | ascend_test   | -0.4 [-0.9, +0.1]        | -2.6 [-4.9, -0.1] *    | -2.6 [-4.9, -0.1] *    | +0.1 [-0.4, +0.7]   | -0.4 [-0.9, +0.1]        | -6.2 [-14.0, +1.4]     |
| full_small rerank_lp_filt                | ft_small | nsc_test      | +0.1 [-0.5, +0.7]        | -2.3 [-4.8, -0.5] *    | -2.3 [-4.8, -0.5] *    | +0.1 [-0.1, +0.3]   | +0.1 [-0.5, +0.7]        | +0.4 [-0.8, +1.7]      |
| full_small rerank_lp_filt                | ft_small | nsc_diff_test | -0.5 [-1.8, +0.7]        | -4.3 [-9.0, -1.1] *    | -4.3 [-9.0, -1.1] *    | +0.2 [-0.1, +0.5]   | -0.5 [-1.8, +0.7]        | +0.3 [-1.7, +2.2]      |
| full_small rerank_mms                    | ft_small | ascend_test   | +0.9 [+0.3, +1.6] *      | -5.4 [-9.1, -1.7] *    | -5.4 [-9.1, -1.7] *    | -0.1 [-1.1, +0.8]   | +0.9 [+0.3, +1.6] *      | -4.9 [-13.1, +3.2]     |
| full_small rerank_mms                    | ft_small | nsc_test      | +0.4 [-0.2, +1.0]        | -2.4 [-4.9, -0.5] *    | -2.4 [-4.9, -0.5] *    | -0.1 [-0.2, +0.1]   | +0.4 [-0.2, +1.0]        | +0.4 [-0.9, +1.8]      |
| full_small rerank_mms                    | ft_small | nsc_diff_test | -0.0 [-1.3, +1.1]        | -3.4 [-8.0, -0.0] *    | -3.4 [-8.0, -0.0] *    | +0.0 [-0.2, +0.2]   | -0.0 [-1.3, +1.2]        | -0.3 [-2.6, +1.8]      |
| full_small rerank_mms_filt               | ft_small | ascend_test   | +0.9 [+0.3, +1.6] *      | -5.4 [-9.1, -1.7] *    | -5.4 [-9.1, -1.7] *    | -0.1 [-1.1, +0.8]   | +0.9 [+0.3, +1.6] *      | -4.9 [-13.1, +3.2]     |
| full_small rerank_mms_filt               | ft_small | nsc_test      | +0.4 [-0.2, +1.0]        | -2.4 [-4.9, -0.5] *    | -2.4 [-4.9, -0.5] *    | -0.1 [-0.2, +0.1]   | +0.4 [-0.2, +1.0]        | +0.4 [-0.9, +1.8]      |
| full_small rerank_mms_filt               | ft_small | nsc_diff_test | -0.4 [-1.6, +0.8]        | -3.8 [-8.6, -0.5] *    | -3.8 [-8.6, -0.5] *    | +0.0 [-0.2, +0.2]   | -0.4 [-1.6, +0.8]        | -0.3 [-2.6, +1.8]      |
| full_small rerank_mms_filt_gated         | ft_small | ascend_test   | +101.8 [+95.5, +108.6] * | +67.4 [+58.0, +77.6] * | +67.4 [+58.0, +77.6] * | -1.3 [-2.4, -0.4] * | +101.9 [+95.7, +108.7] * | +63.0 [+49.2, +77.3] * |
| full_small rerank_mms_filt_gated         | ft_small | nsc_test      | -0.8 [-1.5, -0.3] *      | -2.5 [-5.0, -0.6] *    | -2.5 [-5.0, -0.6] *    | -0.1 [-0.2, +0.1]   | -0.8 [-1.5, -0.3] *      | +0.2 [-1.2, +1.5]      |
| full_small rerank_mms_filt_gated         | ft_small | nsc_diff_test | -1.8 [-3.0, -0.7] *      | -4.2 [-8.9, -1.0] *    | -4.2 [-8.9, -1.0] *    | +0.0 [-0.2, +0.2]   | -1.8 [-3.0, -0.7] *      | -0.6 [-2.7, +1.2]      |
| full_small rerank_ondevice               | ft_small | ascend_test   | +1.9 [+0.3, +4.1] *      | -1.8 [-5.5, +2.2]      | -1.8 [-5.5, +2.2]      | +0.3 [-0.6, +1.2]   | +1.9 [+0.3, +4.1] *      | -7.4 [-14.9, -1.1] *   |
| full_small rerank_ondevice               | ft_small | nsc_test      | +1.7 [+0.8, +2.7] *      | -1.9 [-4.4, -0.0] *    | -1.9 [-4.4, -0.0] *    | -0.0 [-0.2, +0.2]   | +1.7 [+0.8, +2.8] *      | +0.4 [-0.8, +1.8]      |
| full_small rerank_ondevice               | ft_small | nsc_diff_test | +0.8 [-0.6, +2.2]        | -3.2 [-7.9, +0.2]      | -3.2 [-7.9, +0.2]      | +0.1 [-0.1, +0.4]   | +0.8 [-0.6, +2.2]        | -0.5 [-2.5, +1.6]      |
| full_small rerank_ondevice_filt          | ft_small | ascend_test   | -0.5 [-1.1, +0.0]        | -3.8 [-6.8, -0.9] *    | -3.8 [-6.8, -0.9] *    | +0.1 [-0.8, +1.0]   | -0.5 [-1.0, +0.0]        | -6.2 [-14.0, +1.4]     |
| full_small rerank_ondevice_filt          | ft_small | nsc_test      | +0.1 [-0.6, +0.6]        | -2.1 [-4.6, -0.3] *    | -2.1 [-4.6, -0.3] *    | +0.0 [-0.1, +0.2]   | +0.1 [-0.5, +0.6]        | +0.2 [-1.1, +1.5]      |
| full_small rerank_ondevice_filt          | ft_small | nsc_diff_test | -0.6 [-1.9, +0.5]        | -4.0 [-8.9, -0.7] *    | -4.0 [-8.9, -0.7] *    | +0.1 [-0.2, +0.3]   | -0.6 [-1.9, +0.5]        | -0.3 [-2.4, +1.7]      |
| full_small rerank_ondevice_filt_gated    | ft_small | ascend_test   | +101.8 [+95.5, +108.7] * | +67.6 [+58.4, +77.3] * | +67.6 [+58.4, +77.3] * | -1.3 [-2.4, -0.4] * | +101.9 [+95.6, +108.7] * | +61.7 [+47.9, +76.7] * |
| full_small rerank_ondevice_filt_gated    | ft_small | nsc_test      | -0.8 [-1.5, -0.3] *      | -2.2 [-4.7, -0.3] *    | -2.2 [-4.7, -0.3] *    | +0.0 [-0.1, +0.2]   | -0.8 [-1.5, -0.3] *      | +0.1 [-1.2, +1.3]      |
| full_small rerank_ondevice_filt_gated    | ft_small | nsc_diff_test | -1.8 [-3.1, -0.8] *      | -4.3 [-9.2, -1.1] *    | -4.3 [-9.2, -1.1] *    | +0.1 [-0.2, +0.3]   | -1.8 [-3.1, -0.8] *      | -0.6 [-2.5, +1.3]      |
| fullown_base rerank_lp                   | ft_base  | ascend_test   | -1.7 [-4.0, -0.1] *      | -19.1 [-49.9, -1.4] *  | -19.1 [-49.9, -1.4] *  | +0.6 [-0.3, +1.6]   | -1.7 [-4.0, -0.1] *      | -4.9 [-10.3, -1.1] *   |
| fullown_base rerank_lp                   | ft_base  | nsc_test      | -2.1 [-3.3, -1.1] *      | -3.8 [-6.9, -1.5] *    | -3.8 [-6.9, -1.5] *    | +0.1 [-0.1, +0.2]   | -2.1 [-3.3, -1.1] *      | +0.2 [-1.2, +1.7]      |
| fullown_base rerank_lp                   | ft_base  | nsc_diff_test | -0.6 [-1.7, +0.4]        | -2.2 [-3.2, -1.3] *    | -2.2 [-3.2, -1.3] *    | -0.1 [-0.3, +0.2]   | -0.6 [-1.7, +0.4]        | -2.4 [-4.5, -0.3] *    |
| fullown_base rerank_lp_filt              | ft_base  | ascend_test   | -1.9 [-4.2, -0.5] *      | -20.3 [-51.2, -3.2] *  | -20.3 [-51.2, -3.2] *  | +0.6 [-0.3, +1.6]   | -1.9 [-4.2, -0.5] *      | -4.9 [-10.3, -1.1] *   |
| fullown_base rerank_lp_filt              | ft_base  | nsc_test      | -2.2 [-3.4, -1.2] *      | -3.8 [-6.9, -1.5] *    | -3.8 [-6.9, -1.5] *    | +0.1 [-0.1, +0.2]   | -2.2 [-3.4, -1.2] *      | +0.2 [-1.2, +1.7]      |
| fullown_base rerank_lp_filt              | ft_base  | nsc_diff_test | -0.6 [-1.7, +0.4]        | -2.2 [-3.2, -1.3] *    | -2.2 [-3.2, -1.3] *    | -0.1 [-0.3, +0.2]   | -0.6 [-1.7, +0.4]        | -2.4 [-4.5, -0.3] *    |
| fullown_base rerank_mms                  | ft_base  | ascend_test   | -0.1 [-2.4, +1.5]        | -21.5 [-52.6, -4.1] *  | -21.5 [-52.6, -4.1] *  | +0.6 [-0.4, +1.8]   | -0.1 [-2.4, +1.5]        | -7.4 [-13.7, -2.4] *   |
| fullown_base rerank_mms                  | ft_base  | nsc_test      | -2.1 [-3.3, -1.1] *      | -3.6 [-6.7, -1.4] *    | -3.6 [-6.7, -1.4] *    | +0.1 [-0.1, +0.3]   | -2.1 [-3.3, -1.1] *      | +0.3 [-0.9, +1.8]      |
| fullown_base rerank_mms                  | ft_base  | nsc_diff_test | -0.2 [-1.3, +0.8]        | -1.4 [-2.5, -0.4] *    | -1.4 [-2.5, -0.4] *    | -0.0 [-0.2, +0.2]   | -0.2 [-1.3, +0.8]        | -1.8 [-4.3, +0.6]      |
| fullown_base rerank_mms_filt             | ft_base  | ascend_test   | -0.1 [-2.4, +1.5]        | -21.5 [-52.6, -4.1] *  | -21.5 [-52.6, -4.1] *  | +0.6 [-0.4, +1.8]   | -0.1 [-2.4, +1.5]        | -7.4 [-13.7, -2.4] *   |
| fullown_base rerank_mms_filt             | ft_base  | nsc_test      | -2.1 [-3.3, -1.1] *      | -3.6 [-6.7, -1.4] *    | -3.6 [-6.7, -1.4] *    | +0.1 [-0.1, +0.3]   | -2.1 [-3.3, -1.1] *      | +0.3 [-0.9, +1.8]      |
| fullown_base rerank_mms_filt             | ft_base  | nsc_diff_test | -0.2 [-1.3, +0.8]        | -1.5 [-2.5, -0.4] *    | -1.5 [-2.5, -0.4] *    | -0.0 [-0.2, +0.2]   | -0.2 [-1.3, +0.8]        | -1.8 [-4.3, +0.6]      |
| fullown_base rerank_mms_filt_gated       | ft_base  | ascend_test   | +95.4 [+88.9, +102.3] *  | +44.8 [+12.5, +67.7] * | +44.8 [+12.5, +67.7] * | -1.0 [-2.1, -0.1] * | +95.4 [+89.0, +102.4] *  | +58.0 [+44.9, +71.1] * |
| fullown_base rerank_mms_filt_gated       | ft_base  | nsc_test      | -2.7 [-3.9, -1.7] *      | -3.9 [-7.0, -1.7] *    | -3.9 [-7.0, -1.7] *    | +0.0 [-0.1, +0.2]   | -2.7 [-3.9, -1.7] *      | +0.0 [-1.3, +1.4]      |
| fullown_base rerank_mms_filt_gated       | ft_base  | nsc_diff_test | -1.6 [-2.6, -0.7] *      | -2.2 [-3.3, -1.2] *    | -2.2 [-3.3, -1.2] *    | -0.1 [-0.3, +0.1]   | -1.6 [-2.6, -0.7] *      | -2.3 [-4.3, -0.1] *    |
| fullown_base rerank_ondevice             | ft_base  | ascend_test   | -1.4 [-3.8, +0.1]        | -19.0 [-49.2, -1.3] *  | -19.0 [-49.2, -1.3] *  | +0.9 [+0.0, +1.8]   | -1.5 [-3.8, +0.1]        | -4.9 [-10.3, -1.1] *   |
| fullown_base rerank_ondevice             | ft_base  | nsc_test      | -2.4 [-3.6, -1.4] *      | -3.9 [-7.0, -1.6] *    | -3.9 [-7.0, -1.6] *    | +0.1 [-0.1, +0.2]   | -2.4 [-3.6, -1.4] *      | +0.7 [-0.6, +2.0]      |
| fullown_base rerank_ondevice             | ft_base  | nsc_diff_test | -0.8 [-1.9, +0.2]        | -1.9 [-3.0, -0.9] *    | -1.9 [-3.0, -0.9] *    | -0.1 [-0.3, +0.2]   | -0.8 [-1.9, +0.2]        | -2.0 [-4.2, +0.2]      |
| fullown_base rerank_ondevice_filt        | ft_base  | ascend_test   | -1.7 [-3.9, -0.3] *      | -20.3 [-51.0, -3.2] *  | -20.3 [-51.0, -3.2] *  | +0.9 [+0.0, +1.8]   | -1.7 [-4.0, -0.3] *      | -4.9 [-10.3, -1.1] *   |
| fullown_base rerank_ondevice_filt        | ft_base  | nsc_test      | -2.4 [-3.6, -1.4] *      | -3.9 [-7.0, -1.6] *    | -3.9 [-7.0, -1.6] *    | +0.1 [-0.1, +0.2]   | -2.4 [-3.6, -1.4] *      | +0.7 [-0.6, +2.0]      |
| fullown_base rerank_ondevice_filt        | ft_base  | nsc_diff_test | -0.8 [-1.9, +0.2]        | -1.9 [-3.0, -0.9] *    | -1.9 [-3.0, -0.9] *    | -0.1 [-0.3, +0.2]   | -0.8 [-1.9, +0.2]        | -2.0 [-4.2, +0.2]      |
| fullown_base rerank_ondevice_filt_gated  | ft_base  | ascend_test   | +95.4 [+88.9, +102.3] *  | +45.5 [+12.7, +68.3] * | +45.5 [+12.7, +68.3] * | -1.0 [-2.1, -0.1] * | +95.5 [+89.0, +102.3] *  | +59.3 [+46.5, +71.8] * |
| fullown_base rerank_ondevice_filt_gated  | ft_base  | nsc_test      | -2.7 [-3.9, -1.7] *      | -4.0 [-7.0, -1.8] *    | -4.0 [-7.0, -1.8] *    | +0.0 [-0.1, +0.2]   | -2.7 [-3.9, -1.7] *      | +0.0 [-1.2, +1.2]      |
| fullown_base rerank_ondevice_filt_gated  | ft_base  | nsc_diff_test | -1.6 [-2.6, -0.7] *      | -2.1 [-3.1, -1.1] *    | -2.1 [-3.1, -1.1] *    | -0.0 [-0.2, +0.2]   | -1.6 [-2.6, -0.7] *      | -2.1 [-4.2, -0.1] *    |
| fullown_small rerank_lp                  | ft_small | ascend_test   | +8.6 [+3.0, +15.3] *     | -2.6 [-4.9, -0.1] *    | -2.6 [-4.9, -0.1] *    | +0.1 [-0.4, +0.7]   | +8.6 [+3.0, +15.3] *     | -6.2 [-14.0, +1.4]     |
| fullown_small rerank_lp                  | ft_small | nsc_test      | +10.5 [+8.2, +12.9] *    | -1.7 [-4.4, +0.7]      | -1.7 [-4.4, +0.7]      | +0.1 [-0.1, +0.3]   | +10.5 [+8.2, +12.8] *    | +0.4 [-0.8, +1.7]      |
| fullown_small rerank_lp                  | ft_small | nsc_diff_test | +8.3 [+5.2, +11.1] *     | -3.5 [-8.2, -0.3] *    | -3.5 [-8.2, -0.3] *    | +0.2 [-0.1, +0.5]   | +8.3 [+5.2, +11.1] *     | +0.3 [-1.7, +2.2]      |
| fullown_small rerank_lp_filt             | ft_small | ascend_test   | -0.4 [-0.9, +0.1]        | -2.6 [-4.9, -0.1] *    | -2.6 [-4.9, -0.1] *    | +0.1 [-0.4, +0.7]   | -0.4 [-0.9, +0.1]        | -6.2 [-14.0, +1.4]     |
| fullown_small rerank_lp_filt             | ft_small | nsc_test      | +0.1 [-0.5, +0.7]        | -2.3 [-4.8, -0.5] *    | -2.3 [-4.8, -0.5] *    | +0.1 [-0.1, +0.3]   | +0.1 [-0.5, +0.7]        | +0.4 [-0.8, +1.7]      |
| fullown_small rerank_lp_filt             | ft_small | nsc_diff_test | -0.5 [-1.8, +0.7]        | -4.3 [-9.0, -1.1] *    | -4.3 [-9.0, -1.1] *    | +0.2 [-0.1, +0.5]   | -0.5 [-1.8, +0.7]        | +0.3 [-1.7, +2.2]      |
| fullown_small rerank_mms                 | ft_small | ascend_test   | +1.0 [+0.3, +1.6] *      | -3.3 [-6.0, -0.5] *    | -3.3 [-6.0, -0.5] *    | +0.3 [-0.5, +1.0]   | +1.0 [+0.3, +1.6] *      | -3.7 [-12.2, +4.8]     |
| fullown_small rerank_mms                 | ft_small | nsc_test      | +0.4 [-0.3, +0.9]        | -2.4 [-4.8, -0.4] *    | -2.4 [-4.8, -0.4] *    | -0.1 [-0.3, +0.0]   | +0.4 [-0.3, +0.9]        | +0.9 [-0.4, +2.2]      |
| fullown_small rerank_mms                 | ft_small | nsc_diff_test | -0.0 [-1.3, +1.2]        | -3.5 [-8.2, -0.2] *    | -3.5 [-8.2, -0.2] *    | +0.1 [-0.2, +0.3]   | -0.0 [-1.3, +1.2]        | -0.3 [-2.7, +2.0]      |
| fullown_small rerank_mms_filt            | ft_small | ascend_test   | +1.0 [+0.3, +1.6] *      | -3.3 [-6.0, -0.5] *    | -3.3 [-6.0, -0.5] *    | +0.3 [-0.5, +1.0]   | +1.0 [+0.3, +1.6] *      | -3.7 [-12.2, +4.8]     |
| fullown_small rerank_mms_filt            | ft_small | nsc_test      | +0.4 [-0.3, +0.9]        | -2.4 [-4.8, -0.4] *    | -2.4 [-4.8, -0.4] *    | -0.1 [-0.3, +0.0]   | +0.4 [-0.3, +0.9]        | +0.9 [-0.4, +2.2]      |
| fullown_small rerank_mms_filt            | ft_small | nsc_diff_test | -0.3 [-1.6, +0.8]        | -4.0 [-8.8, -0.7] *    | -4.0 [-8.8, -0.7] *    | +0.1 [-0.2, +0.3]   | -0.3 [-1.6, +0.8]        | -0.3 [-2.7, +2.0]      |
| fullown_small rerank_mms_filt_gated      | ft_small | ascend_test   | +102.0 [+95.6, +108.9] * | +68.4 [+58.8, +77.8] * | +68.4 [+58.8, +77.8] * | -1.0 [-1.9, -0.3] * | +102.1 [+95.7, +108.9] * | +64.2 [+50.6, +78.2] * |
| fullown_small rerank_mms_filt_gated      | ft_small | nsc_test      | -0.9 [-1.5, -0.4] *      | -2.5 [-5.0, -0.6] *    | -2.5 [-5.0, -0.6] *    | -0.1 [-0.2, +0.1]   | -0.9 [-1.5, -0.4] *      | +0.1 [-1.2, +1.3]      |
| fullown_small rerank_mms_filt_gated      | ft_small | nsc_diff_test | -1.9 [-3.1, -0.8] *      | -4.3 [-9.1, -1.0] *    | -4.3 [-9.1, -1.0] *    | +0.1 [-0.2, +0.3]   | -1.9 [-3.1, -0.8] *      | -0.3 [-2.4, +1.7]      |
| fullown_small rerank_ondevice            | ft_small | ascend_test   | +1.3 [+0.1, +3.1] *      | -0.2 [-3.2, +3.1]      | -0.2 [-3.2, +3.1]      | +0.5 [-0.1, +1.3]   | +1.3 [+0.1, +3.1] *      | -6.2 [-13.6, +1.2]     |
| fullown_small rerank_ondevice            | ft_small | nsc_test      | +0.9 [+0.2, +1.8] *      | -1.9 [-4.4, -0.0] *    | -1.9 [-4.4, -0.0] *    | -0.0 [-0.2, +0.2]   | +0.9 [+0.2, +1.8] *      | +0.3 [-1.0, +1.6]      |
| fullown_small rerank_ondevice            | ft_small | nsc_diff_test | +1.0 [-0.5, +2.5]        | -3.1 [-7.9, +0.3]      | -3.1 [-7.9, +0.3]      | +0.2 [-0.1, +0.5]   | +1.0 [-0.5, +2.5]        | -0.8 [-2.8, +1.3]      |
| fullown_small rerank_ondevice_filt       | ft_small | ascend_test   | -0.4 [-0.9, +0.1]        | -1.6 [-3.8, +0.7]      | -1.6 [-3.8, +0.7]      | +0.4 [-0.3, +1.1]   | -0.4 [-0.9, +0.1]        | -6.2 [-14.0, +1.4]     |
| fullown_small rerank_ondevice_filt       | ft_small | nsc_test      | +0.1 [-0.5, +0.7]        | -2.2 [-4.6, -0.3] *    | -2.2 [-4.6, -0.3] *    | +0.0 [-0.1, +0.2]   | +0.1 [-0.5, +0.7]        | +0.0 [-1.2, +1.3]      |
| fullown_small rerank_ondevice_filt       | ft_small | nsc_diff_test | -0.7 [-2.0, +0.5]        | -3.9 [-8.8, -0.6] *    | -3.9 [-8.8, -0.6] *    | +0.1 [-0.2, +0.4]   | -0.7 [-2.0, +0.5]        | -0.8 [-2.8, +1.2]      |
| fullown_small rerank_ondevice_filt_gated | ft_small | ascend_test   | +102.0 [+95.6, +108.8] * | +69.0 [+59.7, +78.2] * | +69.0 [+59.7, +78.2] * | -1.0 [-1.9, -0.3] * | +102.0 [+95.6, +108.8] * | +63.0 [+49.4, +77.1] * |
| fullown_small rerank_ondevice_filt_gated | ft_small | nsc_test      | -0.9 [-1.6, -0.4] *      | -2.3 [-4.8, -0.5] *    | -2.3 [-4.8, -0.5] *    | +0.0 [-0.1, +0.2]   | -0.9 [-1.6, -0.4] *      | +0.1 [-1.1, +1.3]      |
| fullown_small rerank_ondevice_filt_gated | ft_small | nsc_diff_test | -1.9 [-3.2, -0.8] *      | -4.2 [-9.0, -1.0] *    | -4.2 [-9.0, -1.0] *    | +0.1 [-0.1, +0.4]   | -1.9 [-3.2, -0.8] *      | -0.5 [-2.4, +1.5]      |
| rw_base rw                               | ft_base  | ascend_test   | -0.7 [-2.8, +0.5]        | -14.6 [-45.2, +1.6]    | -14.6 [-45.2, +1.6]    | +0.1 [+0.0, +0.4]   | -0.7 [-2.8, +0.5]        | +0.0 [-5.1, +5.1]      |
| rw_base rw                               | ft_base  | nsc_test      | -0.6 [-1.6, +0.2]        | -2.7 [-5.6, -0.6] *    | -2.7 [-5.6, -0.6] *    | -0.1 [-0.2, +0.0]   | -0.6 [-1.6, +0.2]        | +0.7 [-0.4, +1.7]      |
| rw_base rw                               | ft_base  | nsc_diff_test | -0.6 [-1.3, +0.0]        | -0.9 [-1.5, -0.2] *    | -0.9 [-1.5, -0.2] *    | -0.1 [-0.2, +0.1]   | -0.6 [-1.3, +0.0]        | -0.8 [-1.8, +0.2]      |
| semonly_base semonly                     | ft_base  | ascend_test   | -1.2 [-3.4, +0.1]        | -14.6 [-44.8, +1.3]    | -14.6 [-44.8, +1.3]    | -0.3 [-0.8, +0.3]   | -1.2 [-3.4, +0.1]        | -3.7 [-9.1, +1.3]      |
| semonly_base semonly                     | ft_base  | nsc_test      | -1.3 [-2.4, -0.4] *      | -1.7 [-4.6, +1.0]      | -1.7 [-4.6, +1.0]      | -0.0 [-0.2, +0.1]   | -1.3 [-2.4, -0.4] *      | +0.8 [-0.5, +2.1]      |
| semonly_base semonly                     | ft_base  | nsc_diff_test | -1.1 [-2.2, -0.1] *      | -0.4 [-1.3, +0.3]      | -0.4 [-1.3, +0.3]      | -0.1 [-0.3, +0.1]   | -1.1 [-2.2, -0.1] *      | -2.1 [-3.9, -0.5] *    |
| sub_small sub                            | ft_small | ascend_test   | -0.5 [-1.0, -0.0] *      | -2.3 [-4.9, +0.5]      | -2.3 [-4.9, +0.5]      | +0.0 [-0.5, +0.5]   | -0.5 [-1.0, -0.0] *      | -1.2 [-5.9, +2.6]      |
| sub_small sub                            | ft_small | nsc_test      | +0.1 [-0.6, +0.9]        | -1.1 [-2.6, +0.2]      | -1.1 [-2.6, +0.2]      | -0.1 [-0.2, +0.1]   | +0.1 [-0.6, +0.9]        | -0.1 [-1.2, +1.2]      |
| sub_small sub                            | ft_small | nsc_diff_test | -0.9 [-1.9, +0.2]        | -3.3 [-8.0, +0.7]      | -3.3 [-8.0, +0.7]      | +0.1 [-0.1, +0.3]   | -0.9 [-1.9, +0.2]        | -0.2 [-1.9, +1.6]      |

## Re-ranking weights (ft_base), chosen on validation

{"rerank_lp": [0.0, 0.0], "rerank_mms": [0.25, 2.0], "rerank_ondevice": [0.0, 2.0]}

## Re-ranking weights (ft_base_filt), chosen on validation

{"rerank_lp_filt": [0.0, 0.0], "rerank_mms_filt": [0.25, 2.0], "rerank_ondevice_filt": [0.0, 0.5]}

## Re-ranking weights (full_base), chosen on validation

{"rerank_lp": [0.0, 0.0], "rerank_mms": [0.25, 2.0], "rerank_ondevice": [0.0, 0.25]}

## Re-ranking weights (full_base_filt), chosen on validation

{"rerank_lp_filt": [0.0, 0.0], "rerank_mms_filt": [0.25, 2.0], "rerank_ondevice_filt": [0.0, 0.25]}

## Re-ranking weights (full_small), chosen on validation

{"rerank_lp": [0.0, 0.0], "rerank_mms": [0.25, 2.0], "rerank_ondevice": [0.0, 2.0]}

## Re-ranking weights (full_small_filt), chosen on validation

{"rerank_lp_filt": [0.0, 0.0], "rerank_mms_filt": [0.25, 2.0], "rerank_ondevice_filt": [0.0, 0.5]}

## Re-ranking weights (fullown_base), chosen on validation

{"rerank_lp": [0.0, 0.0], "rerank_mms": [0.25, 2.0], "rerank_ondevice": [0.0, 0.25]}

## Re-ranking weights (fullown_base_filt), chosen on validation

{"rerank_lp_filt": [0.0, 0.0], "rerank_mms_filt": [0.25, 2.0], "rerank_ondevice_filt": [0.0, 0.25]}

## Re-ranking weights (fullown_small), chosen on validation

{"rerank_lp": [0.0, 0.0], "rerank_mms": [0.25, 2.0], "rerank_ondevice": [0.0, 2.0]}

## Re-ranking weights (fullown_small_filt), chosen on validation

{"rerank_lp_filt": [0.0, 0.0], "rerank_mms_filt": [0.25, 2.0], "rerank_ondevice_filt": [0.0, 0.5]}

## Critical-term flags (precision / recall against real critical-word errors)

| system                                          | set           | precision   |   recall |   flags |   critical_errors |
|:------------------------------------------------|:--------------|:------------|---------:|--------:|------------------:|
| ft_base__fp16__rerank_lp                        | ascend_test   | 43.8        |      5.6 |      16 |                18 |
| ft_base__fp16__rerank_lp                        | nsc_test      | 55.2        |     16.2 |     232 |               253 |
| ft_base__fp16__rerank_lp                        | nsc_diff_test | 60.2        |     17.4 |     113 |               138 |
| ft_base__fp16__rerank_lp_filt                   | ascend_test   | 43.8        |      5.6 |      16 |                18 |
| ft_base__fp16__rerank_lp_filt                   | nsc_test      | 55.4        |     16.4 |     231 |               250 |
| ft_base__fp16__rerank_lp_filt                   | nsc_diff_test | 62.4        |     17.6 |     109 |               136 |
| ft_base__fp16__rerank_mms                       | ascend_test   | 50.0        |      6.7 |      24 |                15 |
| ft_base__fp16__rerank_mms                       | nsc_test      | 62.8        |     14.1 |     226 |               255 |
| ft_base__fp16__rerank_mms                       | nsc_diff_test | 60.0        |     18.5 |     115 |               135 |
| ft_base__fp16__rerank_mms_filt                  | ascend_test   | 50.0        |      6.7 |      24 |                15 |
| ft_base__fp16__rerank_mms_filt                  | nsc_test      | 62.7        |     14.3 |     225 |               252 |
| ft_base__fp16__rerank_mms_filt                  | nsc_diff_test | 60.0        |     18.5 |     115 |               135 |
| ft_base__fp16__rerank_ondevice                  | ascend_test   | 47.4        |      5.9 |      19 |                17 |
| ft_base__fp16__rerank_ondevice                  | nsc_test      | 62.7        |     15.7 |     252 |               261 |
| ft_base__fp16__rerank_ondevice                  | nsc_diff_test | 62.2        |     17.3 |      98 |               139 |
| ft_base__fp16__rerank_ondevice_filt             | ascend_test   | 38.9        |      0   |      18 |                16 |
| ft_base__fp16__rerank_ondevice_filt             | nsc_test      | 58.3        |     14.6 |     240 |               254 |
| ft_base__fp16__rerank_ondevice_filt             | nsc_diff_test | 62.9        |     16.1 |      97 |               137 |
| full_base__fp16__rerank_lp                      | ascend_test   | 42.1        |     12.5 |      19 |                16 |
| full_base__fp16__rerank_lp                      | nsc_test      | 54.0        |      8.9 |     198 |               237 |
| full_base__fp16__rerank_lp                      | nsc_diff_test | 43.3        |      6.5 |      97 |               108 |
| full_base__fp16__rerank_lp_filt                 | ascend_test   | 42.1        |     12.5 |      19 |                16 |
| full_base__fp16__rerank_lp_filt                 | nsc_test      | 54.0        |      8.9 |     198 |               237 |
| full_base__fp16__rerank_lp_filt                 | nsc_diff_test | 43.3        |      6.5 |      97 |               108 |
| full_base__fp16__rerank_mms                     | ascend_test   | 53.8        |      7.1 |      26 |                14 |
| full_base__fp16__rerank_mms                     | nsc_test      | 55.3        |      9.2 |     206 |               239 |
| full_base__fp16__rerank_mms                     | nsc_diff_test | 55.3        |      9.1 |     114 |               110 |
| full_base__fp16__rerank_mms_filt                | ascend_test   | 53.8        |      7.1 |      26 |                14 |
| full_base__fp16__rerank_mms_filt                | nsc_test      | 55.3        |      9.2 |     206 |               239 |
| full_base__fp16__rerank_mms_filt                | nsc_diff_test | 55.3        |      9.1 |     114 |               110 |
| full_base__fp16__rerank_mms_filt_gated          | ascend_test   | –           |      0   |       0 |                67 |
| full_base__fp16__rerank_mms_filt_gated          | nsc_test      | –           |      0   |       0 |               237 |
| full_base__fp16__rerank_mms_filt_gated          | nsc_diff_test | –           |      0   |       0 |               106 |
| full_base__fp16__rerank_ondevice                | ascend_test   | 50.0        |     12.5 |      22 |                16 |
| full_base__fp16__rerank_ondevice                | nsc_test      | 56.4        |      9.5 |     202 |               243 |
| full_base__fp16__rerank_ondevice                | nsc_diff_test | 44.3        |      9   |      97 |               111 |
| full_base__fp16__rerank_ondevice_filt           | ascend_test   | 50.0        |     12.5 |      22 |                16 |
| full_base__fp16__rerank_ondevice_filt           | nsc_test      | 56.4        |      9.5 |     202 |               243 |
| full_base__fp16__rerank_ondevice_filt           | nsc_diff_test | 44.3        |      9   |      97 |               111 |
| full_base__fp16__rerank_ondevice_filt_gated     | ascend_test   | –           |      0   |       0 |                68 |
| full_base__fp16__rerank_ondevice_filt_gated     | nsc_test      | –           |      0   |       0 |               235 |
| full_base__fp16__rerank_ondevice_filt_gated     | nsc_diff_test | –           |      0   |       0 |               109 |
| full_small__fp16__rerank_lp                     | ascend_test   | 36.0        |     18.2 |      25 |                11 |
| full_small__fp16__rerank_lp                     | nsc_test      | 52.2        |     15.2 |     205 |               171 |
| full_small__fp16__rerank_lp                     | nsc_diff_test | 47.9        |     11.8 |      96 |                85 |
| full_small__fp16__rerank_lp_filt                | ascend_test   | 36.0        |     18.2 |      25 |                11 |
| full_small__fp16__rerank_lp_filt                | nsc_test      | 52.2        |     15.2 |     205 |               171 |
| full_small__fp16__rerank_lp_filt                | nsc_diff_test | 47.9        |     11.8 |      96 |                85 |
| full_small__fp16__rerank_mms                    | ascend_test   | 39.1        |     33.3 |      23 |                12 |
| full_small__fp16__rerank_mms                    | nsc_test      | 54.7        |     12.9 |     190 |               171 |
| full_small__fp16__rerank_mms                    | nsc_diff_test | 52.4        |     11.1 |     103 |                81 |
| full_small__fp16__rerank_mms_filt               | ascend_test   | 39.1        |     33.3 |      23 |                12 |
| full_small__fp16__rerank_mms_filt               | nsc_test      | 54.7        |     12.9 |     190 |               171 |
| full_small__fp16__rerank_mms_filt               | nsc_diff_test | 52.4        |     11.1 |     103 |                81 |
| full_small__fp16__rerank_mms_filt_gated         | ascend_test   | –           |      0   |       0 |                67 |
| full_small__fp16__rerank_mms_filt_gated         | nsc_test      | –           |      0   |       0 |               168 |
| full_small__fp16__rerank_mms_filt_gated         | nsc_diff_test | –           |      0   |       0 |                79 |
| full_small__fp16__rerank_ondevice               | ascend_test   | 30.0        |     10   |      20 |                10 |
| full_small__fp16__rerank_ondevice               | nsc_test      | 53.3        |     14.6 |     199 |               171 |
| full_small__fp16__rerank_ondevice               | nsc_diff_test | 54.5        |     12.5 |     101 |                80 |
| full_small__fp16__rerank_ondevice_filt          | ascend_test   | 27.3        |     18.2 |      22 |                11 |
| full_small__fp16__rerank_ondevice_filt          | nsc_test      | 49.2        |     14.9 |     195 |               168 |
| full_small__fp16__rerank_ondevice_filt          | nsc_diff_test | 49.5        |      9.9 |      99 |                81 |
| full_small__fp16__rerank_ondevice_filt_gated    | ascend_test   | –           |      0   |       0 |                66 |
| full_small__fp16__rerank_ondevice_filt_gated    | nsc_test      | –           |      0   |       0 |               167 |
| full_small__fp16__rerank_ondevice_filt_gated    | nsc_diff_test | –           |      0   |       0 |                79 |
| fullown_base__fp16__rerank_lp                   | ascend_test   | 42.1        |     12.5 |      19 |                16 |
| fullown_base__fp16__rerank_lp                   | nsc_test      | 54.0        |      8.9 |     198 |               237 |
| fullown_base__fp16__rerank_lp                   | nsc_diff_test | 43.3        |      6.5 |      97 |               108 |
| fullown_base__fp16__rerank_lp_filt              | ascend_test   | 42.1        |     12.5 |      19 |                16 |
| fullown_base__fp16__rerank_lp_filt              | nsc_test      | 54.0        |      8.9 |     198 |               237 |
| fullown_base__fp16__rerank_lp_filt              | nsc_diff_test | 43.3        |      6.5 |      97 |               108 |
| fullown_base__fp16__rerank_mms                  | ascend_test   | 53.8        |      7.1 |      26 |                14 |
| fullown_base__fp16__rerank_mms                  | nsc_test      | 56.5        |      8.8 |     209 |               239 |
| fullown_base__fp16__rerank_mms                  | nsc_diff_test | 56.4        |      8   |     110 |               112 |
| fullown_base__fp16__rerank_mms_filt             | ascend_test   | 53.8        |      7.1 |      26 |                14 |
| fullown_base__fp16__rerank_mms_filt             | nsc_test      | 56.5        |      8.8 |     209 |               239 |
| fullown_base__fp16__rerank_mms_filt             | nsc_diff_test | 56.4        |      8   |     110 |               112 |
| fullown_base__fp16__rerank_mms_filt_gated       | ascend_test   | –           |      0   |       0 |                67 |
| fullown_base__fp16__rerank_mms_filt_gated       | nsc_test      | –           |      0   |       0 |               235 |
| fullown_base__fp16__rerank_mms_filt_gated       | nsc_diff_test | –           |      0   |       0 |               109 |
| fullown_base__fp16__rerank_ondevice             | ascend_test   | 50.0        |     12.5 |      22 |                16 |
| fullown_base__fp16__rerank_ondevice             | nsc_test      | 56.4        |      9.5 |     204 |               243 |
| fullown_base__fp16__rerank_ondevice             | nsc_diff_test | 43.8        |      9   |      96 |               111 |
| fullown_base__fp16__rerank_ondevice_filt        | ascend_test   | 50.0        |     12.5 |      22 |                16 |
| fullown_base__fp16__rerank_ondevice_filt        | nsc_test      | 56.4        |      9.5 |     204 |               243 |
| fullown_base__fp16__rerank_ondevice_filt        | nsc_diff_test | 43.8        |      9   |      96 |               111 |
| fullown_base__fp16__rerank_ondevice_filt_gated  | ascend_test   | –           |      0   |       0 |                68 |
| fullown_base__fp16__rerank_ondevice_filt_gated  | nsc_test      | –           |      0   |       0 |               235 |
| fullown_base__fp16__rerank_ondevice_filt_gated  | nsc_diff_test | –           |      0   |       0 |               110 |
| fullown_small__fp16__rerank_lp                  | ascend_test   | 36.0        |     18.2 |      25 |                11 |
| fullown_small__fp16__rerank_lp                  | nsc_test      | 52.2        |     15.2 |     205 |               171 |
| fullown_small__fp16__rerank_lp                  | nsc_diff_test | 47.9        |     11.8 |      96 |                85 |
| fullown_small__fp16__rerank_lp_filt             | ascend_test   | 36.0        |     18.2 |      25 |                11 |
| fullown_small__fp16__rerank_lp_filt             | nsc_test      | 52.2        |     15.2 |     205 |               171 |
| fullown_small__fp16__rerank_lp_filt             | nsc_diff_test | 47.9        |     11.8 |      96 |                85 |
| fullown_small__fp16__rerank_mms                 | ascend_test   | 41.7        |     38.5 |      24 |                13 |
| fullown_small__fp16__rerank_mms                 | nsc_test      | 56.1        |     14.2 |     196 |               176 |
| fullown_small__fp16__rerank_mms                 | nsc_diff_test | 53.3        |     12.3 |     107 |                81 |
| fullown_small__fp16__rerank_mms_filt            | ascend_test   | 41.7        |     38.5 |      24 |                13 |
| fullown_small__fp16__rerank_mms_filt            | nsc_test      | 56.1        |     14.2 |     196 |               176 |
| fullown_small__fp16__rerank_mms_filt            | nsc_diff_test | 53.3        |     12.3 |     107 |                81 |
| fullown_small__fp16__rerank_mms_filt_gated      | ascend_test   | –           |      0   |       0 |                68 |
| fullown_small__fp16__rerank_mms_filt_gated      | nsc_test      | –           |      0   |       0 |               167 |
| fullown_small__fp16__rerank_mms_filt_gated      | nsc_diff_test | –           |      0   |       0 |                81 |
| fullown_small__fp16__rerank_ondevice            | ascend_test   | 26.3        |      9.1 |      19 |                11 |
| fullown_small__fp16__rerank_ondevice            | nsc_test      | 51.5        |     15.4 |     196 |               169 |
| fullown_small__fp16__rerank_ondevice            | nsc_diff_test | 51.8        |     12.8 |     110 |                78 |
| fullown_small__fp16__rerank_ondevice_filt       | ascend_test   | 27.3        |     18.2 |      22 |                11 |
| fullown_small__fp16__rerank_ondevice_filt       | nsc_test      | 48.7        |     13.3 |     195 |               166 |
| fullown_small__fp16__rerank_ondevice_filt       | nsc_diff_test | 49.0        |     11.5 |     102 |                78 |
| fullown_small__fp16__rerank_ondevice_filt_gated | ascend_test   | –           |      0   |       0 |                67 |
| fullown_small__fp16__rerank_ondevice_filt_gated | nsc_test      | –           |      0   |       0 |               167 |
| fullown_small__fp16__rerank_ondevice_filt_gated | nsc_diff_test | –           |      0   |       0 |                80 |
