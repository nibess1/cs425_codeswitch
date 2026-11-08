# Efficiency

## Simulated (bitsandbytes on GPU): measured memory footprint

Only linear layers are quantized; conv front-end, embeddings, layer norms and proj_out stay 16-bit.

| model               | precision   | method        |   gpu_footprint_mb |   quantized_linear_modules |   peak_gpu_mb_decoding |
|:--------------------|:------------|:--------------|-------------------:|---------------------------:|-----------------------:|
| base                | fp16        | offshelf_both |              138.5 |                          0 |                 1196.9 |
| base                | int8_bnb    | offshelf_free |               96.5 |                         96 |                 4285.8 |
| base                | nf4_bnb     | offshelf_free |               75.5 |                         96 |                 1136.4 |
| ediff_base          | fp16        | eabl          |              138.5 |                          0 |                 1195   |
| emix_base           | fp16        | eabl          |              138.5 |                          0 |                 1195   |
| esame_base          | fp16        | eabl          |              138.5 |                          0 |                 1195   |
| ft_base             | fp16        | ft            |              138.5 |                          0 |                 1195.9 |
| ft_base             | int8_bnb    | ft            |               96.5 |                         96 |                 4285.8 |
| ft_base             | nf4_bnb     | ft            |               75.5 |                         96 |                 1136.4 |
| ft_base             | nf4_qlora   | qat_ft        |               83.7 |                         96 |                 1944.7 |
| ft_small            | fp16        | ft            |              461.1 |                          0 |                 3412   |
| ft_small            | int8_bnb    | ft            |              272.1 |                        192 |                10105.1 |
| ft_small            | nf4_bnb     | ft            |              177.6 |                        192 |                 3126.5 |
| ft_small            | nf4_qlora   | qat_ft        |              202.3 |                        192 |                 3522.6 |
| ft_tiny             | fp16        | ft            |               72   |                          0 |                  737   |
| ft_tiny             | int8_bnb    | ft            |               56.3 |                         64 |                 3216.3 |
| ft_tiny             | nf4_bnb     | ft            |               48.4 |                         64 |                  713.5 |
| ft_tiny             | nf4_qlora   | qat_ft        |               52.5 |                         64 |                 1454.8 |
| full_base           | fp16        | full          |              138.5 |                          0 |                 1194.9 |
| full_base           | int8_bnb    | full          |               96.5 |                         96 |                 4282.8 |
| full_base           | nf4_bnb     | full          |               75.5 |                         96 |                 1134.6 |
| full_base           | nf4_qlora   | qat_full      |               83.7 |                         96 |                 1944.7 |
| full_small          | fp16        | full          |              461.1 |                          0 |                 3387.6 |
| full_small          | int8_bnb    | full          |              272.1 |                        192 |                10095.6 |
| full_small          | nf4_bnb     | full          |              177.6 |                        192 |                 3116   |
| full_small          | nf4_qlora   | qat_full      |              202.3 |                        192 |                 3522.6 |
| fulltl_base_g0.5    | fp16        | fulltl        |              138.5 |                          0 |                 1195.8 |
| fulltl_base_g0r     | fp16        | fulltl        |              138.5 |                          0 |                 1195.8 |
| fulltl_base_g1      | fp16        | fulltl        |              138.5 |                          0 |                 1195.8 |
| fulltl_base_l0.4_g0 | fp16        | fulltl        |              138.5 |                          0 |                 1195.8 |
| fulltl_base_l0.4_g3 | fp16        | fulltl        |              138.5 |                          0 |                 1195.8 |
| fulltl_base_l1_g0   | fp16        | fulltl        |              138.5 |                          0 |                 1195.8 |
| fulltl_base_l1_g3   | fp16        | fulltl        |              138.5 |                          0 |                 1195.8 |
| rw_base             | fp16        | rw            |              138.5 |                          0 |                 1195   |
| rw_base             | int8_bnb    | rw            |               96.5 |                         96 |                 4282.8 |
| rw_base             | nf4_bnb     | rw            |               75.5 |                         96 |                 1134.6 |
| semonly_base        | fp16        | semonly       |              138.5 |                          0 |                 1195   |
| semonly_base        | int8_bnb    | semonly       |               96.5 |                         96 |                 4282.8 |
| semonly_base        | nf4_bnb     | semonly       |               75.5 |                         96 |                 1134.6 |
| small               | fp16        | offshelf_both |              461.1 |                          0 |                 3413   |
| small               | int8_bnb    | offshelf_free |              272.1 |                        192 |                10104.6 |
| small               | nf4_bnb     | offshelf_free |              177.6 |                        192 |                 3126.5 |
| sub_small           | fp16        | sub           |              461.1 |                          0 |                 3387.6 |
| sub_small           | int8_bnb    | sub           |              272.1 |                        192 |                10096   |
| sub_small           | nf4_bnb     | sub           |              177.6 |                        192 |                 3116   |
| tiny                | fp16        | offshelf_both |               72   |                          0 |                  737   |
| tiny                | int8_bnb    | offshelf_free |               56.3 |                         64 |                 1011.5 |
| tiny                | nf4_bnb     | offshelf_free |               48.4 |                         64 |                  713.5 |

## Deployed (CPU, MacBook M2 Pro, 4 threads): real-time factor, peak RSS, file size

| tag            | format    | mode            |   threads |   audio_s |   seconds |    rtf |   peak_rss_mb |   file_mb |   n_clips |
|:---------------|:----------|:----------------|----------:|----------:|----------:|-------:|--------------:|----------:|----------:|
| full_small     | ct2_int8  | plain           |         4 |     258.9 |    192.03 | 0.7419 |         839.2 |     237.7 |       100 |
| full_small     | ggml_q4_0 | plain           |         4 |     258.9 |    151.22 | 0.5842 |         315.3 |     138.7 |       100 |
| full_small     | ggml_q5_0 | plain           |         4 |     258.9 |    406.43 | 1.5701 |         348.9 |     167.1 |       100 |
| full_small     | ggml_q8_0 | plain           |         4 |     258.9 |    148.34 | 0.5731 |         426.9 |     252.2 |       100 |
| full_small     | ggml_f16  | plain           |         4 |     258.9 |    336.29 | 1.2992 |         650.2 |     465   |       100 |
| offshelf_base  | ct2_int8  | plain           |         4 |     258.9 |     59.04 | 0.2281 |         491.5 |      73.9 |       100 |
| offshelf_base  | ggml_q4_0 | plain           |         4 |     258.9 |     51.51 | 0.199  |         153.5 |      44.3 |       100 |
| offshelf_base  | ggml_q5_0 | plain           |         4 |     258.9 |    111.72 | 0.4316 |         162.3 |      52.7 |       100 |
| offshelf_base  | ggml_q8_0 | plain           |         4 |     258.9 |     50.63 | 0.1956 |         191.3 |      78   |       100 |
| offshelf_base  | ggml_f16  | plain           |         4 |     258.9 |     92.42 | 0.357  |         261.7 |     141.1 |       100 |
| offshelf_small | ct2_int8  | plain           |         4 |     258.9 |    203.32 | 0.7855 |         853.6 |     237.7 |       100 |
| offshelf_small | ggml_q4_0 | plain           |         4 |     258.9 |    154.45 | 0.5967 |         317   |     138.7 |       100 |
| offshelf_small | ggml_q5_0 | plain           |         4 |     258.9 |    410.02 | 1.584  |         343.7 |     167.1 |       100 |
| offshelf_small | ggml_q8_0 | plain           |         4 |     258.9 |    150.33 | 0.5808 |         427.3 |     252.2 |       100 |
| offshelf_small | ggml_f16  | plain           |         4 |     258.9 |    328.97 | 1.2709 |         643.8 |     465   |       100 |
| offshelf_tiny  | ct2_int8  | plain           |         4 |     258.9 |     28.07 | 0.1084 |         373.1 |      39.7 |       100 |
| offshelf_tiny  | ggml_q4_0 | plain           |         4 |     258.9 |     43.98 | 0.1699 |         109   |      24.1 |       100 |
| offshelf_tiny  | ggml_q5_0 | plain           |         4 |     258.9 |     48.85 | 0.1887 |         113.3 |      28.5 |       100 |
| offshelf_tiny  | ggml_q8_0 | plain           |         4 |     258.9 |     26.44 | 0.1022 |         128.1 |      41.5 |       100 |
| offshelf_tiny  | ggml_f16  | plain           |         4 |     258.9 |     44.17 | 0.1706 |         164.3 |      74.1 |       100 |
| qat_ft_small   | ct2_int8  | plain           |         4 |     258.9 |    188.16 | 0.7269 |         847.5 |     234.3 |       100 |
| qat_ft_small   | ggml_q4_0 | plain           |         4 |     258.9 |    146.44 | 0.5657 |         322.1 |     138.7 |       100 |
| qat_ft_small   | ggml_q5_0 | plain           |         4 |     258.9 |    406.53 | 1.5705 |         345.4 |     167.1 |       100 |
| qat_ft_small   | ggml_q8_0 | plain           |         4 |     258.9 |    147.93 | 0.5715 |         425.3 |     252.2 |       100 |
| qat_ft_small   | ggml_f16  | plain           |         4 |     258.9 |    328.15 | 1.2677 |         649.3 |     465   |       100 |
| qat_full_small | ct2_int8  | plain           |         4 |     258.9 |    191.92 | 0.7414 |         847.4 |     234.3 |       100 |
| qat_full_small | ggml_q4_0 | plain           |         4 |     258.9 |    147.56 | 0.57   |         322.4 |     138.7 |       100 |
| qat_full_small | ggml_q5_0 | plain           |         4 |     258.9 |    405.95 | 1.5683 |         343.7 |     167.1 |       100 |
| qat_full_small | ggml_q8_0 | plain           |         4 |     258.9 |    148.74 | 0.5746 |         425.2 |     252.2 |       100 |
| qat_full_small | ggml_f16  | plain           |         4 |     258.9 |    326.14 | 1.2599 |         656.8 |     465   |       100 |
| ft_base        | ct2_int8  | plain           |         4 |     258.9 |     56.02 | 0.2164 |         482.1 |      73.9 |       100 |
| ft_base        | ct2_int8  | lid             |         4 |     258.9 |     78.8  | 0.3044 |         492.9 |      73.9 |       100 |
| ft_base        | ct2_int8  | rerank_ondevice |         4 |     258.9 |    450.13 | 1.7389 |         491.1 |      73.9 |       100 |
| ft_base        | ggml_q4_0 | plain           |         4 |     258.9 |     43.92 | 0.1697 |         156.6 |      44.3 |       100 |
| ft_base        | ggml_q5_0 | plain           |         4 |     258.9 |    104.19 | 0.4025 |         164.5 |      52.7 |       100 |
| ft_base        | ggml_q8_0 | plain           |         4 |     258.9 |     43.82 | 0.1693 |         189.6 |      78   |       100 |
| ft_base        | ggml_f16  | plain           |         4 |     258.9 |     84.45 | 0.3262 |         254.9 |     141.1 |       100 |
| ft_small       | ct2_int8  | plain           |         4 |     258.9 |    188.71 | 0.729  |         840.6 |     237.7 |       100 |
| ft_small       | ggml_q4_0 | plain           |         4 |     258.9 |    146.59 | 0.5663 |         320.4 |     138.7 |       100 |
| ft_small       | ggml_q5_0 | plain           |         4 |     258.9 |    405.13 | 1.5651 |         350.2 |     167.1 |       100 |
| ft_small       | ggml_q8_0 | plain           |         4 |     258.9 |    147.55 | 0.57   |         425   |     252.2 |       100 |
| ft_small       | ggml_f16  | plain           |         4 |     258.9 |    325.39 | 1.2571 |         650.1 |     465   |       100 |
| full_base      | ct2_int8  | plain           |         4 |     258.9 |     54.27 | 0.2097 |         493.6 |      73.9 |       100 |
| full_base      | ct2_int8  | lid             |         4 |     258.9 |     76.67 | 0.2962 |         508.2 |      73.9 |       100 |
| full_base      | ct2_int8  | rerank_ondevice |         4 |     258.9 |    455.85 | 1.761  |         508.1 |      73.9 |       100 |
| full_base      | ggml_q4_0 | plain           |         4 |     258.9 |     44.48 | 0.1718 |         161   |      44.3 |       100 |
| full_base      | ggml_q5_0 | plain           |         4 |     258.9 |    107.29 | 0.4145 |         171.3 |      52.7 |       100 |
| full_base      | ggml_q8_0 | plain           |         4 |     258.9 |     45.51 | 0.1758 |         195.8 |      78   |       100 |
| full_base      | ggml_f16  | plain           |         4 |     258.9 |     86.94 | 0.3359 |         261.3 |     141.1 |       100 |
| qat_ft_base    | ct2_int8  | plain           |         4 |     258.9 |     54.3  | 0.2098 |         491.7 |      71.8 |       100 |
| qat_ft_base    | ggml_q4_0 | plain           |         4 |     258.9 |     44.25 | 0.1709 |         160.7 |      44.3 |       100 |
| qat_ft_tiny    | ct2_int8  | plain           |         4 |     258.9 |     27.46 | 0.1061 |         377.1 |      38.1 |       100 |
| qat_ft_tiny    | ggml_q4_0 | plain           |         4 |     258.9 |     21.87 | 0.0845 |         117.8 |      24.1 |       100 |
| qat_full_base  | ct2_int8  | plain           |         4 |     258.9 |     54.71 | 0.2113 |         489.9 |      71.8 |       100 |
| qat_full_base  | ggml_q4_0 | plain           |         4 |     258.9 |     43.59 | 0.1684 |         160.5 |      44.3 |       100 |
| ft_tiny        | ct2_int8  | plain           |         4 |     258.9 |     27.57 | 0.1065 |         376.8 |      39.7 |       100 |
| ft_tiny        | ggml_q4_0 | plain           |         4 |     258.9 |     22.37 | 0.0864 |         118.1 |      24.1 |       100 |
| ft_tiny        | ggml_q5_0 | plain           |         4 |     258.9 |     45.83 | 0.1771 |         122.5 |      28.5 |       100 |
| ft_tiny        | ggml_q8_0 | plain           |         4 |     258.9 |     22.32 | 0.0862 |         135.1 |      41.5 |       100 |
| ft_tiny        | ggml_f16  | plain           |         4 |     258.9 |     38.4  | 0.1483 |         171.4 |      74.1 |       100 |
