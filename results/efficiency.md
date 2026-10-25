# Efficiency

## Simulated (bitsandbytes on GPU): measured memory footprint

Only linear layers are quantized; conv front-end, embeddings, layer norms and proj_out stay 16-bit.

| model        | precision   | method        |   gpu_footprint_mb |   quantized_linear_modules |   peak_gpu_mb_decoding |
|:-------------|:------------|:--------------|-------------------:|---------------------------:|-----------------------:|
| base         | fp16        | offshelf_both |              138.5 |                          0 |                 1196.9 |
| base         | int8_bnb    | offshelf_free |               96.5 |                         96 |                 4285.8 |
| base         | nf4_bnb     | offshelf_free |               75.5 |                         96 |                 1136.4 |
| ediff_base   | fp16        | eabl          |              138.5 |                          0 |                 1195   |
| emix_base    | fp16        | eabl          |              138.5 |                          0 |                 1195   |
| esame_base   | fp16        | eabl          |              138.5 |                          0 |                 1195   |
| ft_base      | fp16        | ft            |              138.5 |                          0 |                 1195.9 |
| ft_base      | int8_bnb    | ft            |               96.5 |                         96 |                 4285.8 |
| ft_base      | nf4_bnb     | ft            |               75.5 |                         96 |                 1136.4 |
| ft_base      | nf4_qlora   | qat_ft        |               83.7 |                         96 |                 1944.7 |
| ft_small     | fp16        | ft            |              461.1 |                          0 |                 3412   |
| ft_small     | int8_bnb    | ft            |              272.1 |                        192 |                10105.1 |
| ft_small     | nf4_bnb     | ft            |              177.6 |                        192 |                 3126.5 |
| ft_small     | nf4_qlora   | qat_ft        |              202.3 |                        192 |                 3522.6 |
| ft_tiny      | fp16        | ft            |               72   |                          0 |                  737   |
| ft_tiny      | int8_bnb    | ft            |               56.3 |                         64 |                 3216.3 |
| ft_tiny      | nf4_bnb     | ft            |               48.4 |                         64 |                  713.5 |
| ft_tiny      | nf4_qlora   | qat_ft        |               52.5 |                         64 |                 1454.8 |
| full_base    | fp16        | full          |              138.5 |                          0 |                 1194.9 |
| full_base    | int8_bnb    | full          |               96.5 |                         96 |                 4282.8 |
| full_base    | nf4_bnb     | full          |               75.5 |                         96 |                 1134.6 |
| full_base    | nf4_qlora   | qat_full      |               83.7 |                         96 |                 1944.7 |
| full_small   | fp16        | full          |              461.1 |                          0 |                 3387.6 |
| full_small   | int8_bnb    | full          |              272.1 |                        192 |                10095.6 |
| full_small   | nf4_bnb     | full          |              177.6 |                        192 |                 3116   |
| full_small   | nf4_qlora   | qat_full      |              202.3 |                        192 |                 3522.6 |
| rw_base      | fp16        | rw            |              138.5 |                          0 |                 1195   |
| rw_base      | int8_bnb    | rw            |               96.5 |                         96 |                 4282.8 |
| rw_base      | nf4_bnb     | rw            |               75.5 |                         96 |                 1134.6 |
| semonly_base | fp16        | semonly       |              138.5 |                          0 |                 1195   |
| semonly_base | int8_bnb    | semonly       |               96.5 |                         96 |                 4282.8 |
| semonly_base | nf4_bnb     | semonly       |               75.5 |                         96 |                 1134.6 |
| small        | fp16        | offshelf_both |              461.1 |                          0 |                 3413   |
| small        | int8_bnb    | offshelf_free |              272.1 |                        192 |                10104.6 |
| small        | nf4_bnb     | offshelf_free |              177.6 |                        192 |                 3126.5 |
| sub_small    | fp16        | sub           |              461.1 |                          0 |                 3387.6 |
| sub_small    | int8_bnb    | sub           |              272.1 |                        192 |                10096   |
| sub_small    | nf4_bnb     | sub           |              177.6 |                        192 |                 3116   |
| tiny         | fp16        | offshelf_both |               72   |                          0 |                  737   |
| tiny         | int8_bnb    | offshelf_free |               56.3 |                         64 |                 1011.5 |
| tiny         | nf4_bnb     | offshelf_free |               48.4 |                         64 |                  713.5 |

## Deployed (CPU, MacBook M2 Pro, 4 threads): real-time factor, peak RSS, file size

_not run yet_
