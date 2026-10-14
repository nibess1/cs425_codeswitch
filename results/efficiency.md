# Efficiency

## Simulated (bitsandbytes on GPU): measured memory footprint

Only linear layers are quantized; conv front-end, embeddings, layer norms and proj_out stay 16-bit.

| model   | precision   | method        |   gpu_footprint_mb |   quantized_linear_modules |   peak_gpu_mb_decoding |
|:--------|:------------|:--------------|-------------------:|---------------------------:|-----------------------:|
| base    | fp16        | offshelf_both |              138.5 |                          0 |                 1196.9 |
| tiny    | fp16        | offshelf_both |               72   |                          0 |                  737   |
| tiny    | int8_bnb    | offshelf_free |               56.3 |                         64 |                 1011.5 |
| tiny    | nf4_bnb     | offshelf_free |               48.4 |                         64 |                  713.5 |

## Deployed (CPU, MacBook M2 Pro, 4 threads): real-time factor, peak RSS, file size

_not run yet_
