# Quantization (RQ3)

## Accuracy by precision (test sets; cs_ = code-switched utterances only)

| model        | method        | precision   | set           | MER                  | cs_E_emb             | cs_E_mat             |   CWER |   size_mb |
|:-------------|:--------------|:------------|:--------------|:---------------------|:---------------------|:---------------------|-------:|----------:|
| base         | offshelf_both | fp16        | ascend_test   | 101.7 [95.6, 108.2]  | 83.9 [64.8, 117.7]   | 109.4 [100.9, 119.1] |   85.2 |     138.5 |
| base         | offshelf_both | fp16        | nsc_test      | 80.0 [77.7, 82.4]    | 115.0 [108.8, 121.4] | 88.4 [83.4, 93.2]    |   86.7 |     138.5 |
| base         | offshelf_both | fp16        | nsc_diff_test | 77.7 [74.1, 81.9]    | 120.5 [112.3, 130.0] | 93.5 [86.4, 101.8]   |   78.2 |     138.5 |
| base         | offshelf_en   | fp16        | ascend_test   | 94.1 [90.7, 98.1]    | 66.4 [60.1, 73.0]    | 104.8 [98.9, 112.1]  |   84   |     138.5 |
| base         | offshelf_en   | fp16        | nsc_test      | 70.2 [68.7, 71.9]    | 103.2 [99.8, 107.2]  | 75.9 [72.8, 79.3]    |   84.8 |     138.5 |
| base         | offshelf_en   | fp16        | nsc_diff_test | 67.0 [64.5, 69.8]    | 106.1 [100.0, 112.2] | 76.9 [72.5, 81.7]    |   73.6 |     138.5 |
| base         | offshelf_free | fp16        | ascend_test   | 35.8 [34.0, 37.8]    | 83.1 [71.2, 98.3]    | 29.6 [27.1, 32.6]    |   53.1 |     138.5 |
| base         | offshelf_free | fp16        | nsc_test      | 71.1 [68.9, 73.4]    | 101.8 [96.1, 107.5]  | 75.5 [71.2, 80.1]    |   78.6 |     138.5 |
| base         | offshelf_free | fp16        | nsc_diff_test | 64.2 [60.9, 67.7]    | 103.1 [90.6, 115.7]  | 65.2 [60.1, 70.7]    |   69.2 |     138.5 |
| base         | offshelf_free | int8_bnb    | ascend_test   | 37.3 [34.9, 40.3]    | 88.7 [75.0, 105.8]   | 31.7 [28.0, 36.7]    |   54.3 |      96.5 |
| base         | offshelf_free | int8_bnb    | nsc_test      | 70.7 [68.5, 73.0]    | 102.2 [96.8, 107.8]  | 77.0 [72.5, 82.2]    |   77.9 |      96.5 |
| base         | offshelf_free | int8_bnb    | nsc_diff_test | 64.2 [61.0, 67.6]    | 104.6 [92.7, 116.7]  | 65.8 [60.3, 71.7]    |   69.1 |      96.5 |
| base         | offshelf_free | nf4_bnb     | ascend_test   | 41.5 [38.7, 44.5]    | 88.0 [78.8, 98.2]    | 37.5 [32.8, 43.1]    |   55.6 |      75.5 |
| base         | offshelf_free | nf4_bnb     | nsc_test      | 78.8 [76.4, 81.3]    | 108.5 [102.1, 114.9] | 82.6 [77.6, 87.3]    |   81.5 |      75.5 |
| base         | offshelf_free | nf4_bnb     | nsc_diff_test | 70.6 [67.2, 74.0]    | 108.4 [98.4, 119.9]  | 80.3 [73.2, 87.6]    |   69.5 |      75.5 |
| base         | offshelf_free | ct2_f32     | ascend_test   | 35.9 [34.2, 37.7]    | 78.7 [70.0, 87.0]    | 30.6 [27.6, 34.1]    |   51.9 |     139.5 |
| base         | offshelf_free | ct2_f32     | nsc_test      | 64.8 [62.9, 66.6]    | 90.7 [86.5, 95.0]    | 64.0 [60.0, 68.1]    |   79.9 |     139.5 |
| base         | offshelf_free | ct2_f32     | nsc_diff_test | 56.9 [54.6, 59.3]    | 86.8 [80.3, 94.5]    | 57.4 [53.1, 61.6]    |   73.4 |     139.5 |
| base         | offshelf_free | ct2_int8    | ascend_test   | 37.3 [35.0, 39.9]    | 83.6 [74.3, 93.4]    | 30.7 [27.7, 34.3]    |   49.4 |      73.9 |
| base         | offshelf_free | ct2_int8    | nsc_test      | 64.0 [62.3, 65.9]    | 89.9 [85.8, 93.9]    | 63.1 [59.2, 66.8]    |   81   |      73.9 |
| base         | offshelf_free | ct2_int8    | nsc_diff_test | 56.8 [54.5, 59.3]    | 83.6 [77.9, 90.3]    | 57.3 [53.0, 61.5]    |   72.5 |      73.9 |
| base         | offshelf_free | ggml_f16    | ascend_test   | 36.2 [34.6, 38.0]    | 77.6 [69.1, 85.6]    | 29.6 [27.1, 32.6]    |   49.4 |     141.1 |
| base         | offshelf_free | ggml_f16    | nsc_test      | 82.0 [80.8, 83.3]    | 89.9 [86.7, 93.2]    | 75.0 [72.0, 77.9]    |   87.4 |     141.1 |
| base         | offshelf_free | ggml_f16    | nsc_diff_test | 76.3 [74.4, 78.0]    | 84.2 [80.0, 88.6]    | 68.4 [64.4, 72.6]    |   87   |     141.1 |
| base         | offshelf_free | ggml_q8_0   | ascend_test   | 36.2 [34.6, 37.9]    | 78.2 [70.0, 86.4]    | 29.4 [26.9, 32.4]    |   49.4 |      78   |
| base         | offshelf_free | ggml_q8_0   | nsc_test      | 81.6 [80.4, 82.8]    | 89.0 [86.2, 92.0]    | 74.6 [71.6, 77.1]    |   87.4 |      78   |
| base         | offshelf_free | ggml_q8_0   | nsc_diff_test | 76.6 [74.7, 78.4]    | 83.8 [79.5, 88.0]    | 70.7 [67.1, 74.5]    |   87.3 |      78   |
| base         | offshelf_free | ggml_q5_0   | ascend_test   | 37.1 [35.4, 39.0]    | 81.3 [72.3, 90.1]    | 30.3 [27.6, 33.4]    |   58   |      52.7 |
| base         | offshelf_free | ggml_q5_0   | nsc_test      | 79.8 [78.6, 81.0]    | 87.5 [84.4, 90.5]    | 73.1 [70.3, 75.7]    |   86.7 |      52.7 |
| base         | offshelf_free | ggml_q5_0   | nsc_diff_test | 73.9 [71.9, 75.8]    | 83.6 [79.4, 88.3]    | 66.5 [62.4, 70.7]    |   84.3 |      52.7 |
| base         | offshelf_free | ggml_q4_0   | ascend_test   | 39.7 [37.9, 41.7]    | 92.4 [84.2, 100.9]   | 33.2 [30.3, 36.6]    |   54.3 |      44.3 |
| base         | offshelf_free | ggml_q4_0   | nsc_test      | 82.6 [81.2, 84.2]    | 95.7 [90.1, 104.3]   | 77.0 [74.0, 80.1]    |   89.1 |      44.3 |
| base         | offshelf_free | ggml_q4_0   | nsc_diff_test | 77.5 [75.2, 80.0]    | 89.2 [84.5, 93.9]    | 71.6 [66.8, 76.6]    |   87   |      44.3 |
| base         | offshelf_zh   | fp16        | ascend_test   | 49.9 [44.0, 56.7]    | 101.9 [80.2, 127.8]  | 41.6 [30.3, 56.0]    |   59.3 |     138.5 |
| base         | offshelf_zh   | fp16        | nsc_test      | 140.4 [136.6, 144.4] | 117.0 [109.4, 124.4] | 119.9 [113.6, 126.5] |   88.5 |     138.5 |
| base         | offshelf_zh   | fp16        | nsc_diff_test | 134.3 [128.7, 140.3] | 116.2 [101.4, 132.2] | 106.5 [98.1, 115.3]  |   87.7 |     138.5 |
| ft_base      | ft            | fp16        | ascend_test   | 21.3 [19.3, 23.6]    | 55.3 [36.3, 86.2]    | 15.4 [13.7, 17.1]    |   24.7 |     138.5 |
| ft_base      | ft            | fp16        | nsc_test      | 28.4 [27.2, 29.7]    | 31.0 [28.3, 34.4]    | 27.8 [25.0, 31.0]    |   20   |     138.5 |
| ft_base      | ft            | fp16        | nsc_diff_test | 22.3 [21.0, 23.8]    | 24.0 [22.2, 25.8]    | 21.0 [18.9, 23.7]    |   18.6 |     138.5 |
| ft_base      | ft            | int8_bnb    | ascend_test   | 20.4 [19.2, 21.6]    | 40.2 [34.5, 45.9]    | 15.7 [13.9, 17.4]    |   22.2 |      96.5 |
| ft_base      | ft            | int8_bnb    | nsc_test      | 28.2 [27.0, 29.4]    | 29.6 [27.7, 31.4]    | 28.3 [25.4, 31.6]    |   20.5 |      96.5 |
| ft_base      | ft            | int8_bnb    | nsc_diff_test | 22.1 [21.0, 23.5]    | 23.9 [22.0, 25.8]    | 21.1 [19.0, 23.8]    |   18.8 |      96.5 |
| ft_base      | ft            | nf4_bnb     | ascend_test   | 21.6 [20.4, 22.8]    | 45.5 [39.2, 51.5]    | 16.6 [14.8, 18.2]    |   24.7 |      75.5 |
| ft_base      | ft            | nf4_bnb     | nsc_test      | 29.7 [28.4, 30.9]    | 31.8 [29.4, 34.5]    | 27.9 [25.4, 30.7]    |   22.2 |      75.5 |
| ft_base      | ft            | nf4_bnb     | nsc_diff_test | 23.8 [22.4, 25.3]    | 26.9 [24.0, 31.1]    | 22.3 [20.4, 24.5]    |   18.3 |      75.5 |
| ft_base      | ft            | ct2_int8    | ascend_test   | 20.8 [19.3, 22.3]    | 47.1 [35.9, 62.1]    | 16.0 [14.2, 17.7]    |   27.2 |      73.9 |
| ft_base      | ft            | ct2_int8    | nsc_test      | 27.6 [26.4, 28.7]    | 31.2 [28.5, 34.1]    | 24.5 [22.3, 26.9]    |   21   |      73.9 |
| ft_base      | ft            | ct2_int8    | nsc_diff_test | 22.8 [21.3, 24.3]    | 25.3 [22.5, 28.5]    | 21.7 [19.2, 24.7]    |   16.9 |      73.9 |
| ft_base      | ft            | ggml_q8_0   | ascend_test   | 20.6 [19.4, 21.8]    | 40.2 [34.7, 45.8]    | 16.0 [14.3, 17.8]    |   22.2 |      78   |
| ft_base      | ft            | ggml_q8_0   | nsc_test      | 25.6 [24.7, 26.5]    | 28.9 [26.8, 31.1]    | 21.8 [20.2, 23.3]    |   19.9 |      78   |
| ft_base      | ft            | ggml_q8_0   | nsc_diff_test | 20.9 [19.7, 22.1]    | 23.9 [21.2, 26.8]    | 18.6 [16.7, 20.4]    |   16.6 |      78   |
| ft_base      | ft            | ggml_q5_0   | ascend_test   | 21.4 [20.1, 22.6]    | 43.7 [37.2, 49.5]    | 16.7 [14.8, 18.5]    |   29.6 |      52.7 |
| ft_base      | ft            | ggml_q5_0   | nsc_test      | 26.4 [25.5, 27.4]    | 29.7 [27.6, 32.0]    | 22.7 [20.7, 24.9]    |   19.9 |      52.7 |
| ft_base      | ft            | ggml_q5_0   | nsc_diff_test | 21.1 [19.9, 22.3]    | 24.7 [21.9, 27.9]    | 18.8 [17.2, 20.6]    |   15.1 |      52.7 |
| ft_base      | ft            | ggml_q4_0   | ascend_test   | 22.6 [21.3, 23.9]    | 46.2 [39.7, 52.7]    | 17.8 [16.0, 19.7]    |   30.9 |      44.3 |
| ft_base      | ft            | ggml_q4_0   | nsc_test      | 28.2 [27.3, 29.1]    | 32.2 [29.8, 34.9]    | 23.7 [22.2, 25.1]    |   23   |      44.3 |
| ft_base      | ft            | ggml_q4_0   | nsc_diff_test | 22.5 [21.4, 23.6]    | 26.2 [23.5, 29.0]    | 20.8 [19.0, 22.6]    |   17.5 |      44.3 |
| ft_small     | ft            | fp16        | ascend_test   | 14.8 [13.7, 15.8]    | 30.6 [25.3, 35.9]    | 11.2 [9.8, 12.6]     |   19.8 |     461.1 |
| ft_small     | ft            | fp16        | nsc_test      | 21.4 [20.6, 22.2]    | 23.7 [21.6, 26.3]    | 19.1 [18.1, 20.3]    |   14.1 |     461.1 |
| ft_small     | ft            | fp16        | nsc_diff_test | 19.0 [17.6, 20.6]    | 22.9 [19.2, 28.2]    | 17.3 [15.3, 19.8]    |   12.5 |     461.1 |
| ft_small     | ft            | int8_bnb    | ascend_test   | 14.7 [13.6, 15.7]    | 30.2 [24.9, 35.5]    | 10.9 [9.4, 12.3]     |   18.5 |     272.1 |
| ft_small     | ft            | int8_bnb    | nsc_test      | 21.4 [20.6, 22.3]    | 24.6 [22.2, 27.5]    | 19.3 [18.0, 20.6]    |   13.8 |     272.1 |
| ft_small     | ft            | int8_bnb    | nsc_diff_test | 19.2 [17.8, 21.0]    | 24.0 [19.5, 29.8]    | 17.8 [15.7, 20.4]    |   13.4 |     272.1 |
| ft_small     | ft            | nf4_bnb     | ascend_test   | 15.2 [14.1, 16.4]    | 31.4 [25.9, 36.8]    | 11.6 [10.2, 13.1]    |   12.3 |     177.6 |
| ft_small     | ft            | nf4_bnb     | nsc_test      | 23.1 [22.1, 24.2]    | 24.9 [22.7, 27.5]    | 21.7 [20.0, 23.6]    |   15.3 |     177.6 |
| ft_small     | ft            | nf4_bnb     | nsc_diff_test | 20.0 [18.4, 21.9]    | 21.4 [19.3, 23.5]    | 19.4 [16.7, 23.0]    |   14   |     177.6 |
| ft_small     | ft            | ct2_int8    | ascend_test   | 14.8 [13.7, 15.9]    | 31.0 [25.5, 36.3]    | 11.1 [9.7, 12.5]     |   19.8 |     237.7 |
| ft_small     | ft            | ct2_int8    | nsc_test      | 21.6 [20.7, 22.5]    | 24.3 [22.0, 26.7]    | 19.0 [17.4, 20.5]    |   14.1 |     237.7 |
| ft_small     | ft            | ct2_int8    | nsc_diff_test | 18.7 [17.4, 20.0]    | 21.7 [18.8, 24.6]    | 17.1 [15.1, 19.1]    |   13.6 |     237.7 |
| ft_small     | ft            | ggml_q8_0   | ascend_test   | 14.9 [13.9, 16.0]    | 31.1 [25.6, 36.2]    | 11.4 [9.9, 12.9]     |   18.5 |     252.2 |
| ft_small     | ft            | ggml_q8_0   | nsc_test      | 20.7 [19.9, 21.5]    | 23.4 [21.2, 25.7]    | 17.7 [16.4, 19.0]    |   13.5 |     252.2 |
| ft_small     | ft            | ggml_q8_0   | nsc_diff_test | 17.2 [16.1, 18.4]    | 20.2 [17.8, 22.5]    | 15.1 [13.4, 16.8]    |   12.4 |     252.2 |
| ft_small     | ft            | ggml_q5_0   | ascend_test   | 14.9 [13.8, 16.0]    | 31.4 [25.8, 37.2]    | 11.4 [9.9, 12.8]     |   17.3 |     167.1 |
| ft_small     | ft            | ggml_q5_0   | nsc_test      | 21.0 [20.1, 21.8]    | 24.5 [22.2, 26.9]    | 17.8 [16.4, 19.0]    |   14.4 |     167.1 |
| ft_small     | ft            | ggml_q5_0   | nsc_diff_test | 17.3 [16.2, 18.5]    | 20.3 [17.9, 22.6]    | 15.2 [13.6, 17.0]    |   11.8 |     167.1 |
| ft_small     | ft            | ggml_q4_0   | ascend_test   | 15.0 [13.9, 16.1]    | 27.6 [22.8, 32.5]    | 11.2 [9.7, 12.7]     |   14.8 |     138.7 |
| ft_small     | ft            | ggml_q4_0   | nsc_test      | 21.7 [20.8, 22.5]    | 24.7 [22.6, 26.9]    | 18.5 [17.1, 19.9]    |   14.4 |     138.7 |
| ft_small     | ft            | ggml_q4_0   | nsc_diff_test | 18.0 [16.9, 19.3]    | 21.0 [18.4, 23.5]    | 16.1 [14.3, 17.9]    |   12.4 |     138.7 |
| ft_tiny      | ft            | fp16        | ascend_test   | 26.1 [24.6, 27.5]    | 53.2 [46.0, 60.4]    | 20.7 [18.7, 22.9]    |   33.3 |      72   |
| ft_tiny      | ft            | fp16        | nsc_test      | 34.5 [33.3, 35.8]    | 36.2 [33.4, 39.2]    | 33.2 [30.6, 36.3]    |   29.7 |      72   |
| ft_tiny      | ft            | fp16        | nsc_diff_test | 26.8 [25.4, 28.4]    | 29.8 [27.5, 32.7]    | 25.5 [23.0, 28.8]    |   20.6 |      72   |
| ft_tiny      | ft            | int8_bnb    | ascend_test   | 26.8 [25.1, 29.1]    | 53.0 [46.3, 60.3]    | 22.6 [19.2, 27.3]    |   33.3 |      56.3 |
| ft_tiny      | ft            | int8_bnb    | nsc_test      | 34.1 [33.0, 35.3]    | 35.7 [33.2, 38.5]    | 32.8 [30.3, 35.6]    |   29.2 |      56.3 |
| ft_tiny      | ft            | int8_bnb    | nsc_diff_test | 26.7 [25.6, 27.9]    | 29.6 [27.7, 31.9]    | 24.3 [22.9, 25.8]    |   20.9 |      56.3 |
| ft_tiny      | ft            | nf4_bnb     | ascend_test   | 32.8 [30.1, 36.1]    | 68.7 [60.3, 76.9]    | 25.3 [23.0, 27.8]    |   46.9 |      48.4 |
| ft_tiny      | ft            | nf4_bnb     | nsc_test      | 43.6 [41.6, 45.6]    | 49.1 [42.7, 56.7]    | 40.0 [37.2, 43.2]    |   32.5 |      48.4 |
| ft_tiny      | ft            | nf4_bnb     | nsc_diff_test | 34.4 [32.3, 36.6]    | 38.2 [33.3, 44.0]    | 32.2 [28.8, 35.8]    |   25.8 |      48.4 |
| ft_tiny      | ft            | ct2_int8    | ascend_test   | 26.7 [25.2, 28.4]    | 51.8 [44.9, 58.6]    | 21.9 [19.4, 24.8]    |   33.3 |      39.7 |
| ft_tiny      | ft            | ct2_int8    | nsc_test      | 33.5 [32.3, 34.6]    | 36.7 [33.8, 40.1]    | 29.7 [27.8, 31.7]    |   29.5 |      39.7 |
| ft_tiny      | ft            | ct2_int8    | nsc_diff_test | 27.0 [25.6, 28.4]    | 31.3 [28.5, 34.6]    | 24.8 [22.5, 27.0]    |   19.6 |      39.7 |
| ft_tiny      | ft            | ggml_q8_0   | ascend_test   | 26.2 [24.8, 27.7]    | 52.5 [46.0, 59.6]    | 20.8 [18.7, 23.0]    |   35.8 |      41.5 |
| ft_tiny      | ft            | ggml_q8_0   | nsc_test      | 31.6 [30.6, 32.5]    | 33.8 [31.4, 36.5]    | 27.3 [25.8, 28.8]    |   28.3 |      41.5 |
| ft_tiny      | ft            | ggml_q8_0   | nsc_diff_test | 25.5 [24.3, 26.8]    | 29.0 [26.2, 32.0]    | 23.1 [21.2, 24.9]    |   17.8 |      41.5 |
| ft_tiny      | ft            | ggml_q5_0   | ascend_test   | 27.8 [26.2, 29.3]    | 54.9 [48.3, 61.7]    | 22.1 [19.9, 24.5]    |   35.8 |      28.5 |
| ft_tiny      | ft            | ggml_q5_0   | nsc_test      | 33.9 [32.9, 34.8]    | 36.4 [33.9, 39.2]    | 29.6 [27.9, 31.2]    |   29.2 |      28.5 |
| ft_tiny      | ft            | ggml_q5_0   | nsc_diff_test | 26.9 [25.7, 28.1]    | 30.1 [27.3, 33.2]    | 24.2 [22.3, 26.1]    |   20.2 |      28.5 |
| ft_tiny      | ft            | ggml_q4_0   | ascend_test   | 33.4 [31.7, 35.3]    | 60.4 [53.2, 67.9]    | 29.1 [26.4, 32.0]    |   42   |      24.1 |
| ft_tiny      | ft            | ggml_q4_0   | nsc_test      | 39.6 [38.3, 41.0]    | 43.3 [40.5, 46.0]    | 34.1 [32.2, 35.8]    |   32.4 |      24.1 |
| ft_tiny      | ft            | ggml_q4_0   | nsc_diff_test | 31.1 [29.8, 32.5]    | 36.1 [32.9, 39.4]    | 29.3 [27.3, 31.6]    |   26.3 |      24.1 |
| full_base    | full          | fp16        | ascend_test   | 20.3 [19.0, 21.4]    | 39.2 [33.7, 44.7]    | 15.6 [13.9, 17.3]    |   21   |     138.5 |
| full_base    | full          | fp16        | nsc_test      | 26.1 [25.5, 26.6]    | 27.4 [25.9, 28.9]    | 22.9 [22.0, 23.9]    |   20.2 |     138.5 |
| full_base    | full          | fp16        | nsc_diff_test | 21.1 [20.2, 22.0]    | 22.5 [20.7, 24.4]    | 19.9 [18.6, 21.4]    |   16.8 |     138.5 |
| full_base    | full          | int8_bnb    | ascend_test   | 20.7 [19.2, 22.2]    | 38.6 [32.4, 45.4]    | 16.7 [14.7, 19.2]    |   19.8 |      96.5 |
| full_base    | full          | int8_bnb    | nsc_test      | 26.2 [25.6, 26.8]    | 27.7 [26.1, 29.1]    | 22.9 [22.0, 23.9]    |   20.2 |      96.5 |
| full_base    | full          | int8_bnb    | nsc_diff_test | 21.1 [20.3, 22.0]    | 22.7 [20.9, 24.7]    | 20.0 [18.7, 21.4]    |   17.4 |      96.5 |
| full_base    | full          | nf4_bnb     | ascend_test   | 21.6 [20.3, 22.8]    | 41.6 [36.0, 47.5]    | 17.4 [15.5, 19.4]    |   28.4 |      75.5 |
| full_base    | full          | nf4_bnb     | nsc_test      | 27.8 [27.2, 28.4]    | 30.3 [28.7, 31.8]    | 24.7 [23.6, 25.8]    |   22.6 |      75.5 |
| full_base    | full          | nf4_bnb     | nsc_diff_test | 22.4 [21.5, 23.3]    | 24.6 [22.8, 26.6]    | 21.3 [19.9, 22.7]    |   18.9 |      75.5 |
| rw_base      | rw            | fp16        | ascend_test   | 20.5 [19.3, 21.7]    | 40.7 [34.8, 46.3]    | 15.7 [13.9, 17.4]    |   24.7 |     138.5 |
| rw_base      | rw            | fp16        | nsc_test      | 27.8 [26.8, 28.9]    | 28.3 [26.6, 30.1]    | 26.8 [24.3, 29.7]    |   20.7 |     138.5 |
| rw_base      | rw            | fp16        | nsc_diff_test | 21.8 [20.7, 23.1]    | 23.1 [21.3, 25.0]    | 21.0 [18.8, 23.7]    |   17.9 |     138.5 |
| rw_base      | rw            | int8_bnb    | ascend_test   | 20.4 [19.2, 21.6]    | 38.7 [33.0, 44.4]    | 15.8 [14.0, 17.5]    |   23.5 |      96.5 |
| rw_base      | rw            | int8_bnb    | nsc_test      | 27.9 [26.8, 29.0]    | 28.5 [26.3, 31.1]    | 26.4 [23.9, 29.4]    |   21.2 |      96.5 |
| rw_base      | rw            | int8_bnb    | nsc_diff_test | 22.0 [20.8, 23.3]    | 24.6 [21.8, 28.6]    | 21.2 [19.1, 23.9]    |   18.2 |      96.5 |
| rw_base      | rw            | nf4_bnb     | ascend_test   | 22.0 [20.7, 23.3]    | 41.8 [35.9, 48.3]    | 17.9 [15.7, 19.9]    |   25.9 |      75.5 |
| rw_base      | rw            | nf4_bnb     | nsc_test      | 29.6 [28.5, 30.8]    | 29.3 [27.7, 30.8]    | 27.7 [25.2, 30.3]    |   23.2 |      75.5 |
| rw_base      | rw            | nf4_bnb     | nsc_diff_test | 23.8 [22.5, 25.3]    | 24.7 [22.6, 26.8]    | 23.6 [21.0, 26.7]    |   18.8 |      75.5 |
| semonly_base | semonly       | fp16        | ascend_test   | 20.0 [18.8, 21.3]    | 40.6 [34.7, 46.6]    | 15.3 [13.7, 17.0]    |   21   |     138.5 |
| semonly_base | semonly       | fp16        | nsc_test      | 27.1 [26.1, 28.2]    | 29.3 [27.4, 31.3]    | 25.1 [23.0, 27.6]    |   20.8 |     138.5 |
| semonly_base | semonly       | fp16        | nsc_diff_test | 21.2 [20.3, 22.3]    | 23.6 [21.7, 25.5]    | 19.9 [18.5, 21.2]    |   16.5 |     138.5 |
| semonly_base | semonly       | int8_bnb    | ascend_test   | 20.1 [18.8, 21.3]    | 41.8 [35.5, 47.5]    | 15.3 [13.5, 17.0]    |   19.8 |      96.5 |
| semonly_base | semonly       | int8_bnb    | nsc_test      | 26.9 [26.0, 27.9]    | 28.7 [27.1, 30.5]    | 25.1 [23.0, 27.8]    |   20.3 |      96.5 |
| semonly_base | semonly       | int8_bnb    | nsc_diff_test | 21.3 [20.4, 22.3]    | 23.2 [21.4, 25.1]    | 19.8 [18.6, 21.2]    |   16.7 |      96.5 |
| semonly_base | semonly       | nf4_bnb     | ascend_test   | 22.1 [20.8, 23.3]    | 48.1 [41.8, 54.5]    | 17.4 [15.7, 19.1]    |   32.1 |      75.5 |
| semonly_base | semonly       | nf4_bnb     | nsc_test      | 28.7 [27.8, 29.7]    | 31.6 [29.5, 33.9]    | 27.0 [24.8, 29.6]    |   23.7 |      75.5 |
| semonly_base | semonly       | nf4_bnb     | nsc_diff_test | 22.7 [21.8, 23.7]    | 24.9 [23.1, 27.0]    | 21.7 [20.1, 23.6]    |   19.1 |      75.5 |
| small        | offshelf_both | fp16        | ascend_test   | 30.4 [26.7, 35.2]    | 88.5 [55.3, 141.1]   | 26.5 [21.3, 33.6]    |   40.7 |     461.1 |
| small        | offshelf_both | fp16        | nsc_test      | 64.5 [62.6, 66.8]    | 80.8 [76.3, 85.5]    | 60.6 [57.3, 64.3]    |   76.4 |     461.1 |
| small        | offshelf_both | fp16        | nsc_diff_test | 55.9 [53.1, 59.2]    | 73.6 [68.3, 80.2]    | 56.5 [51.5, 61.8]    |   71.2 |     461.1 |
| small        | offshelf_en   | fp16        | ascend_test   | 89.2 [87.2, 91.2]    | 58.7 [52.7, 65.0]    | 97.1 [94.7, 100.1]   |   80.2 |     461.1 |
| small        | offshelf_en   | fp16        | nsc_test      | 57.9 [56.6, 59.2]    | 94.1 [91.5, 97.6]    | 63.3 [60.7, 65.9]    |   77.6 |     461.1 |
| small        | offshelf_en   | fp16        | nsc_diff_test | 54.1 [52.4, 55.9]    | 91.2 [88.0, 94.9]    | 63.9 [60.7, 67.4]    |   70   |     461.1 |
| small        | offshelf_free | fp16        | ascend_test   | 26.3 [24.4, 28.1]    | 56.8 [49.5, 64.3]    | 19.9 [17.1, 23.5]    |   35.8 |     461.1 |
| small        | offshelf_free | fp16        | nsc_test      | 55.1 [53.5, 56.7]    | 76.7 [72.9, 81.3]    | 61.3 [57.9, 64.6]    |   70.6 |     461.1 |
| small        | offshelf_free | fp16        | nsc_diff_test | 48.8 [46.4, 51.3]    | 72.7 [66.5, 79.7]    | 55.5 [51.0, 60.3]    |   63.5 |     461.1 |
| small        | offshelf_free | int8_bnb    | ascend_test   | 26.9 [24.9, 29.1]    | 57.2 [50.2, 63.8]    | 21.5 [18.0, 26.4]    |   34.6 |     272.1 |
| small        | offshelf_free | int8_bnb    | nsc_test      | 54.9 [53.4, 56.5]    | 75.9 [71.7, 80.2]    | 60.0 [56.8, 63.0]    |   71.6 |     272.1 |
| small        | offshelf_free | int8_bnb    | nsc_diff_test | 48.8 [46.4, 51.2]    | 72.9 [66.7, 80.4]    | 56.1 [51.0, 61.4]    |   64   |     272.1 |
| small        | offshelf_free | nf4_bnb     | ascend_test   | 27.4 [25.4, 29.7]    | 62.7 [53.0, 74.3]    | 21.7 [18.4, 25.3]    |   35.8 |     177.6 |
| small        | offshelf_free | nf4_bnb     | nsc_test      | 56.5 [54.9, 58.3]    | 79.8 [75.3, 84.5]    | 62.5 [59.2, 66.1]    |   70.8 |     177.6 |
| small        | offshelf_free | nf4_bnb     | nsc_diff_test | 49.7 [47.3, 52.2]    | 72.4 [66.6, 79.6]    | 56.3 [51.5, 61.6]    |   64.1 |     177.6 |
| small        | offshelf_free | ct2_f32     | ascend_test   | 26.0 [24.4, 27.5]    | 56.5 [49.5, 63.3]    | 19.4 [17.0, 22.0]    |   35.8 |     462.2 |
| small        | offshelf_free | ct2_f32     | nsc_test      | 53.1 [51.6, 54.6]    | 71.5 [67.8, 75.6]    | 56.7 [53.2, 60.0]    |   71.2 |     462.2 |
| small        | offshelf_free | ct2_f32     | nsc_diff_test | 47.3 [45.1, 49.4]    | 71.1 [65.3, 77.6]    | 52.8 [48.1, 57.5]    |   67.4 |     462.2 |
| small        | offshelf_free | ct2_int8    | ascend_test   | 26.6 [25.0, 28.2]    | 57.5 [50.1, 65.3]    | 20.4 [18.1, 23.2]    |   34.6 |     237.7 |
| small        | offshelf_free | ct2_int8    | nsc_test      | 53.8 [52.2, 55.3]    | 71.6 [67.9, 75.3]    | 57.2 [53.8, 60.5]    |   72.4 |     237.7 |
| small        | offshelf_free | ct2_int8    | nsc_diff_test | 47.6 [45.3, 49.8]    | 67.1 [62.0, 73.5]    | 53.7 [48.8, 58.7]    |   68.9 |     237.7 |
| small        | offshelf_free | ggml_q8_0   | ascend_test   | 26.1 [24.6, 27.6]    | 55.6 [48.6, 62.3]    | 19.1 [17.1, 21.6]    |   35.8 |     252.2 |
| small        | offshelf_free | ggml_q8_0   | nsc_test      | 81.4 [80.2, 82.5]    | 80.7 [77.2, 84.3]    | 72.5 [69.4, 75.3]    |   84.9 |     252.2 |
| small        | offshelf_free | ggml_q8_0   | nsc_diff_test | 78.5 [76.7, 80.3]    | 75.2 [70.2, 80.9]    | 67.7 [63.7, 71.7]    |   85.8 |     252.2 |
| small        | offshelf_free | ggml_q5_0   | ascend_test   | 26.4 [25.0, 27.9]    | 57.0 [50.2, 63.9]    | 19.5 [17.3, 22.0]    |   35.8 |     167.1 |
| small        | offshelf_free | ggml_q5_0   | nsc_test      | 82.0 [80.7, 83.2]    | 79.0 [75.7, 82.5]    | 73.3 [70.2, 76.2]    |   85.2 |     167.1 |
| small        | offshelf_free | ggml_q5_0   | nsc_diff_test | 78.5 [76.6, 80.4]    | 73.9 [68.6, 79.8]    | 67.3 [63.2, 71.6]    |   84.6 |     167.1 |
| small        | offshelf_free | ggml_q4_0   | ascend_test   | 26.4 [24.9, 27.9]    | 61.0 [53.9, 68.8]    | 19.9 [17.6, 22.3]    |   33.3 |     138.7 |
| small        | offshelf_free | ggml_q4_0   | nsc_test      | 85.1 [83.8, 86.2]    | 81.3 [77.9, 84.9]    | 74.1 [71.0, 76.8]    |   85.9 |     138.7 |
| small        | offshelf_free | ggml_q4_0   | nsc_diff_test | 82.1 [80.4, 83.9]    | 76.5 [71.5, 82.0]    | 69.0 [65.0, 73.4]    |   84.6 |     138.7 |
| small        | offshelf_zh   | fp16        | ascend_test   | 35.5 [31.4, 40.7]    | 59.8 [49.8, 70.9]    | 24.3 [19.0, 32.5]    |   42   |     461.1 |
| small        | offshelf_zh   | fp16        | nsc_test      | 104.5 [102.1, 107.2] | 78.0 [72.8, 83.1]    | 88.8 [84.6, 92.4]    |   84   |     461.1 |
| small        | offshelf_zh   | fp16        | nsc_diff_test | 103.6 [99.6, 108.3]  | 71.5 [64.8, 79.2]    | 84.1 [77.2, 91.4]    |   83.5 |     461.1 |
| sub_small    | sub           | fp16        | ascend_test   | 14.3 [13.2, 15.3]    | 28.3 [23.0, 33.6]    | 10.2 [8.8, 11.7]     |   18.5 |     461.1 |
| sub_small    | sub           | fp16        | nsc_test      | 21.4 [20.6, 22.4]    | 22.6 [20.6, 25.0]    | 19.1 [17.8, 20.3]    |   14   |     461.1 |
| sub_small    | sub           | fp16        | nsc_diff_test | 18.1 [16.8, 19.6]    | 19.6 [17.2, 22.8]    | 16.1 [14.4, 18.3]    |   12.3 |     461.1 |
| sub_small    | sub           | int8_bnb    | ascend_test   | 14.5 [13.4, 15.5]    | 29.2 [23.8, 34.6]    | 10.4 [9.0, 11.8]     |   19.8 |     272.1 |
| sub_small    | sub           | int8_bnb    | nsc_test      | 21.4 [20.5, 22.3]    | 22.5 [20.6, 24.8]    | 19.0 [17.8, 20.3]    |   14   |     272.1 |
| sub_small    | sub           | int8_bnb    | nsc_diff_test | 17.8 [16.6, 19.2]    | 19.6 [17.2, 22.8]    | 15.3 [14.2, 16.5]    |   12.5 |     272.1 |
| sub_small    | sub           | nf4_bnb     | ascend_test   | 15.0 [14.0, 16.1]    | 28.1 [23.2, 33.1]    | 10.9 [9.6, 12.3]     |   16   |     177.6 |
| sub_small    | sub           | nf4_bnb     | nsc_test      | 21.8 [21.0, 22.6]    | 22.5 [21.0, 24.3]    | 20.3 [18.9, 22.0]    |   15   |     177.6 |
| sub_small    | sub           | nf4_bnb     | nsc_diff_test | 18.9 [17.5, 20.6]    | 21.3 [18.0, 25.6]    | 16.7 [15.0, 18.9]    |   14.3 |     177.6 |
| tiny         | offshelf_both | fp16        | ascend_test   | 65.2 [60.3, 70.8]    | 108.4 [88.5, 131.4]  | 60.7 [53.4, 69.3]    |   72.8 |      72   |
| tiny         | offshelf_both | fp16        | nsc_test      | 81.4 [79.5, 83.4]    | 106.4 [102.1, 111.2] | 87.4 [83.3, 91.5]    |   87.3 |      72   |
| tiny         | offshelf_both | fp16        | nsc_diff_test | 75.9 [73.0, 79.2]    | 106.3 [99.2, 115.1]  | 83.9 [78.3, 90.3]    |   78.7 |      72   |
| tiny         | offshelf_en   | fp16        | ascend_test   | 99.8 [94.4, 105.5]   | 84.0 [71.0, 101.7]   | 103.6 [98.1, 110.7]  |   88.9 |      72   |
| tiny         | offshelf_en   | fp16        | nsc_test      | 78.4 [76.8, 80.0]    | 106.0 [102.8, 109.5] | 82.6 [79.7, 85.8]    |   87.4 |      72   |
| tiny         | offshelf_en   | fp16        | nsc_diff_test | 72.9 [70.4, 75.6]    | 103.3 [99.1, 108.3]  | 84.1 [79.2, 89.5]    |   80.9 |      72   |
| tiny         | offshelf_free | fp16        | ascend_test   | 52.7 [47.3, 59.6]    | 125.6 [108.2, 145.5] | 50.0 [40.1, 64.6]    |   63   |      72   |
| tiny         | offshelf_free | fp16        | nsc_test      | 93.6 [90.5, 96.7]    | 133.0 [124.1, 142.7] | 101.6 [96.1, 107.4]  |   84.5 |      72   |
| tiny         | offshelf_free | fp16        | nsc_diff_test | 81.0 [76.7, 85.4]    | 118.2 [107.8, 129.2] | 94.5 [85.7, 103.4]   |   76.9 |      72   |
| tiny         | offshelf_free | int8_bnb    | ascend_test   | 54.1 [47.5, 61.9]    | 117.0 [103.2, 132.4] | 57.0 [41.8, 76.3]    |   60.5 |      56.3 |
| tiny         | offshelf_free | int8_bnb    | nsc_test      | 93.8 [90.7, 97.0]    | 129.7 [120.7, 139.0] | 101.8 [95.9, 108.2]  |   85.2 |      56.3 |
| tiny         | offshelf_free | int8_bnb    | nsc_diff_test | 84.2 [79.6, 88.6]    | 122.7 [111.9, 134.6] | 102.5 [93.2, 111.8]  |   76.9 |      56.3 |
| tiny         | offshelf_free | nf4_bnb     | ascend_test   | 74.5 [66.6, 82.8]    | 149.7 [120.0, 184.8] | 76.3 [61.7, 92.8]    |   71.6 |      48.4 |
| tiny         | offshelf_free | nf4_bnb     | nsc_test      | 102.0 [98.6, 105.3]  | 137.1 [128.4, 146.0] | 108.9 [102.8, 115.1] |   86.4 |      48.4 |
| tiny         | offshelf_free | nf4_bnb     | nsc_diff_test | 90.4 [85.6, 95.2]    | 139.1 [126.5, 151.6] | 106.7 [97.3, 117.4]  |   79.1 |      48.4 |
| tiny         | offshelf_free | ct2_f32     | ascend_test   | 49.0 [45.7, 52.8]    | 111.3 [100.9, 122.6] | 43.7 [38.0, 51.1]    |   63   |      73.1 |
| tiny         | offshelf_free | ct2_f32     | nsc_test      | 79.0 [76.7, 81.2]    | 104.9 [99.2, 110.7]  | 79.8 [75.7, 83.9]    |   84.5 |      73.1 |
| tiny         | offshelf_free | ct2_f32     | nsc_diff_test | 68.6 [65.7, 71.6]    | 96.0 [89.2, 104.0]   | 74.3 [68.5, 80.0]    |   79.8 |      73.1 |
| tiny         | offshelf_free | ct2_int8    | ascend_test   | 47.6 [45.1, 50.0]    | 106.8 [96.7, 117.8]  | 42.2 [38.3, 46.5]    |   64.2 |      39.7 |
| tiny         | offshelf_free | ct2_int8    | nsc_test      | 75.9 [74.2, 77.8]    | 94.8 [90.9, 98.6]    | 76.0 [72.6, 79.4]    |   85.8 |      39.7 |
| tiny         | offshelf_free | ct2_int8    | nsc_diff_test | 66.5 [63.9, 69.1]    | 92.3 [86.4, 98.7]    | 71.4 [66.2, 76.9]    |   81   |      39.7 |
| tiny         | offshelf_free | ggml_f16    | ascend_test   | 47.1 [45.2, 49.0]    | 106.9 [99.3, 114.2]  | 39.7 [36.3, 43.0]    |   63   |      74.1 |
| tiny         | offshelf_free | ggml_f16    | nsc_test      | 85.2 [84.2, 86.2]    | 92.9 [90.3, 95.7]    | 82.6 [80.1, 84.8]    |   92   |      74.1 |
| tiny         | offshelf_free | ggml_f16    | nsc_diff_test | 80.4 [78.8, 81.9]    | 91.6 [87.4, 95.6]    | 74.6 [71.2, 78.0]    |   91.8 |      74.1 |
| tiny         | offshelf_free | ggml_q8_0   | ascend_test   | 47.1 [45.1, 48.9]    | 108.4 [99.8, 116.6]  | 39.1 [35.9, 42.2]    |   63   |      41.5 |
| tiny         | offshelf_free | ggml_q8_0   | nsc_test      | 85.3 [84.2, 86.3]    | 92.8 [90.3, 95.3]    | 81.1 [78.6, 83.2]    |   91.6 |      41.5 |
| tiny         | offshelf_free | ggml_q8_0   | nsc_diff_test | 80.5 [78.7, 82.2]    | 91.3 [87.5, 95.1]    | 74.8 [71.3, 78.4]    |   90.9 |      41.5 |
| tiny         | offshelf_free | ggml_q5_0   | ascend_test   | 57.7 [52.7, 64.1]    | 116.8 [107.6, 128.8] | 49.1 [44.2, 54.9]    |   64.2 |      28.5 |
| tiny         | offshelf_free | ggml_q5_0   | nsc_test      | 93.4 [91.1, 96.1]    | 107.3 [99.7, 117.8]  | 91.4 [88.3, 94.5]    |   93.2 |      28.5 |
| tiny         | offshelf_free | ggml_q5_0   | nsc_diff_test | 86.3 [83.8, 89.2]    | 95.9 [91.2, 101.3]   | 83.8 [79.8, 88.3]    |   92.7 |      28.5 |
| tiny         | offshelf_free | ggml_q4_0   | ascend_test   | 295.0 [249.4, 342.2] | 568.0 [417.6, 717.6] | 321.1 [254.8, 388.9] |   76.5 |      24.1 |
| tiny         | offshelf_free | ggml_q4_0   | nsc_test      | 298.6 [276.5, 322.2] | 374.0 [322.3, 428.3] | 336.5 [296.8, 378.9] |   95.6 |      24.1 |
| tiny         | offshelf_free | ggml_q4_0   | nsc_diff_test | 250.1 [221.7, 278.1] | 361.4 [308.2, 419.1] | 307.9 [260.1, 358.1] |   93.7 |      24.1 |
| tiny         | offshelf_zh   | fp16        | ascend_test   | 82.0 [70.1, 95.7]    | 127.9 [114.8, 144.6] | 58.2 [42.2, 77.8]    |   66.7 |      72   |
| tiny         | offshelf_zh   | fp16        | nsc_test      | 184.6 [178.9, 190.4] | 157.5 [145.5, 168.8] | 156.0 [148.0, 163.6] |   93.9 |      72   |
| tiny         | offshelf_zh   | fp16        | nsc_diff_test | 197.1 [188.0, 205.2] | 158.7 [142.2, 176.4] | 153.7 [141.3, 165.8] |   92.9 |      72   |

## Fragility ratio R = (Eq_emb - E16_emb) / (Eq_mat - E16_mat)

d_emb / d_mat: change in embedded / matrix error (points) vs the same model at 16-bit, 95% CIs. R is reported only when the CI of d_mat excludes 0.

| model        | method        | precision   | ref16    | set           | main   | d_emb                | d_mat                | R                                         |
|:-------------|:--------------|:------------|:---------|:--------------|:-------|:---------------------|:---------------------|:------------------------------------------|
| base         | offshelf_free | int8_bnb    | fp16     | ascend_test   | zh     | 5.8 [-12.2, 25.3]    | 2.4 [0.0, 6.6]       | 2.43 [-24.52, 19.51]                      |
| base         | offshelf_free | int8_bnb    | fp16     | nsc_test      | en     | 0.4 [-7.6, 8.5]      | 4.3 [-1.4, 9.6]      | not reported (matrix change within noise) |
| base         | offshelf_free | int8_bnb    | fp16     | nsc_test      | zh     | 0.5 [-2.4, 3.8]      | -3.1 [-8.3, 1.1]     | not reported (matrix change within noise) |
| base         | offshelf_free | int8_bnb    | fp16     | nsc_diff_test | en     | 0.4 [-6.8, 6.9]      | 1.9 [-2.7, 6.3]      | not reported (matrix change within noise) |
| base         | offshelf_free | nf4_bnb     | fp16     | ascend_test   | zh     | 2.2 [-12.6, 14.4]    | 6.7 [2.9, 11.2]      | 0.33 [-2.62, 2.31]                        |
| base         | offshelf_free | nf4_bnb     | fp16     | nsc_test      | en     | 5.6 [-4.7, 15.7]     | 8.2 [1.0, 14.9]      | 0.68 [-1.21, 3.31]                        |
| base         | offshelf_free | nf4_bnb     | fp16     | nsc_test      | zh     | 9.4 [-1.0, 21.5]     | 5.3 [-2.2, 12.5]     | not reported (matrix change within noise) |
| base         | offshelf_free | nf4_bnb     | fp16     | nsc_diff_test | en     | 11.7 [-2.5, 26.8]    | 22.5 [12.6, 33.7]    | 0.52 [-0.12, 1.21]                        |
| base         | offshelf_free | ct2_int8    | ct2_f32  | ascend_test   | zh     | 3.8 [-3.1, 11.7]     | 0.3 [-1.8, 2.7]      | not reported (matrix change within noise) |
| base         | offshelf_free | ct2_int8    | ct2_f32  | nsc_test      | en     | -0.6 [-5.6, 4.2]     | 0.4 [-2.6, 3.6]      | not reported (matrix change within noise) |
| base         | offshelf_free | ct2_int8    | ct2_f32  | nsc_test      | zh     | -1.3 [-5.2, 2.3]     | -3.1 [-8.6, 1.3]     | not reported (matrix change within noise) |
| base         | offshelf_free | ct2_int8    | ct2_f32  | nsc_diff_test | en     | -1.0 [-8.8, 4.6]     | 0.5 [-2.0, 3.2]      | not reported (matrix change within noise) |
| base         | offshelf_free | ggml_q8_0   | ggml_f16 | ascend_test   | zh     | 1.1 [-0.9, 3.2]      | -0.3 [-0.8, 0.2]     | not reported (matrix change within noise) |
| base         | offshelf_free | ggml_q8_0   | ggml_f16 | nsc_test      | en     | 0.5 [-0.9, 1.8]      | -0.3 [-3.2, 2.2]     | not reported (matrix change within noise) |
| base         | offshelf_free | ggml_q8_0   | ggml_f16 | nsc_test      | zh     | -4.4 [-10.7, -0.5]   | -0.6 [-2.4, 0.8]     | not reported (matrix change within noise) |
| base         | offshelf_free | ggml_q8_0   | ggml_f16 | nsc_diff_test | en     | -0.8 [-4.9, 3.9]     | 4.0 [0.3, 8.7]       | -0.20 [-3.00, 2.27]                       |
| base         | offshelf_free | ggml_q5_0   | ggml_f16 | ascend_test   | zh     | 4.4 [-0.7, 10.2]     | 0.3 [-0.9, 1.6]      | not reported (matrix change within noise) |
| base         | offshelf_free | ggml_q5_0   | ggml_f16 | nsc_test      | en     | -1.6 [-3.9, 0.5]     | -2.6 [-6.0, 0.8]     | not reported (matrix change within noise) |
| base         | offshelf_free | ggml_q5_0   | ggml_f16 | nsc_test      | zh     | -4.5 [-10.7, 0.4]    | -0.8 [-3.1, 1.6]     | not reported (matrix change within noise) |
| base         | offshelf_free | ggml_q5_0   | ggml_f16 | nsc_diff_test | en     | -0.6 [-4.0, 3.2]     | -2.8 [-7.3, 2.5]     | not reported (matrix change within noise) |
| base         | offshelf_free | ggml_q4_0   | ggml_f16 | ascend_test   | zh     | 13.0 [6.1, 20.4]     | 3.1 [1.5, 4.9]       | 4.25 [1.81, 10.17]                        |
| base         | offshelf_free | ggml_q4_0   | ggml_f16 | nsc_test      | en     | 4.8 [1.6, 8.9]       | -0.8 [-4.4, 2.7]     | not reported (matrix change within noise) |
| base         | offshelf_free | ggml_q4_0   | ggml_f16 | nsc_test      | zh     | 8.6 [-7.9, 35.3]     | 6.8 [3.0, 10.8]      | 1.26 [-1.60, 5.11]                        |
| base         | offshelf_free | ggml_q4_0   | ggml_f16 | nsc_diff_test | en     | 5.7 [0.7, 11.0]      | 1.3 [-6.0, 9.2]      | not reported (matrix change within noise) |
| ft_base      | ft            | int8_bnb    | fp16     | ascend_test   | zh     | -18.8 [-59.0, 0.8]   | 0.3 [-0.2, 0.9]      | not reported (matrix change within noise) |
| ft_base      | ft            | int8_bnb    | fp16     | nsc_test      | en     | -1.0 [-4.0, 1.4]     | -1.6 [-4.0, 0.6]     | not reported (matrix change within noise) |
| ft_base      | ft            | int8_bnb    | fp16     | nsc_test      | zh     | -2.4 [-7.7, 0.9]     | 4.0 [0.8, 7.7]       | -0.61 [-3.18, 0.33]                       |
| ft_base      | ft            | int8_bnb    | fp16     | nsc_diff_test | en     | -0.2 [-0.7, 0.3]     | 0.1 [-0.2, 0.4]      | not reported (matrix change within noise) |
| ft_base      | ft            | nf4_bnb     | fp16     | ascend_test   | zh     | -13.4 [-52.6, 8.2]   | 1.3 [0.4, 2.3]       | -10.08 [-61.78, 8.94]                     |
| ft_base      | ft            | nf4_bnb     | fp16     | nsc_test      | en     | 2.5 [0.2, 4.8]       | -1.3 [-4.4, 1.5]     | not reported (matrix change within noise) |
| ft_base      | ft            | nf4_bnb     | fp16     | nsc_test      | zh     | -3.3 [-9.4, 1.2]     | 2.4 [-2.9, 7.3]      | not reported (matrix change within noise) |
| ft_base      | ft            | nf4_bnb     | fp16     | nsc_diff_test | en     | 3.5 [0.6, 8.0]       | 0.3 [-2.7, 2.5]      | not reported (matrix change within noise) |
| ft_small     | ft            | int8_bnb    | fp16     | ascend_test   | zh     | -0.4 [-1.2, 0.3]     | -0.3 [-0.6, 0.0]     | not reported (matrix change within noise) |
| ft_small     | ft            | int8_bnb    | fp16     | nsc_test      | en     | 1.2 [0.0, 2.5]       | 0.3 [-0.2, 1.1]      | not reported (matrix change within noise) |
| ft_small     | ft            | int8_bnb    | fp16     | nsc_test      | zh     | 0.2 [-0.5, 0.9]      | -0.2 [-0.5, 0.2]     | not reported (matrix change within noise) |
| ft_small     | ft            | int8_bnb    | fp16     | nsc_diff_test | en     | 1.7 [-0.2, 5.1]      | 0.7 [-0.1, 2.1]      | not reported (matrix change within noise) |
| ft_small     | ft            | nf4_bnb     | fp16     | ascend_test   | zh     | 1.2 [-1.8, 4.1]      | 0.4 [-0.2, 1.0]      | not reported (matrix change within noise) |
| ft_small     | ft            | nf4_bnb     | fp16     | nsc_test      | en     | 1.3 [-2.5, 5.0]      | 3.2 [0.7, 6.0]       | 0.40 [-1.78, 1.46]                        |
| ft_small     | ft            | nf4_bnb     | fp16     | nsc_test      | zh     | 0.8 [-0.5, 2.1]      | 1.6 [0.7, 2.6]       | 0.51 [-0.48, 1.41]                        |
| ft_small     | ft            | nf4_bnb     | fp16     | nsc_diff_test | en     | -1.9 [-8.1, 1.5]     | 2.9 [0.2, 6.6]       | -0.65 [-12.25, 1.55]                      |
| ft_tiny      | ft            | int8_bnb    | fp16     | ascend_test   | zh     | -0.7 [-3.8, 2.7]     | 1.9 [-0.5, 6.2]      | not reported (matrix change within noise) |
| ft_tiny      | ft            | int8_bnb    | fp16     | nsc_test      | en     | -0.1 [-2.1, 1.2]     | 0.2 [-1.0, 1.6]      | not reported (matrix change within noise) |
| ft_tiny      | ft            | int8_bnb    | fp16     | nsc_test      | zh     | -1.2 [-2.5, -0.0]    | -1.4 [-4.0, 0.5]     | not reported (matrix change within noise) |
| ft_tiny      | ft            | int8_bnb    | fp16     | nsc_diff_test | en     | 0.2 [-0.4, 0.8]      | -0.2 [-0.6, 0.3]     | not reported (matrix change within noise) |
| ft_tiny      | ft            | nf4_bnb     | fp16     | ascend_test   | zh     | 16.2 [8.9, 23.6]     | 4.7 [2.9, 6.6]       | 3.47 [1.85, 6.13]                         |
| ft_tiny      | ft            | nf4_bnb     | fp16     | nsc_test      | en     | 11.2 [5.5, 18.9]     | 9.2 [6.3, 12.4]      | 1.22 [0.63, 2.13]                         |
| ft_tiny      | ft            | nf4_bnb     | fp16     | nsc_test      | zh     | 17.3 [7.0, 31.9]     | 2.8 [-3.9, 9.4]      | not reported (matrix change within noise) |
| ft_tiny      | ft            | nf4_bnb     | fp16     | nsc_diff_test | en     | 6.4 [2.2, 12.1]      | 8.8 [5.1, 13.3]      | 0.73 [0.29, 1.51]                         |
| full_base    | full          | int8_bnb    | fp16     | ascend_test   | zh     | -2.8 [-5.2, -0.4]    | 0.4 [-0.1, 0.9]      | not reported (matrix change within noise) |
| full_base    | full          | int8_bnb    | fp16     | nsc_test      | en     | 0.3 [-0.1, 0.7]      | -0.1 [-0.3, 0.2]     | not reported (matrix change within noise) |
| full_base    | full          | int8_bnb    | fp16     | nsc_test      | zh     | 0.4 [-0.5, 1.3]      | 0.1 [-0.6, 0.6]      | not reported (matrix change within noise) |
| full_base    | full          | int8_bnb    | fp16     | nsc_diff_test | en     | 0.2 [-0.3, 0.8]      | 0.2 [-0.2, 0.5]      | not reported (matrix change within noise) |
| full_base    | full          | nf4_bnb     | fp16     | ascend_test   | zh     | 1.3 [-2.6, 5.5]      | 1.7 [0.8, 2.6]       | 0.80 [-1.99, 4.08]                        |
| full_base    | full          | nf4_bnb     | fp16     | nsc_test      | en     | 3.0 [2.0, 4.1]       | 1.7 [1.1, 2.5]       | 1.75 [1.05, 3.03]                         |
| full_base    | full          | nf4_bnb     | fp16     | nsc_test      | zh     | 2.6 [0.9, 4.1]       | 1.7 [0.8, 2.6]       | 1.47 [0.50, 3.53]                         |
| full_base    | full          | nf4_bnb     | fp16     | nsc_diff_test | en     | 2.2 [1.1, 3.4]       | 1.4 [0.7, 2.1]       | 1.64 [0.74, 3.29]                         |
| rw_base      | rw            | int8_bnb    | fp16     | ascend_test   | zh     | -2.6 [-5.3, -0.0]    | 0.1 [-0.3, 0.5]      | not reported (matrix change within noise) |
| rw_base      | rw            | int8_bnb    | fp16     | nsc_test      | en     | 1.0 [-0.8, 4.1]      | -1.3 [-2.9, -0.0]    | -0.79 [-11.81, 1.44]                      |
| rw_base      | rw            | int8_bnb    | fp16     | nsc_test      | zh     | -1.7 [-5.3, 0.6]     | 1.1 [-1.5, 3.8]      | not reported (matrix change within noise) |
| rw_base      | rw            | int8_bnb    | fp16     | nsc_diff_test | en     | 2.1 [-0.1, 6.2]      | 0.1 [-0.2, 0.5]      | not reported (matrix change within noise) |
| rw_base      | rw            | nf4_bnb     | fp16     | ascend_test   | zh     | -1.1 [-6.1, 3.9]     | 2.2 [0.9, 3.7]       | -0.53 [-3.76, 1.96]                       |
| rw_base      | rw            | nf4_bnb     | fp16     | nsc_test      | en     | 1.5 [0.5, 2.4]       | -0.3 [-2.7, 1.8]     | not reported (matrix change within noise) |
| rw_base      | rw            | nf4_bnb     | fp16     | nsc_test      | zh     | -0.3 [-4.1, 2.7]     | 3.0 [-1.9, 7.3]      | not reported (matrix change within noise) |
| rw_base      | rw            | nf4_bnb     | fp16     | nsc_diff_test | en     | 1.4 [0.2, 2.6]       | 1.1 [-1.4, 2.6]      | not reported (matrix change within noise) |
| semonly_base | semonly       | int8_bnb    | fp16     | ascend_test   | zh     | 1.3 [-1.6, 4.5]      | -0.1 [-0.6, 0.5]     | not reported (matrix change within noise) |
| semonly_base | semonly       | int8_bnb    | fp16     | nsc_test      | en     | -0.4 [-2.1, 0.8]     | -0.3 [-0.9, 0.2]     | not reported (matrix change within noise) |
| semonly_base | semonly       | int8_bnb    | fp16     | nsc_test      | zh     | -1.1 [-3.3, 0.3]     | 0.6 [-1.4, 2.8]      | not reported (matrix change within noise) |
| semonly_base | semonly       | int8_bnb    | fp16     | nsc_diff_test | en     | -0.4 [-1.0, 0.1]     | 0.1 [-0.2, 0.4]      | not reported (matrix change within noise) |
| semonly_base | semonly       | nf4_bnb     | fp16     | ascend_test   | zh     | 7.1 [1.4, 13.0]      | 2.1 [1.2, 3.1]       | 3.39 [0.63, 7.72]                         |
| semonly_base | semonly       | nf4_bnb     | fp16     | nsc_test      | en     | 3.3 [0.7, 5.9]       | 2.5 [1.0, 4.1]       | 1.32 [0.46, 2.36]                         |
| semonly_base | semonly       | nf4_bnb     | fp16     | nsc_test      | zh     | -0.3 [-4.0, 2.7]     | 1.0 [-2.1, 4.2]      | not reported (matrix change within noise) |
| semonly_base | semonly       | nf4_bnb     | fp16     | nsc_diff_test | en     | 1.1 [0.0, 2.3]       | 2.4 [0.9, 4.8]       | 0.47 [0.01, 1.31]                         |
| small        | offshelf_free | int8_bnb    | fp16     | ascend_test   | zh     | 0.5 [-4.5, 4.8]      | 1.4 [-2.3, 5.9]      | not reported (matrix change within noise) |
| small        | offshelf_free | int8_bnb    | fp16     | nsc_test      | en     | -0.1 [-3.3, 2.9]     | -1.7 [-4.6, 0.8]     | not reported (matrix change within noise) |
| small        | offshelf_free | int8_bnb    | fp16     | nsc_test      | zh     | -2.5 [-8.8, 2.0]     | -0.5 [-2.1, 0.9]     | not reported (matrix change within noise) |
| small        | offshelf_free | int8_bnb    | fp16     | nsc_diff_test | en     | 2.2 [-4.2, 10.5]     | 2.1 [-0.6, 5.3]      | not reported (matrix change within noise) |
| small        | offshelf_free | nf4_bnb     | fp16     | ascend_test   | zh     | 2.1 [-2.0, 6.2]      | 1.5 [0.3, 2.7]       | 1.41 [-2.82, 7.24]                        |
| small        | offshelf_free | nf4_bnb     | fp16     | nsc_test      | en     | 4.5 [-2.3, 10.3]     | 1.5 [-3.3, 6.3]      | not reported (matrix change within noise) |
| small        | offshelf_free | nf4_bnb     | fp16     | nsc_test      | zh     | -0.2 [-8.8, 8.4]     | 0.8 [-1.8, 3.2]      | not reported (matrix change within noise) |
| small        | offshelf_free | nf4_bnb     | fp16     | nsc_diff_test | en     | -1.6 [-8.5, 4.6]     | 2.3 [-3.7, 8.3]      | not reported (matrix change within noise) |
| small        | offshelf_free | ct2_int8    | ct2_f32  | ascend_test   | zh     | 1.6 [-2.3, 5.4]      | 0.9 [0.0, 2.0]       | 1.73 [-3.92, 12.03]                       |
| small        | offshelf_free | ct2_int8    | ct2_f32  | nsc_test      | en     | -0.1 [-3.8, 2.9]     | 0.5 [-1.8, 3.0]      | not reported (matrix change within noise) |
| small        | offshelf_free | ct2_int8    | ct2_f32  | nsc_test      | zh     | 0.7 [-1.3, 3.0]      | 0.5 [-1.2, 2.2]      | not reported (matrix change within noise) |
| small        | offshelf_free | ct2_int8    | ct2_f32  | nsc_diff_test | en     | -5.3 [-10.0, -1.4]   | 2.2 [-0.7, 5.6]      | not reported (matrix change within noise) |
| sub_small    | sub           | int8_bnb    | fp16     | ascend_test   | zh     | 0.7 [-0.8, 2.3]      | 0.1 [-0.2, 0.4]      | not reported (matrix change within noise) |
| sub_small    | sub           | int8_bnb    | fp16     | nsc_test      | en     | 0.1 [-0.4, 0.7]      | 0.1 [-1.0, 1.2]      | not reported (matrix change within noise) |
| sub_small    | sub           | int8_bnb    | fp16     | nsc_test      | zh     | -0.5 [-2.8, 1.4]     | -0.3 [-1.5, 0.7]     | not reported (matrix change within noise) |
| sub_small    | sub           | int8_bnb    | fp16     | nsc_diff_test | en     | -0.1 [-0.6, 0.4]     | -1.2 [-3.6, 0.0]     | not reported (matrix change within noise) |
| sub_small    | sub           | nf4_bnb     | fp16     | ascend_test   | zh     | -1.2 [-4.8, 2.1]     | 0.6 [-0.0, 1.1]      | not reported (matrix change within noise) |
| sub_small    | sub           | nf4_bnb     | fp16     | nsc_test      | en     | -0.6 [-3.1, 1.0]     | 1.4 [-0.3, 3.2]      | not reported (matrix change within noise) |
| sub_small    | sub           | nf4_bnb     | fp16     | nsc_test      | zh     | 1.2 [-1.7, 4.6]      | 0.9 [0.1, 1.9]       | 1.34 [-2.12, 14.03]                       |
| sub_small    | sub           | nf4_bnb     | fp16     | nsc_diff_test | en     | 0.3 [-0.6, 1.2]      | -0.5 [-3.2, 1.0]     | not reported (matrix change within noise) |
| tiny         | offshelf_free | int8_bnb    | fp16     | ascend_test   | zh     | -1.9 [-6.5, 3.1]     | 3.3 [-3.5, 13.4]     | not reported (matrix change within noise) |
| tiny         | offshelf_free | int8_bnb    | fp16     | nsc_test      | en     | -3.4 [-13.7, 6.8]    | 2.5 [-3.2, 8.4]      | not reported (matrix change within noise) |
| tiny         | offshelf_free | int8_bnb    | fp16     | nsc_test      | zh     | -3.1 [-16.0, 11.7]   | -3.9 [-10.7, 3.0]    | not reported (matrix change within noise) |
| tiny         | offshelf_free | int8_bnb    | fp16     | nsc_diff_test | en     | 6.5 [-4.6, 18.1]     | 4.8 [-4.0, 13.0]     | not reported (matrix change within noise) |
| tiny         | offshelf_free | nf4_bnb     | fp16     | ascend_test   | zh     | 45.0 [8.8, 89.2]     | 30.7 [8.8, 52.4]     | 1.47 [0.28, 4.24]                         |
| tiny         | offshelf_free | nf4_bnb     | fp16     | nsc_test      | en     | 8.1 [-5.7, 21.3]     | 1.1 [-8.3, 11.4]     | not reported (matrix change within noise) |
| tiny         | offshelf_free | nf4_bnb     | fp16     | nsc_test      | zh     | -6.1 [-30.4, 15.5]   | 17.5 [6.1, 28.9]     | -0.35 [-3.62, 0.84]                       |
| tiny         | offshelf_free | nf4_bnb     | fp16     | nsc_diff_test | en     | 22.6 [2.0, 43.7]     | -5.3 [-20.3, 10.2]   | not reported (matrix change within noise) |
| tiny         | offshelf_free | ct2_int8    | ct2_f32  | ascend_test   | zh     | -4.4 [-9.4, 1.2]     | -0.8 [-7.6, 3.6]     | not reported (matrix change within noise) |
| tiny         | offshelf_free | ct2_int8    | ct2_f32  | nsc_test      | en     | -8.4 [-15.3, -2.4]   | -4.6 [-9.1, -0.3]    | 1.84 [0.39, 9.29]                         |
| tiny         | offshelf_free | ct2_int8    | ct2_f32  | nsc_test      | zh     | -14.7 [-23.7, -7.1]  | -2.4 [-8.4, 3.0]     | not reported (matrix change within noise) |
| tiny         | offshelf_free | ct2_int8    | ct2_f32  | nsc_diff_test | en     | -6.6 [-15.2, 0.7]    | -7.6 [-12.9, -3.2]   | 0.86 [-0.13, 2.13]                        |
| tiny         | offshelf_free | ggml_q8_0   | ggml_f16 | ascend_test   | zh     | 2.3 [-2.0, 6.5]      | -0.5 [-1.7, 0.7]     | not reported (matrix change within noise) |
| tiny         | offshelf_free | ggml_q8_0   | ggml_f16 | nsc_test      | en     | -0.2 [-1.6, 1.1]     | -2.3 [-4.3, -0.5]    | 0.09 [-0.87, 1.50]                        |
| tiny         | offshelf_free | ggml_q8_0   | ggml_f16 | nsc_test      | zh     | 0.1 [-1.7, 2.1]      | -0.1 [-1.3, 1.0]     | not reported (matrix change within noise) |
| tiny         | offshelf_free | ggml_q8_0   | ggml_f16 | nsc_diff_test | en     | 0.2 [-4.8, 4.5]      | 1.0 [-2.0, 3.7]      | not reported (matrix change within noise) |
| tiny         | offshelf_free | ggml_q5_0   | ggml_f16 | ascend_test   | zh     | 12.3 [2.5, 25.2]     | 9.8 [6.0, 14.5]      | 1.26 [0.27, 2.39]                         |
| tiny         | offshelf_free | ggml_q5_0   | ggml_f16 | nsc_test      | en     | 17.5 [6.6, 31.3]     | 7.3 [3.8, 11.6]      | 2.39 [1.07, 4.65]                         |
| tiny         | offshelf_free | ggml_q5_0   | ggml_f16 | nsc_test      | zh     | 5.9 [1.3, 11.0]      | 11.3 [7.9, 14.9]     | 0.52 [0.12, 0.96]                         |
| tiny         | offshelf_free | ggml_q5_0   | ggml_f16 | nsc_diff_test | en     | 5.4 [0.1, 11.6]      | 7.8 [3.8, 12.6]      | 0.69 [0.01, 1.47]                         |
| tiny         | offshelf_free | ggml_q4_0   | ggml_f16 | ascend_test   | zh     | 508.7 [344.2, 700.1] | 294.1 [222.0, 372.1] | 1.73 [1.26, 2.34]                         |
| tiny         | offshelf_free | ggml_q4_0   | ggml_f16 | nsc_test      | en     | 240.3 [179.1, 311.3] | 175.4 [135.4, 219.5] | 1.37 [1.08, 1.73]                         |
| tiny         | offshelf_free | ggml_q4_0   | ggml_f16 | nsc_test      | zh     | 391.8 [285.9, 521.2] | 389.2 [303.4, 480.5] | 1.01 [0.80, 1.26]                         |
| tiny         | offshelf_free | ggml_q4_0   | ggml_f16 | nsc_diff_test | en     | 232.4 [168.3, 301.3] | 182.1 [133.6, 241.4] | 1.28 [1.04, 1.55]                         |

## Recovery Rec = (Eq_emb - Eq,fix_emb) / (Eq_emb - E16_emb)

loss: embedded-error increase from quantization (points). Rec is reported with a CI only when the loss CI excludes 0.

| fix                             | model    | precision   | set           | main   | loss             | Rec                       |
|:--------------------------------|:---------|:------------|:--------------|:-------|:-----------------|:--------------------------|
| mixed precision top-2 at 16-bit | ft_small | nf4_bnb     | ascend_test   | zh     | 1.2 [-1.8, 4.1]  | -0.93 (loss within noise) |
| mixed precision top-2 at 16-bit | ft_small | nf4_bnb     | nsc_test      | en     | 1.3 [-2.5, 5.0]  | 0.88 (loss within noise)  |
| mixed precision top-2 at 16-bit | ft_small | nf4_bnb     | nsc_test      | zh     | 0.8 [-0.5, 2.1]  | -0.01 (loss within noise) |
| mixed precision top-2 at 16-bit | ft_small | nf4_bnb     | nsc_diff_test | en     | -1.9 [-8.1, 1.5] | 1.11 (loss within noise)  |

## Sensitivity scan (A6)

| file                              | top5_by_KL                                                                                                                              |   all_quantized_KL |   chosen_k |
|:----------------------------------|:----------------------------------------------------------------------------------------------------------------------------------------|-------------------:|-----------:|
| sensitivity_ft_small_nf4_bnb.json | encoder.layers.0 (0.0037), decoder.layers.11 (0.0024), encoder.layers.7 (0.0023), encoder.layers.11 (0.0016), encoder.layers.9 (0.0015) |          0.0296311 |          2 |
