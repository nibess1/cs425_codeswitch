# Data counts (size check before training, proposal §3.1)

`main` = sentence main language from jieba word counts; `embedded_units` = E units.
Detectability: at 20% error on 400 embedded units the 95% CI is about ±4 points.

| set           | main     |   utts |   hours |   mixed_utts |   units |   embedded_units |   Ms_units |   critical_words |   speakers |
|:--------------|:---------|-------:|--------:|-------------:|--------:|-----------------:|-----------:|-----------------:|-----------:|
| ascend_train  | all      |   9869 |    8.77 |         2715 |  120809 |             7080 |       5431 |              667 |         18 |
| ascend_train  | zh       |   6705 |    6.2  |         1882 |   96005 |             4521 |       4436 |              480 |         18 |
| ascend_train  | en       |   3009 |    2.47 |          678 |   23663 |             2559 |        995 |              173 |         18 |
| ascend_train  | balanced |    155 |    0.11 |          155 |    1141 |                0 |          0 |               14 |         17 |
| ascend_val    | all      |   1130 |    0.92 |          208 |   12330 |              471 |        364 |               81 |          3 |
| ascend_val    | zh       |    790 |    0.65 |          130 |    9832 |              259 |        274 |               41 |          3 |
| ascend_val    | en       |    330 |    0.27 |           68 |    2460 |              212 |         90 |               36 |          3 |
| ascend_val    | balanced |     10 |    0    |           10 |      38 |                0 |          0 |                4 |          2 |
| ascend_test   | all      |   1315 |    0.92 |          371 |   11519 |              772 |        747 |               73 |          2 |
| ascend_test   | zh       |    989 |    0.71 |          302 |    9756 |              623 |        690 |               57 |          2 |
| ascend_test   | en       |    298 |    0.19 |           41 |    1642 |              149 |         57 |               16 |          2 |
| ascend_test   | balanced |     28 |    0.01 |           28 |     121 |                0 |          0 |                0 |          2 |
| nsc_train     | all      |  35653 |   40    |        22256 |  537795 |           105472 |      56635 |             5854 |        412 |
| nsc_train     | zh       |  10215 |   10.32 |         8943 |  156294 |            30993 |      26088 |             1423 |        395 |
| nsc_train     | en       |  23582 |   28.44 |        11457 |  365022 |            74479 |      30547 |             4223 |        412 |
| nsc_train     | balanced |   1856 |    1.24 |         1856 |   16479 |                0 |          0 |              208 |        345 |
| nsc_val       | all      |   3367 |    3    |          855 |   39273 |             4093 |       2140 |              371 |         60 |
| nsc_val       | zh       |    669 |    0.56 |          388 |    8408 |             1308 |       1076 |               73 |         57 |
| nsc_val       | en       |   2631 |    2.4  |          400 |   30262 |             2785 |       1064 |              286 |         60 |
| nsc_val       | balanced |     67 |    0.05 |           67 |     603 |                0 |          0 |               12 |         38 |
| nsc_test      | all      |   8876 |    8    |         2534 |  101178 |            11042 |       6084 |             1164 |        120 |
| nsc_test      | zh       |   1535 |    1.28 |          963 |   19132 |             3106 |       2671 |              184 |        114 |
| nsc_test      | en       |   7134 |    6.59 |         1364 |   80458 |             7936 |       3413 |              960 |        120 |
| nsc_test      | balanced |    207 |    0.13 |          207 |    1588 |                0 |          0 |               20 |         81 |
| nsc_diff_test | all      |   4086 |    4    |         1189 |   49530 |             5647 |       2971 |              663 |        626 |
| nsc_diff_test | zh       |    708 |    0.65 |          469 |    9445 |             1605 |       1356 |              116 |        350 |
| nsc_diff_test | en       |   3284 |    3.28 |          626 |   39206 |             4042 |       1615 |              537 |        611 |
| nsc_diff_test | balanced |     94 |    0.07 |           94 |     879 |                0 |          0 |               10 |         87 |
