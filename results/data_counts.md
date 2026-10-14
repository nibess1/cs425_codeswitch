# Data counts (size check before training, proposal §3.1)

`main` = sentence main language from jieba word counts; `embedded_units` = E units.
Detectability: at 20% error on 400 embedded units the 95% CI is about ±4 points.

| set           | main     |   utts |   hours |   mixed_utts |   units |   embedded_units |   Ms_units |   critical_words |   speakers |
|:--------------|:---------|-------:|--------:|-------------:|--------:|-----------------:|-----------:|-----------------:|-----------:|
| ascend_train  | all      |   9869 |    8.77 |         2715 |  120843 |             7081 |       5431 |              699 |         18 |
| ascend_train  | zh       |   6705 |    6.2  |         1882 |   96038 |             4523 |       4436 |              509 |         18 |
| ascend_train  | en       |   3009 |    2.47 |          678 |   23664 |             2558 |        995 |              175 |         18 |
| ascend_train  | balanced |    155 |    0.11 |          155 |    1141 |                0 |          0 |               15 |         17 |
| ascend_val    | all      |   1130 |    0.92 |          208 |   12331 |              471 |        364 |               80 |          3 |
| ascend_val    | zh       |    790 |    0.65 |          130 |    9833 |              259 |        274 |               40 |          3 |
| ascend_val    | en       |    330 |    0.27 |           68 |    2460 |              212 |         90 |               36 |          3 |
| ascend_val    | balanced |     10 |    0    |           10 |      38 |                0 |          0 |                4 |          2 |
| ascend_test   | all      |   1315 |    0.92 |          371 |   11520 |              772 |        747 |               76 |          2 |
| ascend_test   | zh       |    989 |    0.71 |          302 |    9757 |              623 |        690 |               60 |          2 |
| ascend_test   | en       |    298 |    0.19 |           41 |    1642 |              149 |         57 |               16 |          2 |
| ascend_test   | balanced |     28 |    0.01 |           28 |     121 |                0 |          0 |                0 |          2 |
| nsc_train     | all      |  35647 |   39.99 |        22252 |  537777 |           105493 |      56633 |             5809 |        412 |
| nsc_train     | zh       |  10216 |   10.32 |         8944 |  156367 |            31006 |      26093 |             1392 |        395 |
| nsc_train     | en       |  23576 |   28.43 |        11453 |  364945 |            74487 |      30540 |             4207 |        412 |
| nsc_train     | balanced |   1855 |    1.24 |         1855 |   16465 |                0 |          0 |              210 |        345 |
| nsc_val       | all      |   3367 |    3    |          855 |   39276 |             4099 |       2140 |              368 |         60 |
| nsc_val       | zh       |    669 |    0.56 |          388 |    8408 |             1308 |       1076 |               73 |         57 |
| nsc_val       | en       |   2631 |    2.4  |          400 |   30265 |             2791 |       1064 |              283 |         60 |
| nsc_val       | balanced |     67 |    0.05 |           67 |     603 |                0 |          0 |               12 |         38 |
| nsc_test      | all      |   8876 |    8    |         2535 |  101182 |            11042 |       6085 |             1167 |        120 |
| nsc_test      | zh       |   1536 |    1.28 |          964 |   19137 |             3107 |       2672 |              188 |        114 |
| nsc_test      | en       |   7133 |    6.59 |         1364 |   80456 |             7935 |       3413 |              960 |        120 |
| nsc_test      | balanced |    207 |    0.13 |          207 |    1589 |                0 |          0 |               19 |         81 |
| nsc_diff_test | all      |   4085 |    4    |         1189 |   49532 |             5648 |       2971 |              661 |        626 |
| nsc_diff_test | zh       |    708 |    0.65 |          469 |    9450 |             1605 |       1356 |              114 |        350 |
| nsc_diff_test | en       |   3283 |    3.28 |          626 |   39203 |             4043 |       1615 |              537 |        611 |
| nsc_diff_test | balanced |     94 |    0.07 |           94 |     879 |                0 |          0 |               10 |         87 |
