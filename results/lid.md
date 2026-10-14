# Language-ID head (validation, after smoothing)

Frame labels come from MMS forced alignment (PENDING_HUMAN: 30-utterance hand check).

| head        | set        |   speech_lang_acc |   frame_acc |   frame_acc_unsmoothed |   boundary_median_ms |   within_100ms |   switch_detected |
|:------------|:-----------|------------------:|------------:|-----------------------:|---------------------:|---------------:|------------------:|
| lid_ft_base | ascend_val |              96   |        85.9 |                   87.2 |                   60 |           43.8 |              75.4 |
| lid_ft_base | nsc_val    |              95.4 |        82.9 |                   83   |                  120 |           41.1 |              84.6 |
