import os, sys, pandas as pd, numpy as np
sys.path.insert(0, os.path.expanduser("~/cs425-codeswitch-asr"))
from csasr.score.align import C, align
from csasr.text.normalize import normalize
from csasr.text.units import split_units
s = pd.read_parquet(os.path.expanduser("~/cs425-codeswitch-asr/data/leak/signals.parquet"))
def recall(r):
    ru = split_units(normalize(r.ref)); hu = split_units(normalize(r.hyp))
    return sum(op == C for op, _, _ in align(ru, hu)) / max(1, len(ru))
s["ref_recall"] = [recall(r) for r in s.itertuples()]
def rule(r):
    if (r.s5_out_match >= 2 and r.s3_spk_diff > -0.15) or (r.s5_out_match >= 1 and r.s3_spk_diff > 0.1) \
            or (r.s3_spk_diff > 0.2 and r.ref_recall < 0.3):
        return "partner_words_audible"
    if r.s3_spk_diff > 0.2:
        return "faint"
    return "clean"
s["rule_label"] = [rule(r) for r in s.itertuples()]
s.to_parquet(os.path.expanduser("~/cs425-codeswitch-asr/data/leak/signals.parquet"))
# pilot judgements (annotator 1, PROCESS_LOG): index into the 103 val cases sorted by utt_id
v = s[(s.set == "nsc_val") & ((s.s5_out_match >= 1) | (s.s3_spk_diff > 0.2))].sort_values("utt_id").reset_index(drop=True)
P = {1,12,13,14,31,32,33,34,43,52,53,54,55,57,58,59,63,65,66,67,68,69,70,71,72,74,87,89,94,98}
F = {0,11,20,21,22,23,24,25,28,29,30,35,37,38,39,42,47,51,56,60,61,62,64,76,80,82,86,88,92,97,100,101,102}
judge = ["partner_words_audible" if i in P else ("faint" if i in F else "clean") for i in range(len(v))]
v["judge"] = judge
agree = (v.judge == v.rule_label).mean()
print("pilot cases", len(v), "rule vs my judgement agreement", round(agree, 3))
print(pd.crosstab(v.judge, v.rule_label))
print(v[v.judge != v.rule_label][["utt_id", "s3_spk_diff", "s5_out_match", "ref_recall", "judge", "rule_label"]].to_string())
v[["utt_id", "judge"]].to_csv(os.path.expanduser("~/cs425-codeswitch-asr/data/llm_labels/d3_pilot_judgements.csv"), index=False)
for st in ["nsc_val", "nsc_test", "nsc_diff_test"]:
    print(st, s[s.set == st].rule_label.value_counts(normalize=True).round(4).to_dict())
