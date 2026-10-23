import os, pandas as pd
D = os.path.expanduser("~/cs425-codeswitch-asr")
s = pd.read_parquet(f"{D}/data/leak/signals.parquet")
amb = pd.read_parquet(f"{D}/data/leak/ambiguous_test.parquet")
P = {20,55,61,82,83,87,88,89,90,91,92,94,95,96,98,99,100,117,127,128,138,149}
F = {2,3,13,28,42,50,58,59,60,68,69,81,84,93,97,110,129,135,136,144}
amb["judge"] = ["partner_words_audible" if i in P else ("faint" if i in F else "clean") for i in range(len(amb))]
print("ambiguous: rule vs my judgement agreement", round((amb.judge == amb.rule_label).mean(), 3))
print(pd.crosstab(amb.judge, amb.rule_label))
j = dict(zip(amb.utt_id, amb.judge))
s["leak_label"] = [j.get(u, r) if st != "nsc_val" else r for u, r, st in zip(s.utt_id, s.rule_label, s.set)]
s["leak_label_source"] = ["annotator1" if u in j else "rule" for u in s.utt_id]
s.to_parquet(f"{D}/data/leak/signals.parquet")
amb[["utt_id", "set", "rule_label", "judge"]].to_csv(f"{D}/data/llm_labels/d3_test_judgements.csv", index=False)
for st in ["nsc_test", "nsc_diff_test"]:
    t = s[s.set == st]
    vc = t.leak_label.value_counts()
    print(st, len(t), {k: f"{v} ({100*v/len(t):.2f}%)" for k, v in vc.items()})
