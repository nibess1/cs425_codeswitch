import os, sys, csv, random
import pandas as pd
D = os.path.expanduser("~/cs425-codeswitch-asr"); sys.path.insert(0, D); os.chdir(D)
from csasr.score import report as R
from csasr.score.metrics import corpus_metrics
from csasr.data import manifest
s = pd.read_parquet("data/leak/signals.parquet")
flag = set(s[s.leak_label != "clean"].utt_id)
for sysn in ["small__fp16__offshelf_free", "ft_small__fp16__ft", "full_small__fp16__full", "ft_base__fp16__ft", "full_base__fp16__full"]:
    st = R.stats(sysn, use_labse=False)
    row = []
    for setn in ["nsc_test", "nsc_diff_test"]:
        ids, x = st[setn]
        a = corpus_metrics(x); b = corpus_metrics([v for u, v in zip(ids, x) if u not in flag])
        row.append(f"{setn}: MER {100*a['MER']:.2f}->{100*b['MER']:.2f} emb {100*a['E_E']:.2f}->{100*b['E_E']:.2f}")
    print(sysn, " | ".join(row))
t = s[s.set != "nsc_val"]
aud = pd.concat([t[t.leak_label != "clean"].sample(10, random_state=425), t[t.leak_label == "clean"].sample(10, random_state=425)]).sample(frac=1, random_state=425)
man = {r["utt_id"]: r for st in ("nsc_test", "nsc_diff_test") for r in manifest.load(st)}
with open("review/leak_audit.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["utt_id", "audio", "reference", "fine_tuned_small_output", "partner_transcript_within_2s", "automatic_label",
                "signal_speaker_verif", "signal_partner_words", "your_label (clean/faint/partner_words_audible)"])
    for r in aud.itertuples():
        w.writerow([r.utt_id, man[r.utt_id]["audio"], r.ref, r.hyp, r.partner_text, r.leak_label, round(r.s3_spk_diff, 3), r.s5_out_match, ""])
print("review/leak_audit.csv: 20 rows")
