import os, sys, pandas as pd, numpy as np
sys.path.insert(0, os.path.expanduser("~/cs425-codeswitch-asr"))
from csasr.text.normalize import normalize
from csasr.text.units import split_units
COMMON = {"lah","leh","lor","ah","oh","uh","um","eh","ya","yeah","ok","the","a","i","you","and","so","is","to","it","that",
          "like","then","but","what","this","one","no","not","we","they","he","she","my","your","in","of","for","do","be",
          "are","was","have","got","can","know","just","very","also","right","huh","wah","sia","hor","meh","mah",
          "嗯","啊","哦","呃","的","了","是","我","你","他","她","这","那","个","就","也","不","在","有","都","吗","呢","吧","啦"}
s = pd.read_parquet(os.path.expanduser("~/cs425-codeswitch-asr/data/leak/signals.parquet"))
pool = pd.read_parquet(os.path.expanduser("~/cs425-codeswitch-asr/data/nsc/pool.parquet"))
pool["key"] = pool.room + "_" + pool.session
P = pool.set_index("utt_id")
by_chan = {k: g for k, g in pool.groupby(["key", "speaker"])}
s5, s5f, ptxt = [], [], []
for r in s.itertuples():
    me = P.loc[r.utt_id]; g = by_chan.get((me.key, r.partner))
    near = g[(g.end >= me.start - 2) & (g.start <= me.end + 2)] if g is not None else []
    pu = set(u for t in (near.ref if len(near) else []) for u in split_units(t))
    ref_u = set(split_units(normalize(r.ref))); hyp_u = split_units(normalize(r.hyp))
    m = [u for u in hyp_u if u not in ref_u and u in pu and u not in COMMON]
    s5.append(len(m)); s5f.append(len(m) / max(1, len(hyp_u)))
    ptxt.append(" / ".join(near.ref_raw.astype(str))[:200] if len(near) else "")
s["s5_out_match"] = s5; s["s5_frac"] = s5f; s["partner_text"] = ptxt
s.to_parquet(os.path.expanduser("~/cs425-codeswitch-asr/data/leak/signals.parquet"))
v = s[s.set == "nsc_val"]
for t in [1, 2, 3, 4]: print("val s5 >=", t, (v.s5_out_match >= t).sum())
print(pd.crosstab(v.s5_out_match.clip(0, 4), pd.cut(v.s3_spk_diff, [-1, 0, 0.1, 0.2, 1])))
