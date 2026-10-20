# Rubric D1: what replaced an embedded word? (translation / transliteration / other)

Each item is an ASR error block from code-switched Mandarin–English speech:
- `ref_side`: the reference (correct) words. These were *embedded*, i.e. in the sentence's
  minority language (usually English inside a Chinese sentence).
- `out_side`: what the recogniser wrote instead, containing text in the *other* script.

Label each item with exactly one of:

| label | when |
|---|---|
| `translation` | `out_side` expresses the **same meaning** as `ref_side` in the other language, fully or for its main content word. Minor function words added or missing are fine (`technical → 的技术`, `friends → 朋友`, `graduate students → 绮丽的学生`: the head noun is translated). **Equivalent fillers and interjections** in the other language also count (`呃 → uh`, `哦 → oh`, `um → 嗯`). |
| `transliteration` | `out_side` is a **phonetic rendering**: it sounds like `ref_side` but doesn't mean it (`deadline → 得来`, `python → 派森`, `hong kong → 红矿`, `alipay → 阿里佩`). This includes partial phonetic matches covering most of the word, and English words that sound like a Chinese reference (`那个 → nigga`, `是 → sure`). |
| `other` | Neither: an unrelated word, a hallucination, a different word with a different meaning that doesn't sound alike, or a block too mixed to say (`one year → 玩意儿` is transliteration if it sounds alike; `tell → 聊` is translation only if the meaning matches, otherwise other). |

Rules:
1. Judge `out_side` against `ref_side` only. Ignore whether the rest of the sentence was right.
2. If `out_side` mixes a translation with extra unrelated words, choose by the main content word.
3. If both readings apply (sounds alike **and** means the same, e.g. a loanword such as `sofa → 沙发`, `coffee → 咖啡`), label `translation`. The recogniser produced the standard Chinese word.
4. Function words that map one-to-one (`the → 的`, `a → 一`) are `transliteration` if chosen for sound (`the → 的` sounds alike) and `translation` only if clearly chosen for meaning. When unsure, `other`.
5. Give a one-line reason (≤ 12 words).

Output: one CSV line per item: `pair_id,label,reason`.
