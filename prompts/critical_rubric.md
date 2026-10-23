# Rubric D2: is this a critical word?

**Use case.** Captions for clinics, counselling and deaf users. A *critical word* is one whose
mis-recognition would change a fact the listener acts on. If it's wrong, the caption gives
wrong information, not just awkward wording.

Each item is a word (one English word, one digit string, or one Chinese word) with an example
sentence it occurs in. Label it `critical` or `not_critical`.

## Critical
| Category | Examples |
|---|---|
| **Numbers and quantities** | `10`, `3`, `二十`, `thirty`, `half`, ordinals used as facts (`third year` → `third`) |
| **Dates and times** | weekdays, months, `tomorrow`, `deadline`, `号`, `点` (o'clock), clock times, semesters (`sem`) |
| **Money and amounts** | `dollars`, `块`, `元`, `cents`, prices, `deposit`, `rent`, `salary` |
| **Names of people** | given names, surnames, nicknames: `tony`, `michael`, `姚明` |
| **Places and organisations** | cities, countries, schools, companies, apps, brands, stations: `hong kong`, `singapore`, `alipay`, `nus`, `mrt` |
| **Drugs and medical terms** | `panadol`, `insulin`, `vaccine`, `clinic`, symptoms (`fever`), doses |
| **Technical or academic terms** | course and task words a student acts on (`assignment`, `quiz`, `midterm`, `module`, `gpa`), and domain terms whose replacement loses the content (`python`, `neural network`, `blockchain`, `ocr`) |

## Not critical
- Common everyday words, even if rare in the data: `smartphone`, `pizza`, `friends`, `basketball` (a hobby mention, not a fact to act on), `happy`.
- Fillers, discourse particles and function words: `lah`, `um`, `the`, `那个`.
- Generic nouns, verbs and adjectives: `school`, `food`, `interesting`, `apply`.
- Single letters, unless they are part of a code or name (`a` → not; `nus`, `ocr` → critical).

## Rules
1. Judge the word *as used in the example sentence*. Bare `one` in "one of them" is not critical. A count ("one year", "one dollar") is.
2. When unsure between a generic noun and a domain term, ask: would a wrong caption make the reader act wrongly? If yes, `critical`.
3. Give a one-line reason (≤ 10 words).

Output: one CSV line per item: `item_id,label,reason`.
