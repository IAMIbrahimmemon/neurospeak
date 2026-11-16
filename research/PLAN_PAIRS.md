# Study 2: do independently written examples find the same direction?

Written on 2026-11-05 before building any direction from these pairs. The self-doubt direction came from 4 pairs I wrote. If its effect depends on my wording, then it's a lucky vector and not a stable feature of the model.

**Pairs.** Two Claude writers, Fable and Haiku, each wrote 4 pairs from the same short brief, with no access to the project or any result. A third writer, Sonnet, produced nothing: a safety classifier stopped its write, and I didn't retry it. So this is two independent sets, not the three planned, and both writers are Claude, so they aren't as independent as different people would be.

**Directions.** Built by the same code and context as the original, at L19 with thinking on, with no other changes. `research/build_pair_directions.py` must first rebuild the original direction from the original pairs with cosine at least 0.999, or the build is wrong and nothing proceeds.

**Generation.** GSM8K test rows 240–303, 64 unused problems. Seed 6, 3,000-token cap. Arms: unedited, writer 2's direction, writer 3's direction, and the same matched random direction as before. Scored with the value-based rule from `whyfix/sensitivity_replicate_rules.py`.

| Bar | Needed | Why |
|---|---|---|
| P1. Same direction | each new direction has cosine at least 0.3 with the original, against about 0.03 for a random direction | is it the same vector? |
| P2. Same effect, correct | each new direction minus unedited at least +5 points, CI above 0 | does it do the same thing? |
| P3. Control | each new direction beats matched random on correct, CI above 0 | |
| P4. Intact | each new direction has at most 2 more collapsed runs than unedited | |
| P5. On target | each new direction has doubt words per 100 at most 2/3 of unedited | |

**How to read it.**

- P1 to P5 pass for both: the direction is stable across writers, and "luck from my pairs" is ruled out for these two sets.
- P2 passes and P1 fails: different vectors do the same thing, which would be interesting, and means one vector isn't the whole story.
- P1 passes and P2 fails: closeness isn't enough, and the exact vector matters.
- Both fail for a writer: the effect depends on how the examples are written. That is a real limit on the method.

**Limits known in advance.** Two sets, both from Claude, from a brief I wrote. The brief asks for self-doubt, so it can't show the behavior is found without being named. One seed, one model. P2 has no correction for two tests.
