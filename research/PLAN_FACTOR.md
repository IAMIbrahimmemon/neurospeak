# Study 30: is it doubt, or not concluding?

Written on 2026-11-09 after study 29's review, before the new directions were built or any generation was run.

**Why.** Studies 28 and 29 left open whether d is a doubt direction or a "keep thinking, don't conclude" direction. The original pairs (`investigate/behaviors.py`, `self_doubt`) mix the two. Each doubt side is calculation plus "Wait, let me double-check", and each plain side is calculation plus a conclusion ("that is the final answer", "The answer is 15"). So d contrasts doubting with concluding, and either part could carry the gain.

**Directions.** Two new sets of 4 pairs in `research/factor_pairs.json`, on the original 4 problems, with the same opening clause, built by the same code at block 19 (`brain.concept`, as in `research/build_pair_directions.py`, which first checks it rebuilds d at cosine 0.999 or more).

- `dir_factor_doubt`: doubt words against plain next steps ("Next, ... Then ..."). Neither side concludes.
- `dir_factor_conclude`: plain next steps against concluding ("So ... The answer is 15, and that is the final answer."). Neither side doubts. The sign matches d's: the not-concluding side is the "is" side.

**Part A, before generation.** Cosines of each new direction with d, with each other, and with repetition. A least-squares fit of d on the two, reported as weights and R². Dose of each (mean |h·d|/|h| at L19 on 8 unedited traces, `research/otherdir_dose.py`). Descriptive only.

**Part B.** Each new direction removed at block 19's output, h − (h·u)u, on rows 800 to 863, seed 20, 3,000-token cap, as in studies 20 to 28. Study 20's unedited (4) and d-removed (32; replications 23 and 26) arms are reused.

| Bar | Needed | Why |
|---|---|---|
| F1. Doubt part helps | remove doubt-only minus unedited, CI above 0 | |
| F2. Conclude part helps | remove conclude-only minus unedited, CI above 0 | |
| F3. Which carries more | doubt-only minus conclude-only, signed CI | |
| F4. Does either match d | d minus each, signed CI | is one part the whole effect? |
| F5. Intact | at most 2 more runs collapsed or with 200 spaces in a row than unedited, per arm | |

Secondary columns, no bars: finished, closes its thinking, doubt words per 100, "I will" per 1,000 tokens, answer written but run cut.

**Predictions, written before data.** Cosine with d is about 0.6 for doubt-only and 0.4 for conclude-only. Doubt-only gets 15 to 25 and conclude-only 8 to 18, so F1 passes, F2 is unsure, and F3 is positive.

**How to read it.**

- Doubt-only near d and conclude-only near unedited: the doubt part carries the gain, and "self-doubt" is the right name.
- Conclude-only near d and doubt-only near unedited: the gain is from letting the model conclude. The "self-doubt" name would be dropped, and that would be reported first.
- Both help partly: both parts contribute. The two arms' sum against d shows whether they add.
- Neither helps: the effect needs the mixed contrast, or depends on these exact pairs.

**Limits known in advance.** I wrote these pairs myself, knowing the hypothesis. The "Next, ... Then ..." side is a new style of text the original pairs didn't use, so each new direction may carry that style. Four pairs per set. One seed, one model, one layer, 64 problems. Doses won't be matched, and Part A reports them.
