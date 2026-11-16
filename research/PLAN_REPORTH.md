# Study 27: does a learned direction that works, with no self-doubt in it, also help?

Written on 2026-11-09 at about 06:35, after study 25's review and before any generation or dose measurement for this study.

**Why.** Study 25's three controls turned out to do nothing measurable on math text. investigate/RESULTS.md already showed that removing bullets and restating didn't change those behaviors even on the text they were built from. The GSM traces contain no Chinese for that direction to remove. So study 25 shows only that inert directions don't help. A real control needs a learned direction that demonstrably changes GSM thinking and has no overlap with self-doubt.

Repetition is that direction. In study 14, removing the raw repetition direction (`dir_neurospeak`, cosine 0.42 with self-doubt) got 22 of 64 against 7 unedited, and cut repeats from 49% to 7%. But with 0.42 overlap, some of that may be its self-doubt part. `dir_both_rep` is repetition with self-doubt projected out (cosine 0.000, built by `research/build_both.py`). It has only ever been run together with self-doubt, never alone.

**Arms.** Removed at block 19's output, every token, h − (h·d)d. Same generation as studies 20 to 26: GSM8K test rows 800 to 863, seed 20, 3,000-token cap, thinking on. Study 20's unedited (4) and self-doubt (32; earlier replications 23 and 26) are reused.

| Arm | Direction |
|---|---|
| repetition, self-doubt part removed | `dir_both_rep` |
| raw repetition | `dir_neurospeak` |

**Dose.** After this plan is logged and before generation is read, `research/otherdir_dose.py` measures each direction's mean |h·d|/|h| at L19 on unedited traces, beside self-doubt's.

| Bar | Needed | Why |
|---|---|---|
| R1. The control works | repetition-only cuts repeats (`loopiness`, as in study 14) by at least a third against unedited | it must change its own behavior here, unlike study 25's controls |
| R2. Does it help? | repetition-only minus unedited, signed CI | |
| R3. Self-doubt beats it | self-doubt minus repetition-only, CI above 0 | specific against a working learned direction |
| R4. Raw repetition | raw repetition minus repetition-only, signed CI | how much of raw repetition's gain rides on its self-doubt part |
| R5. Intact | at most 2 more runs collapsed or with 200 spaces in a row than unedited, per arm | |

Secondary columns, no bars: correct with a non-empty last box, finished and right, doubt words per 100, "wait" per 1,000 tokens, runs that enter a loop.

**Predictions, written before data.** R1 passes. Repetition-only gets 8 to 16, so R2 is positive but smaller than self-doubt. R3 passes. Raw repetition gets about 18 to 24, so R4 is positive.

**How to read it.**

- R1 passes, repetition-only near unedited, R3 passes: a working learned direction with no self-doubt in it doesn't help. The gain needs self-doubt, and raw repetition's gain rides on its self-doubt part.
- R1 passes and repetition-only helps clearly (R2 CI above 0): breaking loops helps on its own. Self-doubt is one of several ways to get the model to commit, and "specific to self-doubt" would be dropped.
- Repetition-only near self-doubt: the claim that self-doubt is special is wrong. That would be reported first.
- R1 fails: this control is inert here too, and the arm can't be read.

**Limits known in advance.** Projecting self-doubt out of repetition changes the repetition direction too (cosine with raw repetition about 0.91). Doses aren't matched. One seed, one model, one layer, 64 problems. Self-doubt's 32 is high against its replications, so R3 is also read against about 27.
