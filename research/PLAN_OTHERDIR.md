# Study 25: does removing other learned directions do the same?

Written on 2026-11-09 at about 04:00, after studies 23 and 24 and before any generation or dose measurement for this study.

**Why.** Study 22 showed that removing self-doubt at L19 beats removing same-size random directions. Its review flagged the gap: a non-random direction, learned from examples of some other behavior, might also shorten thinking. If it does, the gain comes from disturbing a learned axis at L19, not from self-doubt. The project already has directions built the same way as self-doubt, from contrasting example pairs (`runs/whyfix/directions.pt`). Three of them have little overlap with self-doubt.

| Arm | Behavior | Cosine with self-doubt |
|---|---|---:|
| `dir_bullets` | writing bullet lists | 0.006 |
| `dir_chinese` | switching to Chinese | −0.001 |
| `dir_restating` | restating the problem | 0.116 |

`dir_hedging` (0.41) and the writer directions (0.63, 0.67) overlap too much with self-doubt to serve as controls, so they're left out.

**Arms.** Each direction removed at block 19's output, every token, h − (h·d)d. Same generation as studies 20 to 24: GSM8K test rows 800 to 863, seed 20, 3,000-token cap, thinking on, batches of 8. Study 20's unedited (4) and full self-doubt (32) arms are reused.

**Dose.** After the plan is logged and before reading any generation, `research/otherdir_dose.py` measures each direction's mean |h·d|/|h| at L19 on unedited GSM traces, beside self-doubt's. It reports dose but doesn't change arms.

| Bar | Needed | Why |
|---|---|---|
| N1. Self-doubt beats each | full self-doubt (32) minus each arm, CI above 0 | the gain is specific against learned directions too |
| N2. Others against unedited | each arm minus unedited (4), signed CI | does any other learned direction help? |
| N3. Doubt cut is specific | self-doubt's doubt words (0.23 per 100) at or below two thirds of unedited (0.95), and each other arm above that line | |
| N4. Intact | at most 2 more runs collapsed or with 200 spaces in a row than unedited, per arm | |

Secondary columns, no bars: correct with a non-empty last box, finished and right, "wait" per 1,000 tokens, mean tokens.

**Predictions, written before data.** N1 passes for all three. All three land at 2 to 8 correct. Restating may come out a little higher, since it's a thinking-style direction.

**How to read it.**

- All three near unedited: the gain is specific to self-doubt against learned directions as well as random ones.
- One helps clearly (N2 CI above 0) but less than self-doubt: removing that behavior also shortens thinking. Check whether it cuts doubt words too.
- One matches self-doubt: the "self-doubt" label is wrong or too narrow, and that would be reported first.
- A direction whose dose on GSM text is under half of self-doubt's makes a weak control. Its null result would be reported as weak.

**Limits known in advance.** These directions were built on different example sets, so their sizes on math text differ from self-doubt's and aren't matched. Three directions, one seed, one model, one layer, 64 problems.
