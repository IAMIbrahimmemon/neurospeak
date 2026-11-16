# Study 22: is self-doubt special, or would removing any direction that size help?

Written on 2026-11-08 after study 21's review, before any direction is built or any generation run. Every "beats its control" result so far compares against one random direction, the same one in studies 20 and 21, made by `matched_control` with seed 1. It catches about a third as much of the residual as the self-doubt direction (Claude (Fable), `runs/research/xfer_dose.json`: at L19, mean |h·d|/|h| is 0.083 for self-doubt, 0.029 for the control). So the headline (removing self-doubt at L19 lifts GSM8K from about 4 to about 30 of 64) hasn't been tested against removing a different direction of the same size. If a same-size random direction does as well, the effect isn't about doubt.

**Controls.** `research/build_dosectl_dirs.py` builds 4 random directions at L19, seeds 101 to 104:

1. Collect block 19's outputs over the project's default corpus (`core._default_corpus()[:24]`), dropping position 0. Take their centered covariance C.
2. For seed s, draw z ~ N(0, I) and set d(α) = C^α z, then project out the self-doubt direction and the residual's mean, then normalize. α = 0 is a plain random direction. Larger α leans toward the directions the residual varies along most.
3. Pick α by bisection so the mean of |h·d|/|h| on the same corpus tokens equals self-doubt's on those tokens, within 5%.
4. Check before generation, on 6 unedited GSM8K traces (the text `measure_dose.py` uses, a different text from the build corpus): each control's mean |h·d|/|h| must be within 25% of self-doubt's there. If a control misses, it's rebuilt with the next seed, and that's logged. The check numbers are logged before generation.

These controls avoid the doubt direction and the mean by construction, so they test "a same-size direction that isn't self-doubt".

**Generation.** Same as studies 20 and 21: GSM8K test rows 800 to 863, seed 20, 3,000-token cap, no penalty, thinking on, batches of 8, removal at block 19's output at every token. Arms: `dir_dosectl_101` to `dir_dosectl_104`. Study 20's unedited (4 correct) and `dir_layer_19` (32 correct, cosine 1.0 with the original direction) arms are reused as the comparison.

| Bar | Needed | Why |
|---|---|---|
| D1. Beats each control | self-doubt minus control k, CI above 0, for all 4 controls | does it beat every same-size random removal? |
| D2. Beats the controls pooled | self-doubt minus the per-problem mean of the 4 controls, CI above 0 | the same, with less noise from one draw |
| D3. Controls don't help | pooled controls minus unedited, CI includes 0 or lies below | does any big removal at L19 help? (signed; a pass of D1 with D3 failing means part of the gain is generic) |
| D4. Doubt cut is specific | each control's doubt words per 100 above 2/3 of unedited's (the Y3 line), while self-doubt's is below it | does a same-size removal cut doubt words? |
| D5. Intact, reported per control | collapsed or 200 spaces in a row, against unedited | a control that breaks the model is reported as such, not as a fair comparison |

Secondary columns, no bars: correct with a non-empty last box, finished and right, "wait" per 1,000 tokens, mean tokens.

**Predictions, written before data.** D1, D2 and D4 pass. Controls land at 3 to 10 correct of 64. I'm least sure of D3, since a large removal may shorten thinking and gain a few points.

**How to read it.**

- D1 and D2 pass, D3 holds: the gain is specific to self-doubt, against same-size directions too.
- D2 passes but D3 fails: some of the gain is any large removal at L19. The part left is what self-doubt adds.
- D2 fails: a same-size random removal does about as well, and the headline isn't about doubt. That would undercut every study since the replication, and it would be reported first.

**Limits known in advance.** 4 controls is a small sample of random directions. Matching on mean |h·d|/|h| isn't the only way to match dose. The controls lean toward high-variance directions, so they may hit structure that self-doubt doesn't (for example massive-activation dimensions). The intact check covers that. One seed, one model, 64 problems, one layer.
