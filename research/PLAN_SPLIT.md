# Study 23: how much removal does it take, and which part of it does the work?

Written on 2026-11-09 after study 22's review, before any generation for this study. Study 22 showed that removing self-doubt at L19 beats same-size random removals. Two questions remain open from its review.

1. **Strength.** On late text, three of study 22's four controls removed only 61 to 68% of self-doubt's dose. Nobody has measured how much of self-doubt's gain survives at that dose. If it mostly vanishes, the controls' late shortfall could still be part of why they failed. It's also the agenda's strength sweep.
2. **Constant or varying.** Self-doubt's projection at L19 is mostly one sign (signed mean over absolute 0.38 on early text, 0.67 on late text, from `runs/research/dosectl_dose_check.json`). So the removal does two things. It subtracts a near-constant offset along d, and it flattens the token-to-token variation along d. If the gain comes from the offset alone, the edit is a steering shift, not "removing doubt as it arises". That's a different mechanism.

**Arms.** All at block 19's output, every token, with d the original self-doubt direction and μ the mean of block 19's output over the project's default corpus (`core._default_corpus()[:24]`, first 2,000 characters each, position 0 dropped). μ is built by `research/build_split_mean.py` before any generation, and μ·d is logged then.

| Arm | Edit | Question |
|---|---|---|
| gain 0.65 | h − 0.65 (h·d) d | does about two thirds of the dose keep the gain? |
| gain 0.35 | h − 0.35 (h·d) d | where does it fall off? |
| centered | h − ((h − μ)·d) d | only the varying part removed |
| offset | h − (μ·d) d | only the constant part removed |

Generation is the same as studies 20 to 22: GSM8K test rows 800 to 863, seed 20, 3,000-token cap, no penalty, thinking on, batches of 8. Comparisons reuse study 20's unedited (4) and full-removal (32) arms and study 22's four controls (pooled).

| Bar | Needed | Why |
|---|---|---|
| S1. Two-thirds beats the controls | gain 0.65 minus study 22's pooled controls, CI above 0 | the controls' late shortfall doesn't explain their failure |
| S2. Strength curve | gain 0.35 and gain 0.65, each minus unedited and minus full, signed | is the effect graded or a threshold? |
| S3. Varying part carries it | centered minus unedited, CI above 0 | |
| S4. Constant part alone | offset minus unedited, signed | does a constant shift along d do the job? |
| S5. Which part | centered minus offset, signed, CI excludes 0 for a winner | |
| S6. Intact, per arm | at most 2 more runs collapsed or with 200 spaces in a row than unedited | |

Secondary columns, no bars: correct with a non-empty last box, finished and right, doubt words per 100, "wait" per 1,000 tokens.

**Predictions, written before data.** S1 passes, with gain 0.65 near 20. Gain 0.35 is near 10. S3 passes, with centered close to full. Offset is near unedited. I'm least sure of the offset arm, since the projection is mostly one sign late in a run.

**How to read it.**

- Centered near full and offset near unedited: the gain is removing the varying doubt signal, the reading every study so far has assumed.
- Offset near full and centered near unedited: the gain is a constant push along d, a steering effect. The "removing doubt" story would need rewriting, and that would be reported first.
- Both partial: both parts matter. The sum shows whether they add up.
- S1 fails: the edit needs nearly its full dose. Study 22's controls were short of dose late in the run, so the "it's specific" claim then rests on control 104 alone.

**Limits known in advance.** μ is one mean over general text, while the real offset drifts through a run (the projection's signed share rises from 0.38 to 0.67). So "offset" removes a fixed amount that is too small late in a run. Two strengths aren't a full curve. One seed, one model, one layer, 64 problems.
