# Study 19: is the 61 from self-doubt removal or from the presence penalty?

Written on 2026-11-08 after study 18 was scored and before any data from this arm. In study 18, self-doubt removal with MIT's presence penalty of 1.5 got 61 of 96 right at 9,000 tokens. The penalty alone took the unedited model from 8 to 16. Study 18 had no self-doubt arm without the penalty, so it can't say how much of the 61 the direction earns alone. Study 17 suggests the penalty matters: on different problems without it, self-doubt removal got 44% right and finished 33%, against 64% and 75% here.

**Run.** One new arm, `doubt`: self-doubt removed at L19, no presence penalty. Same as study 18 in every other way: GSM8K test rows 704 to 799, seed 18, 9,000-token cap, thinking on, batches of 8, the same prompt. It is compared with study 18's `none`, `none_pp`, `doubt_pp` and `astra_pp` traces on the same problems. Scored with the same value-based rule, at 9,000 and as if cut at 3,000.

| Bar | Needed | Why |
|---|---|---|
| N1. The direction works without the penalty | correct at 9,000, `doubt` minus `none`, CI above 0 | does the direction alone help on these problems? |
| N2. The penalty adds on top of the direction | correct at 9,000, `doubt_pp` minus `doubt`, CI above 0 | is part of the 61 from the penalty? |
| N3. Interaction | per problem, (`doubt_pp` minus `doubt`) minus (`none_pp` minus `none`), at 9,000, CI excludes 0 | does the penalty help more with the direction removed than without? Signed; a CI across 0 is reported as no interaction shown |
| N4. Our setting against theirs | correct at 9,000, `doubt` minus `astra_pp`, CI excludes 0 | self-doubt removal with our usual sampling against their edit with theirs. Signed |
| N5. Intact | `doubt` has at most 2 more runs with a run of 200 or more spaces, or collapsed by the study 14 rule, than `none` | study 18's whitespace failure, checked directly this time |

CIs are paired bootstrap 95% over problems, 10,000 resamples, seed 0, as in `whyfix/gsm_score.ci`.

**How to read it.**

- N1 passes and N2 fails: the direction does the work, and the penalty adds nothing shown on top of it.
- N1 and N2 pass: both contribute. N3 says whether they compound.
- N1 fails: on these problems, the direction alone doesn't help at 9,000, and study 18's result depends on the penalty. That would be the headline, and the README sentence must change.
- N4 favoring `doubt`: our edit with our sampling still beats their edit with theirs at this cap. N4 favoring Astra or showing no winner: the study 18 lead needs the penalty.

**Power, rough.** At study 17's rates (44% against an unedited 8%), N1's difference would be about +35 and passes easily. N2 at a true +10 passes about 40% of the time at n=96, judging from study 18's amendment. N3 is a difference of differences and has lower power still, so "no interaction shown" is the likely outcome even if one exists.

**Limits known in advance.** One seed, one model, 96 problems, no correction across bars. This arm runs a day after the others on the same machine, so anything that drifted between runs isn't controlled. The other arms' traces are reused and not regenerated.
