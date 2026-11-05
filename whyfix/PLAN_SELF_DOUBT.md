# Does the self-doubt direction hold up on new problems?

Written on 2026-11-05 before generating. The lead: on the 16 practice problems at seed 2, removing the self-doubt direction gave 21.3% repeats and 8/16 correct (unedited 34.2%, 5/16). That is one seed, n=16, and it wasn't a bar. This test checks it on new problems with a control.

**Setup.** The same 64 GSM8K test problems, seed 3 and 3,000-token cap as the repetition direction's test, so the existing unedited and repetition arms are compared directly. None of these problems helped find the self-doubt direction or pick this lead. The direction is unchanged, from the 4 hand-written pairs, removed at L19 at every token.

**New arms.**

- `dir_self_doubt`: the direction itself.
- `dir_self_doubt_matched`: a random direction with the same overlap with the residual mean. That overlap is about 0 (−0.005), so this is close to a plain random direction.
- `dir_self_doubt_orth`: the direction with the repetition direction projected out. They share a cosine of 0.42, so this asks if anything remains without that overlap.

**Scorer changes, made before any new data.** Testing `score_self_doubt.py` on the two old arms found three misses in the old scorer: a backslash before `$`, a backslash before `%`, and empty `\\boxed{}` echoes of the instruction hiding a real answer written before them. All three are fixed and it rescores every arm. The old arms change from 3/64 and 10/64 to the counts it prints below, and those are the baselines here. The recorded verdicts for A1 and A2 aren't changed. The scorer is frozen from this point.

| Bar | Needed | Why |
|---|---|---|
| S1. Correct answers | self-doubt minus unedited at least +5 points, CI above 0 | same as A1 for the repetition direction |
| S2. Repeats | at most 2/3 of unedited | same as A2 |
| S3. Control | self-doubt beats the matched control on correct (CI above 0) and on repeats (CI below 0) | rules out "any direction does this" |
| S4. Intact | collapsed runs at most 2 more than unedited | a collapse can lower repeats without helping |
| S5. On target | doubt words per 100 words at most 2/3 of unedited | the direction should reduce the behavior it was built for |

The lead holds only if all five pass. If S1 fails, the 8/16 was noise or too small to see here, and I'll say so.

**Does it add anything?** O1: `dir_self_doubt_orth` passes S1 and S2. If it passes, the effect doesn't depend on the overlap with repetition. If it fails, nothing separate from the repetition direction has been shown. Self-doubt against the repetition direction is reported with an interval, with no bar.

**Limits known in advance.** One seed. Unedited correct is only about 3/64, so counts are small. Three new arms against one bar each, with no correction for multiple tests. Bars S1 to S3 are paired bootstrap intervals by problem. I wrote the pairs, scores and bars myself.

**Baselines from the frozen scorer.** Unedited 4/64 correct, 39.9% repeats, 1.44 doubt words per 100, 0 collapsed. Repetition direction 13/64 correct, 9.2% repeats. So S2 needs at most 26.6% and S5 at most 0.96.
