# Replicating the self-doubt result on unused problems

Written on 2026-11-05 and committed before generating. On 64 GSM8K problems at seed 3, removing the self-doubt direction gave 23/64 correct against 4/64 unedited and 4/64 for a matched random direction ([results](RESULTS.md)). That is one seed, and those problems had already been used for the repetition direction's test.

**Setup.** GSM8K test rows 144–207 after the same fixed shuffle. They share nothing with rows 0–143 or the practice problems, which I checked by exact text. Seed 4, the same 3,000-token cap, thinking on. All directions are unchanged, saved before this run, and removed at L19 at every token. The problem count, cap and bars don't depend on any new output.

**Arms.** Unedited, self-doubt direction, repetition direction, and the same matched random direction as before.

**Scorer.** The frozen rules from `score_self_doubt.py`, moved unchanged into `gsm_score.py`. Before generating, I ran the new script on the seed 3 arms, and it must reproduce 4, 13, 23 and 4 correct. Any later scoring problem gets reported, and these verdicts stay as scored.

| Bar | Needed |
|---|---|
| R1. Correct | self-doubt minus unedited at least +5 points, CI above 0 |
| R2. Repeats | self-doubt at most 2/3 of unedited |
| R3. Control | self-doubt beats matched random on correct (CI above 0) and on repeats (CI below 0) |
| R4. Intact | collapsed runs at most 2 more than unedited |
| R5. On target | doubt words per 100 at most 2/3 of unedited |
| R6. Stricter count | finished and correct: self-doubt minus unedited, CI above 0 |
| R7. Against repetition | correct: self-doubt minus repetition direction, CI above 0 |

**How to read it.**

- The result replicates only if R1 to R5 all pass. If any fails, I report that seed 3 didn't fully replicate, with the numbers.
- R6 asks if the gain survives counting only runs that stopped by themselves. At seed 3 this was 9 against 2.
- R7 is a separate claim. Self-doubt can replicate against unedited and still not be better than the repetition direction.
- No result here shows why it works, or that the direction means self-doubt.

**Limits known in advance.** Still one model, one kind of problem and one cap. Bars are paired bootstrap intervals by problem, with no correction for testing several. I wrote the pairs, scores and bars myself. This doesn't test whether longer runs close the gap.
