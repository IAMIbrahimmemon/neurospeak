# Study 7: can removing a direction free a run that's already stuck?

Written on 2026-11-05 before any rescue data. Seen so far, as exploration in `research/loop_onset.py`: of 87 unedited runs that entered a loop, 0 left it, and 0 finished. Removing the self-doubt direction from the first token means fewer runs enter a loop (17 of 128), and 2 of those 17 left. That can't separate two explanations. The direction may help start loops, or it may also help hold them. Every earlier study removed it from the first token, so none of them can tell these apart.

**Runs.** All unedited traces from `gsm64_none` and `gsm64b_none` that entered a loop by the rule in `loop_onset.py`: a 200-word window with at least 90% of its 8-grams already seen earlier in the run. That gives 56 runs. No direction was built from these problems.

**Rescue.** Cut each run at the end of its first looped window. Continue for 1,500 new tokens at seed 7 with the same sampling. The prefix isn't edited: the direction is removed only from the prefix's last token onward. Arms: unedited, self-doubt, repetition, and the self-doubt matched random direction.

**Escaped.** A run escaped if its continuation isn't collapsed and either it finishes under the cap or a full 200-word window in the continuation has under 50% of its 8-grams seen earlier in the run, prefix included.

| Bar | Needed | Why |
|---|---|---|
| E1. Loops hold on their own | unedited escaped at most 10% | fresh sampling alone shouldn't free them |
| E2. Self-doubt frees | self-doubt minus unedited at least +20 points, CI above 0 | does the direction hold the loop? |
| E3. Control | self-doubt minus matched random, CI above 0 | |
| E4. Repetition frees | repetition minus unedited at least +20 points, CI above 0 | same question for the other direction |
| E5. Intact | self-doubt and repetition each at most 2 more collapsed continuations than unedited | an escape into nonsense isn't an escape |

CIs are paired bootstraps over runs. Finishing and correct answers are reported for each arm, without a bar.

**How to read it.**

- E2, E3 and E5 pass: the direction helps hold a loop that has already started, so it's active in keeping it going.
- E2 fails with E1 passing: removing doubt doesn't free a stuck run. Its benefit is in stopping loops from forming, and a loop keeps itself going by some other route.
- E1 fails: loops aren't as absorbing as the exploration suggested, which changes the story above, and E2 still measures the gap.

**Limits known in advance.** The loop rule and the 0-of-87 figure were found on these same traces, so E1 isn't a fresh test of the rule on new problems, only of new sampling. 1,500 tokens is about 800 words, so a slow escape would be missed. Runs are cut at the first looped window, so this can't speak to loops that have run much longer. One model.
