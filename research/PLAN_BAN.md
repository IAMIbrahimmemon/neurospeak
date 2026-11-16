# Study 28: is removing self-doubt more than banning doubt words?

Written on 2026-11-09 after study 27's review, before any generation or logit-lens measurement for this study.

**Why.** Study 27 showed that removing repetition stops verbatim copying but leaves the model second-guessing in new words, while removing self-doubt ends that stuck state. The simplest account of self-doubt removal is that d mostly pushes up the output tokens "Wait", "Hmm" and "Actually", and removing it just stops those words. If that's true, banning those words at the output, with no internal edit, should do about as well. If the ban does much worse, the internal edit changes something beyond which words come out. Nobody in this project has tried a plain word ban.

**Arm.** Unedited model. At every step, the logits of 11 tokens are set to minus infinity: `Wait` 13784, ` Wait` 13428, `wait` 11158, ` wait` 3655, `Hmm` 77264, ` Hmm` 85152, ` hmm` 84485, `Actually` 50821, ` Actually` 32645, `actually` 70546, ` actually` 3404. These are the whole-word tokens for the main words of `DOUBT` in `investigate/behaviors.py`. Same generation as studies 20 to 27: GSM8K test rows 800 to 863, seed 20, 3,000-token cap, thinking on. Study 20's unedited (4) and self-doubt (32; replications 23 and 26) arms are reused.

**Part A, no generation.** Logit lens of d. Push d through the final norm (its weight only) and the unembedding, and list the 30 tokens d raises most. Record where the 11 banned tokens rank among all tokens. Also run block 19's real outputs on unedited traces with and without d removed, through the remaining blocks, and record the mean change in log-probability of the 11 tokens. Descriptive only, no bar.

| Bar | Needed | Why |
|---|---|---|
| B1. The ban works | `DOUBT` words per 100 under 0.5 | manipulation check |
| B2. Does the ban help? | ban minus unedited, signed CI | |
| B3. Self-doubt beats the ban | self-doubt minus ban, CI above 0 | the edit is more than a word ban |
| B4. Stuck state | runs with no box but "= <right answer>" in the text, ban beside unedited (26) and self-doubt (11) | does the ban end the stuck state? |
| B5. Intact | at most 2 more runs collapsed or with 200 spaces in a row than unedited | |

Secondary columns, no bars: finished, finished and right, substitute checking phrases per 1,000 tokens (`let me (check|verify|confirm|re-?examine|re-?calculate)`, `double-check`, `re-?read`, `is there a trick`, `hold on`, `but let me`), 8-gram repeats, runs in a loop.

**Predictions, written before data.** B1 passes. The ban gets 6 to 14, and substitute phrases rise sharply, so B4 stays near unedited. B3 passes. In Part A, at least one banned token is in d's top 30.

**How to read it.**

- B3 passes and B4 stays high: banning the words doesn't end the stuck state. The direction carries a state, not just word choice.
- The ban near self-doubt: the edit is mostly a word ban. That would be reported first, and the story of a doubt state would be dropped.
- Ban in between: some of the gain is word choice, and B4 shows how much of the stuck state is left.

**Limits known in advance.** The ban also blocks ordinary uses ("wait 5 minutes", "actually costs"), which may cost a few problems. It leaves variants like "Hmmm" and "WAIT". One seed, one model, 64 problems. Self-doubt's 32 is high against its replications (about 27 of 64), so B3 is also read against 27.
