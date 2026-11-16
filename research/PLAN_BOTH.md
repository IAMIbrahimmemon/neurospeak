# Study 14: is removing both directions from the start better than either alone?

Written on 2026-11-06 before any data from these arms. On stuck runs in study 11, removing repetition and self-doubt together finished 5 and got 5 correct of 56, against 1 finished and 2 correct for repetition alone. Those are small counts, with no bar, on stuck runs only. Removing self-doubt from the start has been the strongest single edit for answers. So does removing both from token one beat it?

**Runs.** GSM8K test rows 448 to 511, 64 unused problems. Seed 11, 3,000-token cap, value-based scoring. L19, every token.

**Arms.** Unedited. Self-doubt removed. Repetition removed. Both removed, which is the two-dimensional span of the two directions. Two random directions removed, each with the original control's overlap with the mean and orthogonal to each other, so it matches the size of the edit in the mean's direction.

| Bar | Needed | Why |
|---|---|---|
| B1. Beats self-doubt | both minus self-doubt on correct at least +5 points, CI above 0 | is the pair better than the best single edit? |
| B2. Beats repetition | both minus repetition on correct at least +5 points, CI above 0 | |
| B3. Beats its control | both minus random two at least +10 points, CI above 0 | |
| B4. Intact | at most 2 more broken runs than unedited, counted as collapsed or under 20% distinct words in the first 200 | |

**How to read it.**

- B1 to B4 pass: the interaction seen on stuck runs carries to whole runs, and the pair is the better edit.
- B1 fails, B3 and B4 pass: adding repetition's removal doesn't improve on self-doubt's, and the study 11 finish count may be noise or specific to stuck runs.
- B3 fails: the pair does nothing beyond a random edit of the same shape.

Repeats, loops, doubt words in the first 500 words, answered, and right-when-answered are reported with no bars.

**Limits known in advance.** One seed, one model, 64 problems, 4 bars with no correction. A 2D removal is a larger edit than either alone and may cost more on normal text, which this study doesn't measure.
