# Study 17: given time, does the pair get more answers right, or only finish more?

Written on 2026-11-06 before any data from these arms and after seeing study 14. Study 14 tied on correct answers at 28 of 64, yet the pair finished 22 runs against 10 and left no loops against 10. That finish gap had no bar, and I picked it after looking. 42 of the pair's runs were still going at 3,000 tokens. So a run that doesn't loop might use more time well, and a tie at 3,000 may not hold at a longer cap. Or it may keep going without ever settling.

**Runs.** GSM8K test rows 640 to 703, 64 unused problems. Seed 17, 9,000-token cap, value-based scoring, L19, every token. Each run is also scored as if cut at 3,000 tokens by truncating its text, as in study 1.

**Arms.** Self-doubt removed. Both removed. No unedited arm: study 1 found the unedited model stayed at 2 of 32 from 3,000 to 10,000 tokens, with 0 finished.

| Bar | Needed | Why |
|---|---|---|
| E1. Finish gap replicates | finished by 3,000, both minus self-doubt, at least +10 points, CI above 0 | study 14's gap on new problems, with a bar |
| E2. More right given time | correct at 9,000, both minus self-doubt, at least +10 points, CI above 0 | does the extra finishing buy answers? |
| E3. The pair uses the time | both's correct at 9,000 minus its correct at 3,000, at least +10 points, CI above 0 | is the cap what limits it? |
| E4. Most runs end | both finishes at least 32 of 64 by 9,000 | is there an end at all? |
| E5. Intact | both has at most 2 more broken runs than self-doubt | |

Broken uses the same rule as study 14. Finishing counts a run that stopped before the cap.

**How to read it.**

- E1 to E5 pass: the pair is the better edit when time allows, and the tie at 3,000 was about the cap.
- E1 passes, E2 fails: stopping loops makes runs end, and doesn't make more answers right. Self-doubt's removal gets the same answers by writing them down sooner.
- E3 passes, E2 fails: both arms use the extra time about equally.
- E4 fails: runs without loops still don't settle, and the limiter isn't the cap. That would point at whatever keeps them re-checking.
- E1 fails: study 14's finish gap was post-hoc noise, and the rest is reported with that first.

Also with no bars: right-when-boxed, tokens to first box, and correct among finished runs for each arm.

**Limits known in advance.** One seed, one model, 64 problems, no correction across bars. Truncated scoring assumes sampling up to 3,000 tokens is what a 3,000 cap would have given, which holds per run in text but not across batches of differing lengths. No unedited arm here, so gains over unedited lean on earlier studies.

## Amendment, 2026-11-06, after a Fable review of study 14 and before any data

The review showed the finish gap I'd credited to the pair follows repetition's removal: repetition alone finished 19 of 64 and both 22. So comparing both with self-doubt can't say which part does the work. A `rep` arm joins, repetition removed, at the same seed and cap, and it runs first. E1 to E5 stay. New bars:

| Bar | Needed | Why |
|---|---|---|
| E6. Repetition alone finishes | finished by 3,000, `rep` minus self-doubt, at least +10 points, CI above 0 | is finishing repetition's effect? |
| E7. Pair versus repetition, given time | correct at 9,000, both minus `rep`, at least +10 points, CI above 0 | does self-doubt's removal still add correct answers once time is plentiful? |

**Added readings.** E6 passes and both ties `rep` on finishing: finishing belongs to repetition's removal, as study 14 suggests. E7 passes: the two edits do different jobs and both are worth having. E7 fails with E2 passing: repetition's removal alone gets what the pair gets, given time, and self-doubt's removal only matters under a tight cap.
