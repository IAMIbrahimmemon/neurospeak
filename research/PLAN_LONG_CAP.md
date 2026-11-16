# Study 1: does the self-doubt gain survive a longer cap?

Written on 2026-11-05 before generating. At a 3,000-token cap, removing the self-doubt direction gave 49/128 correct against 11/128 unedited. Most unedited runs never box an answer, so more tokens might close the gap. This test asks whether the gain is only speed.

**Setup.** GSM8K test rows 208–239 after the fixed shuffle, 32 problems no earlier test used. Seed 5, cap 10,000 tokens, thinking on. Arms: unedited, self-doubt direction, the same matched random direction. Each run is also scored as if cut at 3,000 tokens, by re-encoding its text, so each problem is compared with itself at both caps.

**Scorer.** Fixed before generating: the last non-empty box, numbers compared by value (`whyfix/sensitivity_replicate_rules.py`). Later scoring problems get reported, and verdicts stay as scored.

| Bar | Needed | Why |
|---|---|---|
| L1. Same effect at 3,000 | self-doubt minus unedited, CI above 0 | checks this sample behaves like the last two |
| L2. Gain persists | at 10,000, self-doubt minus unedited at least +15 points, CI above 0 | the main question |
| L3. Control | at 10,000, self-doubt minus matched random, CI above 0 | |
| L4. Unedited catches up | unedited at 10,000 minus unedited at 3,000, CI above 0 | does more time help the unedited model at all? |
| L5. Long enough to tell | at least half of unedited runs finish by 10,000 | if not, this cap can't settle the question |

**How to read it.**

- L2 and L5 pass: the gain is more than speed, up to this cap.
- L2 fails and L4 passes: mostly speed.
- L5 fails: unsettled. I'll report how far the gap closed and not call it either way.
- Better math needs a separate look. Among problems where both arms box an answer, I'll report how often each is right, with no bar, because that count will be small.

**Limits known in advance.** 32 problems, one seed, one model. CIs are paired bootstrap by problem and will be wide. Truncating by re-encoded tokens is close to the true cut, not exact.
