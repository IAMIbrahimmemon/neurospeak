# Study 9: does adding the direction cause the behavior?

Written on 2026-11-06 before any data from this study. Removing the self-doubt direction lowers doubt words and raises answers. That shows the direction is needed for part of the behavior. It doesn't show the direction is enough to cause it. The older repetition test added the direction and saw no loops within 250 tokens, which was short. This tests both directions over a full 3,000 tokens.

**Runs.** GSM8K test rows 304 to 351, 48 unused problems. Seed 8, 3,000-token cap, value-based scoring. The edit is `h + a d` at L19 at every token, where `a` is a fixed amount.

**Dose.** Mean per-token KL of the next-token distribution over the last 128 tokens of teacher-forced text, with the edit on every token. Calibration uses the first 400 words of 8 `long_none` runs, in file order, before any loop has formed, because this edit runs from token one. The grid for `a` is 0.5, 1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, and the value closest on a log scale is used. Low target is 0.0176, repetition's removal dose in study 8. High target is 0.088, 5 times that. Amounts go in `runs/research/add_gains.json` before any arm runs and aren't changed afterward.

**Arms.** Unedited. `doubt_low` and `doubt_high`, self-doubt at each target. `random_high`, the matched random direction at the high target. `rep_high`, repetition at the high target.

**Sign.** Both directions point toward their behavior: self-doubt's activity predicted doubt with AUROC 0.935 in study 3, and repetition's activity rose from 1.8 to 4.3 inside loops in study 7.

**Measures.** Doubt words per 100 words, with the same pattern as earlier studies. Loop entry by the rule in `loop_onset.py`. CIs are paired bootstraps over problems.

| Bar | Needed | Why |
|---|---|---|
| A1. Doubt at high dose | `doubt_high` minus `random_high` at least +1.0 per 100, CI above 0 | enough to cause doubt words? |
| A2. Doubt at low dose | `doubt_low` minus unedited at least +0.5 per 100, CI above 0 | does it work at a small dose? |
| A3. Dose response | `doubt_high` minus `doubt_low` above 0, CI above 0 | more of it, more doubt |
| A4. Specific | `doubt_high` minus `rep_high` at least +0.5 per 100, CI above 0 | not just any behavior direction |
| A5. Intact | `doubt_high` at most 2 more collapsed runs than unedited, else A1 to A4 are void | garbage text can't count as doubt |
| A6. Repetition causes loops | `rep_high` minus `random_high` at least +20 points on runs entering a loop, CI above 0 | the older 250-token test was short |

**How to read it.**

- A1, A3, A4 and A5 pass: adding self-doubt causes doubt words specifically, so the direction works in both ways.
- A1 fails and A5 passes: removal helps, but adding doesn't cause it, so the direction is needed without being enough.
- A4 fails: adding either direction raises doubt words, so this isn't specific to self-doubt.
- A6 passes: repetition causes loops when added over a longer run, and the older negative result was about length or dose.

Correct answers, answers given and doubt words in each run's first 500 words are reported with no bars. Correct has no room to fall, because only about 5 of 64 are right unedited, so it can't serve as a bar. The first-500 count is there because loops inflate doubt words by repeating them.

**Limits known in advance.** Doubt words are a word count, not a judgment of meaning. A fixed amount added at every token isn't how the model's own activity behaves. Calibration on teacher-forced text doesn't guarantee the same dose while generating. 6 bars with no correction, one seed, one model, 48 problems.

## Amendment, 2026-11-06, after a Fable review and before any edited arm generated

The unedited arm was still running, so no edited output existed. Three design faults were found.

1. **Doubt words are measured in each run's first 500 words for A1 to A4, not the whole run.** Loops repeat doubt words, and on earlier data looped runs averaged 1.66 per 100 against 0.97 for others, so a whole-run count confounds doubt with looping. Earlier loop entry sat at 750 words or later in the lower quartile. Whole-run rates are reported with no bar.
2. **A2 now compares `doubt_low` with a new arm, `random_low`,** the same random direction at the low target by the same rule. Study 2's random direction moved the count by +0.45 without doing anything specific, close to the +0.5 bar.
3. **A5 now covers every edited arm, and any bar using an arm with more than 2 extra collapsed runs is void.** A collapsed control would otherwise let a bar pass for free. A collapsed run also no longer counts as a loop for A6.

`random_low` runs after the others, using the same seed. Its amount was added to `add_gains.json` from the existing calibration curve. Other amounts didn't change. Realized doses are coarse: `doubt_low` 0.0117, `doubt_high` 0.096, `rep_high` 0.081, `random_high` 0.110.

## Addition, 2026-11-06, after scoring and before any reading

Doubt words are only a count. So Claude readers check meaning, labelled as Claude and blind to arm. For each problem, a reader sees the first 200 words from two arms in random order and picks the one where the writer second-guesses its own work more. Bars: for self-doubt high against random high, the self-doubt run is picked in at least 75% of 24 problems. For self-doubt low against random low, at least 60% of 24 problems. Each bar also needs a two-sided binomial p under 0.05. This was added after seeing the count results, so it's a check on them and it's weaker than a bar set before data.
