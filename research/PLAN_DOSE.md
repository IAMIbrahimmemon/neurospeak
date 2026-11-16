# Study 8: with matched doses, what frees a stuck run?

Written on 2026-11-05 before any data from these arms. Study 7's removal of repetition moved 33 of 56 stuck runs out of their loop, and removing self-doubt moved 1. But repetition's edit was about 6 times larger, and it had no control with the same overlap with the mean. This asks which of three explanations holds:

- **Any big push frees a loop.** Then direction doesn't matter here.
- **One axis.** Self-doubt and repetition act through what they share.
- **Repetition has its own part.** It frees loops through what self-doubt lacks.

**Runs.** The same 56 prefixes, seed 7, 1,500 new tokens, and edits from the prefix's last token on, as in study 7. Unedited and repetition at gain 1 are reused from study 7, and their seed and code don't change.

**Dose.** An edit is `h - g (h@d) d`, where gain `g` is 1 for plain removal. Dose is mean per-token KL of the next-token distribution, with the edit applied to the last 128 tokens of teacher-forced text. The target is repetition's dose at gain 1. For each direction below that is marked "at dose", `g` is the value from the grid 1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64 whose dose is closest to the target on a log scale. Calibration uses 8 runs from `long_none` that entered a loop, chosen in file order, which aren't in the 56. Gains are written to `runs/research/dose_gains.json` before any rescue arm runs and aren't changed afterward. If even gain 64 is under a third of the target, that arm is reported as not reaching dose, and its bars are void.

**Arms.**

| Arm | Direction | Gain |
|---|---|---|
| `matched_mean` | random with repetition's overlap with the mean (`dir_matched`) | 1 |
| `doubt_dose` | self-doubt | at dose |
| `random_dose` | random (`dir_self_doubt_matched`) | at dose |
| `rep_only` | repetition with self-doubt projected out | at dose |
| `doubt_only` | self-doubt with repetition projected out (`dir_self_doubt_orth`) | at dose |

**Measure.** Escaped, unchanged from study 7, so results compare directly. Still out at the end, finished and correct are reported for each arm, with no bars. CIs are paired bootstraps over runs.

| Bar | Needed | Why |
|---|---|---|
| F1. Repetition beats its control | repetition minus `matched_mean` at least +20 points, CI above 0 | E4's missing control |
| F2. Direction matters at equal dose | repetition minus `random_dose` at least +20 points, CI above 0 | rules out "any big push" |
| F3. Self-doubt at dose | `doubt_dose` minus `random_dose` at least +20 points, CI above 0 | was E2's failure only about dose? |
| F4. Repetition's own part | `rep_only` minus `random_dose` at least +20 points, CI above 0 | does it work without the overlap? |
| F5. Self-doubt's own part | `doubt_only` minus `random_dose` at least +20 points, CI above 0 | does it work without the overlap? |

**How to read it.**

- F2 fails: a big enough push frees loops, and direction hasn't been shown to matter here.
- F2, F3 and F4 pass, F5 fails: self-doubt frees only through what it shares with repetition, so the one-axis view gains ground and repetition carries the effect.
- F2, F4 and F5 pass: two separate routes out of a loop.
- F2 and F4 pass, F3 and F5 fail: repetition has an effect self-doubt can't produce at this dose.
- F2 passes, F4 and F5 fail: only the shared part frees loops.
- Any other pattern gets reported as it is, without a story fitted to it.

**Limits known in advance.** Gains above 1 overshoot, so they push past removal and reverse the component. That isn't the same as removing it, so a high-gain result says what happens along this axis, not what removing the direction does. KL is one definition of dose. Calibrating on teacher-forced text doesn't guarantee the same dose while generating. `rep_only` and `doubt_only` overlap the mean differently from their parents. 5 bars with no correction for multiple tests. One seed, one model.
