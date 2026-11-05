# Is the loop direction spread across many words?

Written on 2026-11-04 before any result. Word lists failed to stop loops, and the concept edit works. The results offered an untested reason: the edit's direction may be spread across too many words. This measures that without generating text.

**Method.** At L19 on the 24 mining traces, every residual splits exactly: x − b_dec = Σ value × word direction + error. So x's projection on a unit direction d splits into one part per word and one for the dictionary's error. Sum each word's absolute part over all tokens. Do this for the `repetition` concept direction (the one the working edit removes) and Astra's removed direction if it can be read back, plus 20 random directions as the control.

| Bar | Needed | Meaning if it holds |
|---|---|---|
| S1. Spread | the top 16 words carry under 50% of the total absolute projection on `repetition` | a list of 16 can't carry the direction |
| S2. Not just any direction | the top 16 share for `repetition` differs from the random directions' range | spread is a fact about this direction, not about every direction |
| S3. Error share | error's share of the total, reported with no bar | how much of the direction the dictionary doesn't see |

Also reported, no bar: words needed for 50% and 90%; the overlap of the top 16 with active, why2, contrast2 and diff words; cosine of `repetition` with its nearest word.

S1 holding fits the explanation and doesn't prove it. The real proof would be an edit that removes only those words' share of the direction, which this doesn't run.

## Amendment, 2026-11-04, after the measurement and before any generation

The word-list line is closed, so this asks a different question: is the direction alone enough, when removed during generation at L19, with no words and no weight edit? Each arm subtracts x's projection on one unit direction at every token, seed 2, same 16 problems. The unedited arm is the existing one (34.2%).

Arms: **dir_neurospeak** and **dir_astra**, each edit's top direction at L19, and **dir_random**, one random direction.

| Bar, at seed 2 | Needed |
|---|---|
| F13. dir_neurospeak | at most 22.8%, and at least 10 points below dir_random |
| F14. dir_astra | same |

If these pass and word lists don't, the direction carries the effect and word removal doesn't reach it. If they fail, L19 alone isn't enough, since the real edits also act at L15 and change weights, and this says nothing about words.

## Amendment, 2026-11-04, after dir_neurospeak and dir_random at seed 2, before anything below runs

dir_neurospeak reached 10.3% with 7/16 correct, against 34.2% and 5/16 unedited and 39.2% and 1/16 for dir_random. One seed on 16 problems isn't enough, and other results in this line have failed to repeat. Before believing it:

| Bar | Needed |
|---|---|
| F15. Seed 0 | dir_neurospeak at most 30.6% (two thirds of 45.9%), and at least 10 points below dir_random at seed 0 |
| F16. Seed 1 | dir_neurospeak at most 27.6% (two thirds of 41.4%), and at least 10 points below dir_random at seed 1 |

**Accuracy test, separate, on problems nobody here has used.** 7/16 against 5/16 means little. If F15 and F16 pass: 64 new problems from `data/practice_sample.jsonl`, none in the first 16 test problems, the mining traces or the reasoning traces, same 3,000-token cap, one seed, arms unedited and dir_neurospeak.

| Bar | Needed |
|---|---|
| A1. Correct answers | dir_neurospeak at least 5 points above unedited, with a paired CI above 0 |
| A2. Repeats | dir_neurospeak at most two thirds of unedited |

## Amendment, 2026-11-04, before the accuracy test runs

The 300-problem sample doesn't have 64 unused problems left. A1 and A2 use problems from the GSM8K test split instead, picked by a fixed shuffle, with any that appear in `data/practice_sample.jsonl` left out. Bars don't change.

## Amendment, 2026-11-04, before the accuracy test runs

F15 and F16 passed, so the accuracy test runs. 64 GSM8K test problems, a seed not used so far (3), arms none and dir_neurospeak. Correct is judged by the same last-\boxed scorer as `summarize_traces.py`. CIs are paired bootstrap by problem.

## Amendment, 2026-11-04, after review and before these controls run

A cross-model review found that dir_neurospeak may overlap the residual's constant part (the dictionary's `b_dec`), while one random direction removes almost nothing. So "10 points below random" can't tell a repetition direction from any shrinking of L19's mean. Two new controls, seed 2, same 16 problems:

- **dir_mean**: `b_dec` as a unit vector.
- **dir_matched**: a random direction with the same cosine to `b_dec` as dir_neurospeak.

| Bar | Needed | If it fails |
|---|---|---|
| C1. dir_matched | above 22.8% | the effect comes from the shared mean part, not this direction |
| C2. dir_mean | above 22.8% | removing the mean alone also stops loops, so the direction isn't specific |

Until both are in, the results don't call this a repetition direction.

## Amendment, 2026-11-04, before measuring harm

Does removing the direction harm normal text? Next-token loss on held-out text from the word test's scan: Wikipedia, and GSM8K test solutions. Compared with no hook, with dir_random as a control. The weight edits cost about 0.03 nats per token on good reasoning, so that sets the scale.

| Bar | Needed |
|---|---|
| N1. dir_neurospeak on Wikipedia | loss rises by at most 0.05 nats per token |
| N2. dir_neurospeak on GSM8K solutions | loss rises by at most 0.05 nats per token |
