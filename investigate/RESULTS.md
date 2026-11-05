# Can NeuroSpeak debug a behavior?

Run on 2026-11-04 and 2026-11-05 with `brain.investigate`, using bars written first in [PLAN.md](PLAN.md) and [PLAN_READER.md](PLAN_READER.md).

**Short answer: on Qwen it worked for Chinese and repetition, and self-doubt fell. It failed on hedging, restating and bullet lists. For behaviors in general, no.** The second model failed too.

## Corrected rerun, which replaces the first table for four behaviors

A review found that example pairs were built with the model's thinking block open, while hedging, restating, bullets and Chinese were tested with it closed. I fixed that in `investigate` and reran those four with the same pairs, prompts, seed and checks.

| Behavior | Unedited | Direction removed | Matched random | Cost | Result |
|---|---|---|---|---|---|
| Hedging | 0.40 | 0.53 | 0.58 | +0.020 | Did not pass |
| Restating | 24.6% | 21.7% | 23.8% | +0.027 | Did not pass |
| Bullet lists | 35.7% | 43.6% | 37.5% | +0.008 | Did not pass, wrong way |
| Chinese | 78.6% | **1.8%** | 78.7% | +0.008 | Works |

**B1, corrected: 1 of 5, so it still fails.** The earlier failures weren't caused by the confound. Random directions alone moved hedging from 0.40 to about 0.57, so that score is too noisy here to detect a small change.

## Direction from 4 hand-written example pairs, as first run

| Behavior | Unedited | Direction removed | Matched random | Cost, nats per token | Result |
|---|---|---|---|---|---|
| Answering in Chinese when asked (share of CJK characters) | 78.6% | **1.2%** | 78.4% | 0.008 | Works |
| Repetition, 16 problems (repeated 8-grams) | 34.2% | **8.4%** | 29.8% | 0.035 | Works after a detector fix, see below |
| Self-doubt (per 100 words) | 0.45 | **0.16** | 0.44 | 0.007 | Could not count: below my floor of 0.5 |
| Hedging (per 100 words) | 0.40 | 0.41 | 0.57 | 0.022 | Did not pass |
| Restating the question (share of its 3-grams) | 24.6% | 21.8% | 22.5% | 0.011 | Did not pass |
| Bullet lists (share of lines) | 35.7% | 39.2% | 36.2% | 0.018 | Did not pass |
| distilgpt2 repetition | 46.3% | 57.2% | 47.9% | −0.008 | Did not pass, and went the wrong way |

## Direction learned from the model's own outputs

The second method had no hand-written examples, and it was worse.

| Behavior | Unedited | Direction removed | Matched random | Result |
|---|---|---|---|---|
| Hedging | 0.55 | 0.53 | 0.54 | Did not pass |
| Restating | 22.8% | 23.3% | 22.8% | Did not pass |
| Bullet lists | 38.8% | 35.8% | 39.1% | Did not pass |
| Self-doubt | 0.48 | 0.65 | 0.50 | Did not pass, wrong way |
| distilgpt2 repetition | 40.8% | 73.8% | 37.0% | Did not pass, wrong way, and cost 0.12 |

## Bars

| Bar | Needed | Result | Verdict |
|---|---|---|---|
| B1. Other behaviors on Qwen | at least 3 of 5 work | 1 of 5. Would be 2 if self-doubt's floor didn't apply | **Fail** |
| B2. Second model | distilgpt2 works | rose from 46.3% to 57.2% | **Fail** |
| B3. Calibration from examples alone | repetition works | failed as run, passes after the fix | **Fail as run**, see below |
| B4. Specificity | no unrelated direction reaches 22.8% repeats | hedging 47.2%, restating 26.8%, bullets 44.0%, Chinese 41.8% | Pass |
| B5. Off-target | own score falls most | Chinese 98%, others rose. Repetition 76%, self-doubt 59%, hedging 44% | Pass |
| M1. Learned directions | at least 3 of 5 work | 0 of 5 | **Fail** |
| M2. Learned, second model | distilgpt2 works | rose from 40.8% to 73.8% | **Fail** |
| E1. Evidence is enough | at least 75%, Claude readers | 83.3% | Pass |
| E2. Evidence beats nothing | 10 points, interval above 0 | +4.8 (−7.1 to +19.0) | **Fail** |
| E3. Evidence beats wrong evidence | 10 points, interval above 0 | +2.4 (−4.8 to +9.5) | **Fail** |

**B3 needs its reason stated.** The repetition run failed one check, output intact, by 1 collapsed trace against 0. Reading the trace showed 28 spaces of normal indentation, which my detector wrongly called collapse. I fixed the detector, added a test, and rescored every run with `investigate/rescore.py`. No other result changed. Passing depends on that fix, made after seeing the failure.

## Reports: Claude readers, not people

| Share right (168 judgments, 42 per condition) | No note | Wrong evidence | Name only | Evidence |
|---|---|---|---|---|
| All | 78.6% | 81.0% | 83.3% | 83.3% |
| Chinese | 100% | 100% | 100% | 100% |
| Repetition | 71.4% | 78.6% | 85.7% | 92.9% |
| Self-doubt | 64.3% | 64.3% | 64.3% | 57.1% |

Readers could mostly tell which output changed with no note at all. So this test can't show much value from reports, and evidence did no better than a name. Repetition points the right way on 14 trials per condition, which is too few to rely on. Batches ranged from 18 to 27 trials rather than 21 each, because readers were assigned at random.

## What this means

- **What held up.** Removing a direction built from four example pairs can turn some behaviors off, cheaply. That's new for Chinese. Self-doubt fell too, though the plan's floor means it doesn't count as a pass.
- **What isn't new.** The repetition direction from examples is almost the same vector as the weight edit's direction, cosine 0.997, because that edit was built from the same concept. So B3 retests one direction a different way, without changing weights. It doesn't show that examples find something the edit didn't.
- **What's a limit.** Hedging, restating and bullet lists didn't respond, including after the fix. Why is unknown. They may not sit in one direction at L19, four pairs may be too few, or the scores may be too noisy. "Strong behaviors" is my description of what worked, and it rests on two cases: Chinese, where the prompt asks for it, and repetition, which passed only after a detector fix.
- **Second model.** It doesn't carry to distilgpt2, where removal made repetition worse with both methods. "Any model" is still untested beyond the plumbing, and this is a real point against it.
- **`investigate` will say "did not pass".** It reported failures correctly in most runs here, so a person using it gets a warning, not false confidence.
- **Self-doubt is worth a look.** Its direction reached 21.3% repeats and 8/16 correct on the 16 problems, the highest correct count at seed 2 (unedited 5/16), and it has a 0.42 cosine with the weight edit's direction. That wasn't a bar, one seed, and n=16. A later test on 64 problems with a matched control and its own bars passed: 23/64 correct against 4/64, mostly because more runs gave an answer under the cap. See [the results](../whyfix/RESULTS.md).

## Limits found by review

- B4's directions for hedging, restating, bullets and Chinese came from before the context fix, and those arms were not rerun. A low repeat rate also doesn't prove much alone: removing the mean gave 10.9% repeats by collapsing the model. Only the 21.3% self-doubt arm, which kept reasoning normally, is worth following up.
- B5's rise in repetition for Chinese is an artifact, since the score splits on spaces and reads Chinese as zero.
- The reader test can't isolate what evidence adds. Output endings show whether a run finished or hit the cap, and removal makes runs finish more often, so readers can often tell with no note. Wrong evidence is also fixed per behavior, and self-doubt trials got repetition's, which is close to right. A and B balance per condition, not per behavior.

## Limits

Four pairs each, one seed per behavior apart from the earlier repetition work, 40 prompts per behavior, and I wrote the behaviors, the examples and the scores myself. Scores are simple word and pattern counts. The specificity test ran on one seed and the 16 problems. Only one matched random direction was drawn per behavior. Readers are Claude models.

Scripts: `investigate/behaviors.py`, `run.py`, `rescore.py`, `make_reader.py`, `score_reader.py`. Data: `runs/investigate/`, including `summary.json` and `reader/results.json`.
