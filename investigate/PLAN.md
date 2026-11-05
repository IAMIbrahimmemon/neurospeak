# Can NeuroSpeak debug behaviors other than repetition?

Written on 2026-11-04 before any result. Removing one direction fixed loops, and that's one behavior on one model. This asks whether `brain.investigate` works in general. Examples, prompts and scores are in `investigate/behaviors.py`, fixed before any run.

**What each run does.** Build a direction from 4 hand-written example pairs (mean difference at one layer). Generate on 40 prompts not in the pairs, three ways: unedited, with the direction removed at every token, and with a matched random direction removed (same overlap with the residual's mean). Same seed for all three. Measure normal-text cost.

**Bar for each behavior: `investigate` reports "works", meaning all of these hold:**

| Check | Needed |
|---|---|
| Present | unedited score at least the floor below, else "could not test", which counts as not passing |
| Falls | at most two thirds of unedited, paired 95% interval below 0 |
| Beats control | paired 95% interval against the matched random direction below 0 |
| Intact | no more collapsed texts than unedited |
| Cheap | at most 0.05 nats per token on normal text |

| Behavior | Model, layer | Score (higher means more) | Floor | Thinking, tokens |
|---|---|---|---|---|
| Calibration: repetition, 16 problems, seed 2 | Qwen L19 | repeated word 8-grams | 5% | on, 3,000 |
| Hedging | Qwen L19 | hedging words per 100 words | 0.3 | off, 200 |
| Restating the question | Qwen L19 | share of the question's word 3-grams in the answer | 0.05 | off, 200 |
| Self-doubt in reasoning | Qwen L19 | "wait", "hmm", "actually" and similar per 100 words, on GSM8K test problems 65–104 | 0.5 | on, 600 |
| Bullet lists | Qwen L19 | share of lines that start a list item | 0.1 | off, 200 |
| Answering in Chinese when asked | Qwen L19 | share of CJK characters | 0.3 | off, 200 |
| Second model: repetition | distilgpt2 L4 | repeated word 8-grams | 5% | none, 200 |

**Overall bars**

| Bar | Needed | If it fails |
|---|---|---|
| B1. Other behaviors on Qwen | at least 3 of the 5 report "works" | this method is mostly about repetition, and the plan to debug behaviors is wrong as written |
| B2. Second model | distilgpt2 reports "works" | the method doesn't yet carry across models |
| B3. Calibration | repetition reports "works" from examples alone | the example direction isn't enough. Weight-edit directions did pass, so I'd test those next |
| B4. Specificity | no behavior direction except self-doubt reaches at most 22.8% repeats on the 16 problems at seed 2 (unedited 34.2%) | a direction stopping loops without being about repetition means removal isn't specific |
| B5. Off-target | for each "works" behavior, its own score falls more, in share of unedited, than any other behavior's score on the same texts | the direction is broad, not specific to the behavior |

Self-doubt is excluded from B4 because it likely overlaps with loops. Its result is still reported. No dictionary names are used anywhere in the bars.

Limits known now: four pairs each, one seed each, hand-picked scores, and I wrote both the behaviors and the scores.

## Second method, written 2026-11-04 after B1 failed and before this runs

Results so far from hand-written pairs: Chinese works; hedging, restating, bullets and distilgpt2 repetition fail; self-doubt fell from 0.452 to 0.163 but missed my floor of 0.5, so it can't count. B1 is at most 1 of 5, so it fails.

**Idea to test.** Hand-written text may not look like what the model writes. New method: generate on training prompts with the model unedited, score each output, and take the direction as mean residual over the top third of outputs minus the bottom third. No hand-written examples. Then run the same checks on new prompts that didn't train it.

| Behavior | Training prompts | Test prompts | Floor |
|---|---|---|---|
| Hedging | the 40 questions | 40 new questions | 0.3 |
| Restating | the 40 questions | 40 new questions | 0.05 |
| Bullet lists | the 40 questions | 40 new questions | 0.1 |
| Self-doubt | GSM8K test problems 65–104 | problems 105–144 | 0.3, changed after seeing 0.452 at the old floor |
| distilgpt2 repetition | the 40 starts | 40 new starts | 5 |

Everything else is unchanged, including the five checks, layers, token limits and thinking settings.

| Bar | Needed | If it fails |
|---|---|---|
| M1. Learned directions | at least 3 of the 5 report "works" | neither method generalizes well, and NeuroSpeak should say it works for strong behaviors like language and looping, not behaviors in general |
| M2. Second model | distilgpt2 reports "works" | still no evidence the method carries across models |

Limit: a top-third split can pick up topic as well as behavior, and only the test prompts and the control can catch that.

## Amendment, 2026-11-05: a confound found by review, before this rerun

Example pairs were always built with the model's thinking block open, because `brain.concept` calls `chat_prompt` with thinking on. Hedging, restating, bullets and Chinese were then tested with thinking off, so their directions came from a different context than the one they were removed in. So those results don't fairly test the pairs method.

Fixed: `investigate` now builds pairs in the same context as its test. Rerun hedging, restating, bullets and Chinese, with the same pairs, prompts, seed and checks. These results replace the earlier ones for B1. Self-doubt, repetition and distilgpt2 are unaffected.

| Bar | Needed |
|---|---|
| B1, corrected | at least 3 of the 5 report "works", counting self-doubt's "could not test" as not passing |

If it passes, hedging or another style behavior isn't beyond this method, and the earlier explanation was wrong. If it fails, the failure is at least not from this confound.
