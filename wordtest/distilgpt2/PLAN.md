# Word test on a second model: distilgpt2

Written on 2026-11-04 before this dictionary was finished or any of its words were labeled. The framework claims to work on any model, and only Qwen3.5-0.8B has been through the word test. Repeating the test on a different architecture is the first real check of that.

Model: `distilbert/distilgpt2`, layer 4 of 6. Dictionary: 16× width, k=32, 12M tokens, same trainer. Same scripts, briefs, sample sizes per layer (100 features, 30 neurons), and scorer design as the Qwen run, using the balanced steering design from the start (Qwen's replication).

| Bar | Needed | Qwen result |
|---|---|---|
| Read: balanced accuracy, real labels | at least 75% | 71.5% (fail) |
| Better than neurons | CI above 0 | +13.0 (pass) |
| E1: steering, real sentence | at least 65% | 79.6% (pass) |
| Gate: null and floor | 45–55% | pass |

A model this small and old may simply have weaker words, so a miss would mean the method's results don't carry over cleanly. It wouldn't by itself mean the plumbing is wrong. Only 1 layer and half the features of the Qwen run, so intervals will be wider.
