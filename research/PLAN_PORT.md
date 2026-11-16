# Study 12: does the method carry to another model?

Written on 2026-11-06 before any data from this model. Everything so far is on Qwen3.5-0.8B. The framework claims to work on any model, and `investigate` already failed on distilgpt2. This asks whether the same steps, with the same 4 self-doubt pairs and no other change, find a working direction in MobileLLM-R1-360M, a model built for math.

**Model.** `facebook/MobileLLM-R1-360M-base`, 15 layers, width 1024, no chat template, so prompts are plain text from `chat_prompt`. Layer 12, the closest to Qwen's 19 of 24. Other layers aren't tried, so a fail here doesn't cover them.

**Runs.** GSM8K test rows 352 to 383, 32 unused problems. Seed 9, 1,500-token cap, same sampling as before. Removal at every token.

**Step 1, a gate.** Unedited first, on the 32 problems. There has to be a behavior to remove.

| Gate | Needed |
|---|---|
| Doubt exists | at least 0.3 doubt words per 100 words over whole runs, at least 10 of 32 runs with any |
| It reasons | at least 20 of 32 runs have at least 100 words |

If the gate fails, the study stops there and reports that this model doesn't show the behavior under this prompt, with sample text. No direction is built and nothing is tuned to make it pass. That's still a limit on "any model".

**Step 2, only if the gate passes.** Build the direction from the 4 original pairs, unchanged, at L12. Build the matched random direction with `matched_control`, using the mean L12 activation over the unedited traces, seed 1. Arms are removal of each.

| Bar | Needed | Why |
|---|---|---|
| M1. On target | doubt per 100 removed at most 2/3 of unedited, and removed minus matched has CI below 0 | same effect on the behavior? |
| M2. Answers | correct removed minus unedited at least +5 points, CI above 0, and removed minus matched CI above 0 | same effect on answers? |
| M3. Intact | at most 2 more runs than unedited that are collapsed or under 20% distinct words in the first 200 | study 9's review showed collapsed alone misses broken runs |

**How to read it.**

- Gate fails: the behavior isn't there to test.
- M1 and M3 pass, M2 fails: the direction carries, the benefit doesn't.
- M1, M2 and M3 pass: the method carries to a second model, for this behavior.
- M1 fails: the method fails on this model at this layer, as it did on distilgpt2.

**Limits known in advance.** One layer, one behavior, one seed, a base model with plain prompts unlike Qwen's thinking format, 32 problems, and pairs written for a different model's style. The correct rate may have little room for change either way.
