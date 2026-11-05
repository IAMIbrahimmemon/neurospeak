# `diff` test plan: reading an edit in dictionary words

Written on 2026-11-04 before any result. `brain.diff(edit, texts)` reports which words fire more or less after a weight edit. A useful diff should say similar things about edits that do the same job.

**The check.** The README's two edits were made independently and act alike: Astra's (from its logs) and NeuroSpeak's. A random edit of the same size is the control. Run each over the same texts: the 16 unedited traces, first 256 tokens after skipping 2 per trace, plus a further 256 from the middle of each. Word change = mean activation with the edit minus without, at L19.

| Bar | Needed |
|---|---|
| D1. Astra and NeuroSpeak edits move the same words: Spearman of word changes over the 200 words that changed most across the two | at least 0.5 |
| D2. Random edit differs: Spearman of random vs each real edit, over those same 200 words | at most 0.2 in absolute value for both |
| D3. Concentrated enough to read: the top 20 words' share of total absolute change, for each real edit | reported, no bar |

A pass means the diff is measuring something stable about what an edit does. It doesn't show that the names of those words are right.
