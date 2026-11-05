# `diff` test results

Run on 2026-11-04 on Qwen3.5-0.8B at L19, using the bars in [PLAN.md](PLAN.md), which were written before any result. 8,127 tokens from the 16 unedited traces.

| Bar | Needed | Result | Verdict |
|---|---|---|---|
| D1. Astra and NeuroSpeak edits move the same words | Spearman at least 0.5 | 0.61 | Pass |
| D2. A random edit differs from each | at most 0.2 in absolute value | 0.03 vs Astra, 0.17 vs NeuroSpeak | Pass |
| D3. Top 20 words' share of all change | reported, no bar | Astra 26%, NeuroSpeak 25%, random 9% | |

The two edits were made independently, and `diff` reports nearly the same picture of them. L19.f976 falls most under both: 0.80 to 0.05 per token under Astra's edit, 0.80 to 0.16 under NeuroSpeak's.

What this does and doesn't show:

- `diff` is stable: it gives alike answers for edits that act alike. That is the only thing the bars tested.
- It doesn't show the words caused the loop fix. Removing the words it finds did not reproduce the edits: see [the loop test](../whyfix/RESULTS.md).
- It doesn't test any name. The top word had no name when this ran, and none of the names in the report are marked tested.
- 74% of the change sits outside the top 20 words, so a short list leaves most of an edit undescribed.
- The random edit moved less in total (3.5 against 9.2), though it changes weights by the same amount. So total movement alone separates the real edits from the random one, and D2's rank test is the stronger check.
- One pair of real edits, on one model, on traces from one model version. Both edits remove nearly the same direction (cosine 0.75 at L15, 0.70 at L19), so agreement was likely. A harder test would compare edits that reach the same behavior by different directions.

Script: `difftest/run.py`. Data: `runs/difftest/results.json` and `report.txt`.
