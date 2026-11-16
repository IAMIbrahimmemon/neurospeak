# Study 3: does the direction predict doubt before it happens?

Written on 2026-11-05 before measuring. Studies so far remove the direction and watch what changes. This one reads it without touching anything: on normal unedited runs, does activity along the direction tell us the model is about to write a doubt word? A direction that predicts behavior in text it wasn't built from starts to say what it does.

**Data.** All 64 unedited traces from rows 144–207 (`runs/whyfix/gsm64b_none_traces.jsonl`), truncated to their first 1,500 completion tokens. No direction was built from these traces. Doubt words use the same pattern as every earlier score (`DOUBT` in `investigate/behaviors.py`), and a token is positive if a match starts on it.

**Measure.** The projection of L19's output on the direction, read `lag` tokens before the word. AUROC is computed within each trace and averaged across traces with at least 3 positives. It answers: at a doubt word, is the projection higher than at another token of the same run? Intervals bootstrap over traces.

**Comparison directions.** Original self-doubt, writer 2, writer 3, repetition, and the matched random direction.

| Bar | Needed | Why |
|---|---|---|
| D1. Predicts the next token | original at lag 1, AUROC at least 0.70, CI above 0.5 | does it predict at all? |
| D2. Beats a control | original minus matched random at lag 1, at least +0.10, CI above 0 | |
| D3. Not just sentence starts | at lag 1, comparing only tokens right after a sentence or line break, AUROC at least 0.60, CI above 0.5 | doubt words open sentences, so a direction tracking sentence starts would pass D1 |
| D4. Leads the word | original at lag 5, AUROC at least 0.60, CI above 0.5 | is it a state before the word, or only the word's own setup? |
| D5. Different from repetition | original minus repetition at lag 1, CI above 0 | does it tell us something the loop direction doesn't? |

**How to read it.** D1 to D4 passing would give a tested reading: the direction is high when the model is headed toward doubting itself. It still wouldn't prove the direction causes doubt, or that this is all it does. If D3 fails, a large share of what looks like prediction is only sentence structure.

**Limits known in advance.** One model, one layer, one kind of problem. The doubt-word pattern misses doubt written other ways. Reading doesn't prove use: a direction can predict a word without the model using it.

## Amendment, after the first run

D3 as first run is void. It passed with 1 trace, because the rule didn't count a word after a list marker, such as `*   Wait`, as a line start, and that's where most doubt words sit in these runs. The other bars' numbers had been seen when this was written (D1 0.935, D2 +0.36, D4 0.747, D5 +0.29).

The new rule for D3: compare only tokens that begin a word and open a sentence or a line, with any list marker allowed before them. This makes negatives words like "So" or "The" in the same position, a stricter test than the first version would have been with enough data. D3 counts only with at least 20 traces. The 0.60 bar is unchanged. Since the other numbers are already known, this D3 was fixed with sight of them, though the D3 result itself on the new rule wasn't known.
