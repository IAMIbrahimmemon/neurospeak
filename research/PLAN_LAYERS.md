# Study 20: at which layers does removing self-doubt help?

Written on 2026-11-08 before any generation for this study. Every result so far removes the self-doubt direction at L19, a layer chosen early on. If the effect is a property of how this model represents doubt, it should show at more than one depth, and how it changes with depth says where the behavior is built. Qwen3.5-0.8B has 24 layers, 0 to 23.

**Directions.** `research/build_layer_dirs.py` builds the direction from the original 4 pairs at L3, L7, L11, L15, L19 and L23, with the same code and context as the original. It first rebuilds L19 and stops unless the cosine with the original is at least 0.999. It came out at 1.0. Each layer also gets a random control with the same overlap with that layer's residual mean, built by `matched_control`, seed 1, from that layer's mean over the project's default corpus. Before any generation, the cosines with the L19 direction are: L3 0.47, L7 0.52, L11 0.56, L15 0.79, L23 0.68 (`runs/research/layer_dirs.json`). The new L19 control is a fresh random direction, not the `dir_self_doubt_matched` used before, because the corpus behind the mean has changed (cosine 0.80 with the old mean).

**Generation.** GSM8K test rows 800 to 863, 64 unused problems. Seed 20, 3,000-token cap, thinking on, no presence penalty, batches of 8. The direction is projected out at the output of the named block at every token, as before.

- Stage 1 arms: unedited, then the direction removed at L19, L15, L23, L11, L7 and L3, in that order.
- Stage 2 arms: the matched control at each layer that passes Y1, plus L19's control in any case.

**Scoring.** By the value-based rule used in every study (`right`: the last box, wherever it is). After study 19, the secondary score, finished and right, is pre-registered here as a secondary column, with no bar on it.

| Bar | Needed | Why |
|---|---|---|
| Y1. Helps, per layer | correct at layer l minus unedited, CI above 0 | does removal at this depth help? |
| Y2. Beats its control, per layer passing Y1 | correct at l minus l's matched control, CI above 0 | is it the direction, not any removal at that depth? |
| Y3. On target, per layer | doubt words per 100 at most 2/3 of unedited | does removal at this depth cut doubt? |
| Y4. Against L19, per layer | correct at l minus correct at L19, signed; CI excludes 0 for a winner | is L19 special? |
| Y5. Intact, per layer | at most 2 more runs collapsed or with 200 or more spaces in a row than unedited | |

A layer "carries the effect" only if Y1, Y2 and Y5 pass. Doubt words use `investigate/behaviors.DOUBT`, per 100 words, mean over runs, as in study 2. CIs are paired bootstrap 95% over problems, 10,000 resamples, seed 0 (`whyfix/gsm_score.ci`).

**Predictions, written before data.** L19 and L15 carry the effect. L3 doesn't. L7, L11 and L23 I can't call. If doubt cuts (Y3) and correct gains (Y1) don't line up across layers, cutting doubt words isn't the whole mechanism.

**How to read it.**

- A run of adjacent layers carries the effect: the behavior has a span of depths, and L19 was one choice among several.
- Only L19, or L15 and L19: the effect is local to a narrow middle-late band.
- Y3 passes at a layer where Y1 fails: removing doubt words there doesn't help the math, which separates the words from the gain.
- Y1 passes where Y3 fails: the gain at that depth comes from something other than fewer doubt words.

**Limits known in advance.** Removal is the full projection at each layer, not a matched dose. The residual's size and the direction's share of it differ by depth, so "fails at L3" could mean "too small a change at L3". The directions also differ by layer, so this compares direction-and-depth together, not depth alone. One seed, one model, 64 problems, six layers with no correction across them. The L19 arm should land near the replicated rate of 49 of 128, about 24 of 64. If it lands far below, the problem set or the code path is off, and I'll say so before reading the other layers.
