# Study 21: is it the depth or the direction?

Written on 2026-11-08 after study 20 was scored and reviewed, before any generation for this study. In study 20, directions built at L15 and L19 helped and directions built at L3 to L11 didn't. But each layer had its own direction, so that can't say whether early layers are the wrong place or the early directions are the wrong vectors. Fable's review found the early directions remove barely more than random (1.1 to 1.4 times their controls, against 2.8 to 4.6 late), which points to the vectors. This study holds the vector fixed: the original L19 self-doubt direction, removed at L7, L11, L15 and L23.

**Generation.** Same as study 20: GSM8K test rows 800 to 863, seed 20, 3,000-token cap, no penalty, thinking on, batches of 8, value-based scoring. Compared with study 20's unedited, L19 and per-layer arms on the same problems. The residual stream shares one basis across layers, so the same vector can be removed anywhere.

- Stage 1 arms: `dir_self_doubt` at L15, L11, L7 and L23, in that order.
- Stage 2 arms: a random direction with the same overlap with that layer's residual mean, removed at the same layer, at each layer that passes T1. The L11 control runs in any case, so at least one early layer has a control. The controls are built by `research/build_xfer_dirs.py` before any data. The L23 control's mean comes from block 23's output hook, not `hidden_states[24]`, which is post-norm.

| Bar | Needed | Why |
|---|---|---|
| T1. Helps, per layer | correct, L19 vector at layer l, minus unedited, CI above 0 | does the vector work away from where it was built? |
| T2. Beats control, per layer passing T1 | minus that layer's matched control, CI above 0 | |
| T3. Against the layer's own direction | L19 vector at l minus study 20's direction built at l, signed, CI excludes 0 for a winner | is the early failure the vector? |
| T4. Against L19 | L19 vector at l minus the same vector at L19, signed | how much does depth cost? |
| T5. Intact | at most 2 more runs collapsed or with 200 spaces in a row than unedited | |

Secondary columns, no bars: correct with a non-empty last box, and finished and right. Doubt words per 100, as in study 20.

**How to read it.**

- T1 passes at L7 or L11, and T3 favors the L19 vector there: early layers can carry the effect, and study 20's early failure was the vector built there.
- T1 fails at L7 and L11: early layers are the wrong place for this vector too, and depth matters.
- L11's control raises "wait" as much as study 20's L11 direction did: early removal of anything causes it, and study 20's doubt rise wasn't specific.

**Predictions, written before data.** L15 passes T1. L23 passes weakly. L7 and L11 I expect to fail, from study 20's pattern, but Fable's dose finding makes me less sure.

**Limits known in advance.** The vector has different overlap with each layer's residual, so the dose differs by layer. One seed, 64 problems, four layers. Stage 2 controls depend on stage 1, which is stated here in advance.
