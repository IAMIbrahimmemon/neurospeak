# Study 24: constant or varying, at the right size

Written on 2026-11-09 at about 02:15, after study 23's offset and centered arms and before any generation for this study.

**Why.** Study 23's offset arm removed the corpus mean along self-doubt, μ·d = 0.169. On unedited GSM text the signed mean of h·d is 0.49 overall (0.23 in the first 500 tokens, 0.60 after 2,000), measured by `research/gsm_offset.py` on 12 traces (`runs/research/split_gsm_offset.json`). So the offset arm removed about a third of the real constant, and the centered arm left about 0.3 of it in place. Study 23 therefore mixed "which part" with "how much". Its offset null (2 of 64) is weak, and centered's shortfall against full removal (21 against 32) may just be the constant it left.

**Arms.** Self-doubt d at block 19's output, every token, same generation as studies 20 to 23 (GSM8K test rows 800 to 863, seed 20, 3,000-token cap, thinking on, batches of 8). The constant is c = 0.49, fixed before data.

| Arm | Edit | Flag |
|---|---|---|
| offset 0.49 | h − c d | `--mode offset --level 0.49` |
| centered 0.49 | h − ((h·d) − c) d | `--mode centered --level 0.49` |

Per-token removal in the offset arm is 0.49, near gain 0.65's average removal (0.65 × 0.807 = 0.52 mean absolute). That makes gain 0.65 (13 of 64) the size-matched comparison for "constant only".

| Bar | Needed | Why |
|---|---|---|
| L1. Constant alone | offset 0.49 minus unedited (4), signed CI | does a right-sized constant push help at all? |
| L2. Constant against a matched varying removal | offset 0.49 minus gain 0.65 (13), signed CI | same average size, one constant, one tracking the signal |
| L3. Leaving the constant | centered 0.49 minus full (32), signed CI | does the constant matter once variation is gone? |
| L4. Centered at both sizes | centered 0.49 minus study 23 centered (21), signed | does a right-sized constant close the gap? |
| L5. Intact | at most 2 more runs collapsed or with 200 spaces in a row than unedited | |

Secondary columns, no bars: correct with a non-empty last box, finished and right, doubt words per 100, "wait" per 1,000 tokens.

**Predictions, written before data.** Offset 0.49 lands near unedited, 4 to 8 correct (L1 CI includes 0, L2 below 0). Centered 0.49 lands near 25, between study 23's centered and full. I'm not confident in either.

**How to read it.**

- Offset 0.49 near unedited and below gain 0.65: tracking the signal matters, not the push. The removal story holds.
- Offset 0.49 near or above gain 0.65: a constant push of that size does as much. The edit works as steering, and that is reported first.
- Centered 0.49 near full: the varying part carries the whole gain once the constant is held at its real size. Centered 0.49 still well below full: the constant part adds something on top, so both parts matter.

**Limits known in advance.** One constant for the whole run, while the real mean drifts from 0.23 to 0.60. Early in a run the offset arm pushes past zero, and late it falls short. The 0.49 comes from 12 traces of one arm. One seed, one model, one layer, 64 problems.
