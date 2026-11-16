# Study 15: replicating the before-the-answer lead on MobileLLM-R1-360M

Written on 2026-11-06 before any data from these problems. Study 13's whole-run bars failed, but text before the first box showed a lead that had no bar: fewer repeats and more correct first boxes against the matched control. Against unedited, the gain's CI included 0. This tests it with bars set first and twice the problems.

**Runs.** MobileLLM-R1-360M-base, L12, plain text prompts. GSM8K test rows 512 to 639, 128 unused problems. Seed 12, 1,500-token cap. The directions are frozen from study 13, `runs/research/port_dirs.pt`, not rebuilt. Arms are unedited, repetition removed and matched removed. All measures use each run's first box and text before it, or the whole run if it has no box.

| Bar | Needed | Why |
|---|---|---|
| R1. Against control, answers | first-box correct, removed minus matched, at least +5 points, CI above 0 | does the lead replicate? |
| R2. Against unedited, answers | first-box correct, removed minus unedited, at least +5 points, CI above 0 | the question study 13 left open |
| R3. Against control, repeats | repeats before first box, removed minus matched, at most -5 points, CI below 0 | same direction of effect as study 13 |
| R4. Intact | at most 3 more broken runs than unedited, counted as collapsed or under 20% distinct words in the first 200 | scaled for twice the problems |

**How to read it.**

- R1 to R4 pass: the repetition method carries to a second model on what happens before the answer, the first port to work.
- R1 passes and R2 fails: the matched edit does harm and removal doesn't help, and there's no gain from removal itself.
- R1 and R3 fail: study 13's lead was noise.

**Limits known in advance.** The direction's job on this model may not be what its name says, since repeats are low here before the answer. One layer, one seed, base model, 4 bars with no correction.

## Amendment, 2026-11-06, while the unedited arm was running and before any score

The first box means the first box containing a digit, because a Fable review of study 13 found runs that echo a placeholder like `\\boxed{answer}`. The bars don't change.
