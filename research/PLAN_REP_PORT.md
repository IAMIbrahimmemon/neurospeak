# Study 13: does the repetition method carry to MobileLLM-R1-360M?

Written on 2026-11-06 after study 12's gate failed and before any data from this study. Study 12's 32 runs look repetitive, so repetition may exist where self-doubt doesn't. These are new problems, so that look doesn't touch the data here.

**Runs.** MobileLLM-R1-360M-base, L12, plain text prompts. GSM8K test rows 384 to 447, 64 unused problems, seed 10, 1,500-token cap. Removal at every token. The direction is built from `REPETITION_PAIRS` with no change. Matched control from `matched_control`, using the mean L12 activation over the unedited arm, seed 1.

**Gate.** Unedited repeats at least 20%, and at least a quarter of runs enter a loop by the rule in `loop_onset.py`. If not, stop.

| Bar | Needed | Why |
|---|---|---|
| Q1. Repeats fall | removed at most 2/3 of unedited, and removed minus matched has CI below 0 | does the direction do its job here? |
| Q2. Loops fall | removed minus matched on runs entering a loop at most -20 points, CI below 0 | same in the loop rule's terms |
| Q3. Answers | correct removed minus unedited at least +5 points, CI above 0 | does it help? |
| Q4. Control is quiet | matched minus unedited on repeats has CI that includes 0, or moves less than 1/3 as far as removal | a matched random edit shouldn't do this |
| Q5. Intact | at most 2 more broken runs than unedited, counted as collapsed or under 20% distinct words in the first 200 | |

**How to read it.**

- Q1, Q2, Q4 and Q5 pass: the repetition method carries to a second model with real weights, which distilgpt2 failed.
- Q1 fails: it fails here too, and two of two ports have failed.
- Q3 is separate. The direction can carry without answers improving.

**Note on Q5.** A model already looping heavily may count as broken at the start, so this bar compares arms and isn't an absolute test.

**Limits known in advance.** One layer, one seed, plain text prompts, base model that doesn't stop after answering, so scoring reads the last box. Repeats in LaTeX step lists may be partly normal formatting.

## Amendment, 2026-11-06, before reading any result from this study

A Fable review of study 12's traces found this model answers in about 200 words and then repeats itself: loopiness 10% before the first box and 92% after it. So the gate and Q1 and Q2 as written mostly measure repeating after the answer, and last-box scoring depends on the cap. The original gate and bars stay as written and are reported, but they don't answer the question. **The reading of this study now rests on text before each run's first box, or the whole run if it has none:** repeats there, the first box's correctness, and a new gate of at least 20% repeats before the first box. If that gate fails, the plan's reading is that this model doesn't loop while reasoning, and the bars measure only repeating after the answer. These amendments were made after the unedited arm had run and before the edited arms, per file times, with no score read by me.


Correction after a Fable review: the first box now means the first box containing a digit. One repetition run and two matched runs echoed a placeholder like `\\boxed{answer}`, which set their before-box text and first-box score wrongly. Verdicts are unchanged.
