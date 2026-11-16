# Research loop results

Each study has its plan and bars in this folder, written before its data. Everything here is on Qwen3.5-0.8B unless it says otherwise.

## Study 1: does the gain survive a 10,000-token cap?

Bars in [PLAN_LONG_CAP.md](PLAN_LONG_CAP.md). **L1, L2 and L3 pass, and L4 and L5 fail. By my plan's rule, L5 failing leaves "more than speed?" unsettled.** The gain didn't shrink at 10,000 tokens, but no unedited run finished, so this can't show what the gap would be if they all did.

32 unused problems, seed 5, value-based scoring.

| Arm | Correct at 3,000 | Correct at 10,000 | Finished by 10,000 | Boxed at 10,000 |
|---|---|---|---|---|
| Unedited | 2 | 2 | 0 | 2 |
| Self-doubt removed | 9 | 13 | 10 | 17 |
| Matched random removed | 1 | 1 | 1 | 1 |

| Bar | Needed | Result | Verdict |
|---|---|---|---|
| L1. Same effect at 3,000 | CI above 0 | +21.9 points (CI +9.4 to +37.5) | Pass |
| L2. Gain persists at 10,000 | at least +15, CI above 0 | +34.4 points (CI +18.8 to +50.0) | Pass |
| L3. Beats matched random at 10,000 | CI above 0 | +37.5 points (CI +21.9 to +53.1) | Pass |
| L4. Unedited catches up | CI above 0 | 0 points, the same 2 runs | Fail |
| L5. Long enough to tell | at least 16 of 32 unedited finish | 0 of 32 | Fail |

What this adds:

- **More time doesn't help the unedited model.** It ended with 2 correct at both caps. By 10,000 tokens, 96% of the 8-grams in the second half of each run are repeats, and 71% across the whole run.
- **Better math is still not shown.** Both arms boxed an answer on only 2 problems, and both were right on those. Of the self-doubt arm's 17 boxed answers, 13 were right.

### Exploration: loops look like a trap

No bars, found while waiting on study 1 (`loop_onset.py`). A run enters a loop at the first 200-word window where at least 90% of 8-grams already occurred earlier in the run, and leaves when a later window falls under 50%.

| Runs | Entered a loop | Left one | Finished after entering | Median entry, in words |
|---|---|---|---|---|
| Unedited, 10,000 cap | 31 of 32 | 0 | 0 | 1,600 |
| Self-doubt removed, 10,000 cap | 22 of 32 | 0 | 0 | 2,300 |
| Matched random, 10,000 cap | 31 of 32 | 0 | 0 | 1,550 |
| Unedited, two 3,000-cap sets | 56 of 128 | 0 | 0 | 1,150 and 1,050 |
| Self-doubt removed, two 3,000-cap sets | 17 of 128 | 2 | 0 | 1,250 and 1,100 |

- **Correction.** After the 3,000-cap counts I said removal looked like it mainly stops loops from starting. The 10,000 cap shows that was too strong: 22 of 32 still enter one. It delays them and lets some runs finish first.
- **In all of these runs, nothing finished after entering a loop.** Every finish came before any loop. That's why [study 7](PLAN_RESCUE.md) tests whether removal can free a run already stuck.

Limits: 32 problems, one seed, one model. The loop rule was chosen after a first version failed on these same traces. My first rule reported 6 of 32 looped against 96% overall repeats, because it couldn't see loops with long periods.

Scripts: `score_long.py`, `loop_onset.py`. Data: `runs/research/long_results.json`, `loop_onset.json`.

## Study 2: do independently written examples find the same direction?

Bars in [PLAN_PAIRS.md](PLAN_PAIRS.md). **Both writers pass all five bars as scored, but writer 2's pass over random depends on one scorer miss. Writer 3's holds either way. Two Claude writers given the same brief got the effect, so it doesn't need my exact 4 pairs. That's weaker than saying it doesn't depend on how the examples were written.** The brief named self-doubt, and both sets copy the original's format. Their directions are closer to each other (cosine 0.80) than to the original (0.63, 0.67), which fits the brief shaping the vector. A third writer produced nothing, so this is two sets, not three.

64 unused problems, seed 6, 3,000-token cap, value-based scoring.

| Arm | Correct | Boxed | Finished | Doubt per 100 words | Repeats | Entered a loop |
|---|---|---|---|---|---|---|
| Unedited | 5 | 6 | 2 | 1.24 | 44.7% | 25 |
| Writer 2 (Fable) | 17 | 19 | 3 | 0.32 | 35.1% | 21 |
| Writer 3 (Haiku) | 25 | 29 | 12 | 0.28 | 32.0% | 19 |
| Matched random | 10 | 11 | 5 | 1.69 | 46.1% | 29 |

| Bar | Needed | Writer 2 | Writer 3 | Verdict |
|---|---|---|---|---|
| P1. Same direction | cosine with the original at least 0.3 | 0.63 | 0.67 | Pass, both |
| P2. Beats unedited | at least +5 points, CI above 0 | +18.8 (CI +9.4 to +28.1) | +31.3 (CI +20.3 to +43.8) | Pass, both |
| P3. Beats matched random | CI above 0 | +10.9 (CI +1.6 to +21.9) | +23.4 (CI +14.1 to +34.4) | Pass, both |
| P4. Intact | at most 2 more collapsed | 0 | 0 | Pass, both |
| P5. On target | doubt at most 2/3 of unedited | 26% of unedited | 23% of unedited | Pass, both |

What this adds:

- **A close direction does a similar thing.** Directions at cosine 0.63 and 0.67 with the original cut doubt words by about three quarters and raise correct answers.
- **Writer 2's margin over random is fragile.** A random-arm run boxed `800\text{ or }800` against a right answer of 800, and the frozen scorer marks that wrong. Counting it as right gives random 11 of 64 and writer 2 +9.4 points with a CI low of exactly 0, which fails P3. The plan's verdict stays as scored, but I wouldn't rely on it. P2 and P3 also have no correction for two tests. The random arm's 10 or 11 correct is the highest any random control has reached so far, so controls can move this much on one seed.
- **Still more answers, not better math.** When an arm boxes an answer, it's right at about the same rate: 5 of 6, 17 of 19, 25 of 29, 10 of 11.
- **The gain looks like writing the answer sooner, not escaping loops.** Entering a loop fell only from 25 to 21 and 19, and repeats only to 35% and 32%, which I didn't predict. Writer 2 boxed 16 of its 19 inside thinking, and 14 of its 17 correct runs still hit the cap. Of writer 3's 25, 15 hit the cap, and 3 of its correct boxes sit inside a loop. So "correct" here often means the right answer got written down before the cap, not that the run finished. That reading is inferred: the direct evidence is the timing above, and the review's check that 38 of 58 unboxed unedited runs contain the right value somewhere in their text is only a rough proxy, since that value can appear for other reasons.
- **The answers aren't echoes.** The boxed value matches a number in the problem in 1 of 17 and 1 of 25 correct runs.
- **P5 isn't inflated by writers' wording.** Their example texts have 0 to 2 doubt-word matches each, against 9 in each of mine.

Limits: both writers are Claude. The brief named the behavior, so this can't show the behavior is found without naming it. One seed, one model.

Scripts: `build_pair_directions.py`, `run_pairs.sh`, `score_pairs.py`, with the repeats column from `neurospeak.loopiness` and the loop column from `loop_onset.py`. Data: `runs/research/pairs_results.json`, `pair_cosines.json`, `loop_onset.json`.

## Study 30: is it doubt, or not concluding?

Plan in [PLAN_FACTOR.md](PLAN_FACTOR.md) (plan hash 25a87433b8acc0a9), written before the new directions were built or any generation ran. **The split didn't come out as designed, so the study can't settle doubt against concluding. The doubt-only direction came out as a near copy of d (cosine 0.87), and the conclude-only one came out unrelated to d (0.04). Removing the doubt-only direction got 17 of 64 right, against 4 unedited and 32 with d removed. Removing the conclude-only direction got 5, which is nothing. The doubt-only direction tracks d's rise and fall almost exactly but misses its steady level, so this arm mostly repeats study 23's flattening (19 to 21).**

d's example pairs contrast doubting with concluding. Study 30 built two new directions from 4 pairs each on the same 4 problems (`research/factor_pairs.json`), with the same code at block 19. The doubt-only direction contrasts doubt words with plain next steps ("Next, ... Then ..."). The conclude-only direction contrasts plain next steps with concluding ("The answer is 15, and that is the final answer."). Each was removed at block 19's output on GSM8K test rows 800 to 863, seed 20, 3,000-token cap, as in studies 20 to 28. Study 20's unedited and d-removed arms are reused.

**Part A, the directions.** The build code rebuilt d at cosine 1.0. The doubt-only direction has cosine 0.87 with d, and the conclude-only one 0.04. They have −0.33 with each other. A fit of d on the two has R² 0.87, with weights 0.99 on doubt-only and 0.36 on conclude-only. I predicted cosines of about 0.6 and 0.4, so both were wrong, and the plan's reading rules assumed two parts of similar size.

| Arm | Correct | Finished | Closes thinking | Doubt words per 100 |
|---|---|---|---|---|
| Unedited | 4 | 0 | 0 | 1.42 |
| d removed (study 20) | 32 | 17 | 19 | 0.23 |
| Doubt-only removed | 17 | 5 | 5 | 0.40 |
| Conclude-only removed | 5 | 1 | 1 | 1.80 |

| Bar | Result |
|---|---|
| F1. Doubt part helps | **pass**: +20.3 (CI 9.4 to 32.8), 15 better, 2 worse |
| F2. Conclude part helps | **fail**: +1.6 (CI −4.7 to 7.8) |
| F3. Doubt-only minus conclude-only | +18.8 (CI 7.8 to 29.7) |
| F4. d minus each | doubt-only +23.4 (CI 10.9 to 35.9), conclude-only +42.2 (CI 29.7 to 54.7) |
| F5. Intact | pass, 0 collapsed runs in either arm |

**Why this doesn't answer the question.** Reviewed by Claude (Fable), which read block 19 on 12,000 unedited tokens.
- The doubt-only direction's reading correlates with d's at 0.97 token by token. Removing it takes away 92% of the variation in h·d but only 5% of its average level (0.50 to 0.47). So this arm acts much like study 23's flattening of d, which got 19 and 21, and 17 fits that.
- The part of d the doubt-only direction misses holds almost all of d's steady level, a projection of about 0.95 on 97% of tokens. That part leans toward the conclude-only direction (cosine 0.65), but the conclude-only arm didn't remove it.
- Removing the conclude-only direction leaves h·d unchanged. So its 5 says nothing about whether d's concluding side matters.
- "Doubt part" and "varying part" make the same prediction here, and this study can't tell them apart.

**What it does show.** d's rise and fall during a run is almost entirely what separates doubt words from plain next steps (r = 0.97). That fits the self-doubt name for the varying part of d, the part study 23 found carries most of the gain. A direction built only from concluding text, unrelated to d, does nothing when removed.

**Limits and side notes.**
- 15 of doubt-only's 17 credits are problems d also solved.
- On the strict column, finished and right, doubt-only gets 5 against d's 17.
- d's 32 is high against its replications, which give about 27 of 64. Against 27 the gap to doubt-only is about 10 points, and that comparison is unpaired.
- Removing the doubt-only direction also removes a steady push of about 0.8 along a direction outside d, probably the "Next, ... Then" style. d's removal doesn't do this, and its effect is untested.
- The 13% of d outside the two directions leans toward repetition and hedging, and neither arm tests it.
- The conclude-only arm's rise in doubt words (1.80 against 1.42) comes from a few traces, and the medians are equal, so it is noise.
- I wrote the pairs knowing the hypothesis. One seed, one model, one layer, 64 problems.
- The doubt-only arm was paused after 8 problems and resumed five days later. Resume reseeds at the batch boundary. Code and directions were unchanged across the pause, and the first 8 (1 right) don't stand out from the rest (16 of 56). `whyfix/generate.py` and `research/` aren't committed, so none of this is pinned in git.

**What would settle it.** Three arms: remove only the part of d outside the doubt-only direction, which carries d's steady level; remove a copy of d tilted to cosine 0.87 toward a random direction, to see whether 17 is just overlap; and remove the doubt-only direction plus d's steady level, which should recover d's full gain.

Reviewed by Claude (Fable), which rescored all four arms with its own parser and bootstrap and confirmed every count. Its CIs differ from these by at most one problem at the upper end.

## Study 29: does the self-doubt signal fire when the model finds its answer, and does it predict concluding?

Plan in [PLAN_MOMENT.md](PLAN_MOMENT.md) (plan hash 0ed7d2dffe6c3d20), written before any activation was measured. **Mostly a failed study. The signal is a little higher where the model first writes the right answer, but after the review's fixes the effect is small, about 0.09 standard deviations of h·d. The test of whether it predicts concluding measured the wrong thing and can't be read. The third bar couldn't fail as designed.**

`research/moment.py` fed the 192 unedited traces at the 3,000-token cap back through the model and read block 19's output along d. An event is a "= number" in the text. The answer event is the first "= <right answer>", found in 127 traces. The score is h·d at the event token minus the mean over the 200 tokens before it. The same scores were computed for the repetition direction and one random direction (seed 0). Output is in `runs/research/moment.json`.

| Bar | As planned | After the review | Verdict |
|---|---|---|---|
| M1. Signal at the answer | +0.19, CI 0.13 to 0.25 | +0.086, CI 0.033 to 0.139, against only earlier results | passes, small |
| M2. Predicts concluding | AUC 0.41, CI 0.32 to 0.51, wrong sign | the label doesn't measure concluding | can't be read |
| M3. Not just the next word | same as M1 | couldn't fail | vacuous |

**M1 was mostly a comparison with later text.** Reviewed by Claude (Fable). Per trace, a median of 7 intermediate results come before the answer and 25 after it. The later ones sit in checking text with three times the doubt words. Against only the results before the answer, d's rise is +0.086, which I reran and confirmed. Against the nearest earlier result it is +0.048 (CI 0.006 to 0.093). In units of the signal's spread on unedited text (0.91), that's about 0.09 standard deviations.

**The random direction also moved.** In the original design it shifted as much as d did, in the other direction, once each is scaled by its own spread. After restricting to earlier results it still moved (−0.024, CI −0.042 to −0.006). Only after also matching on whether "$" follows the number did the random direction's shift lose significance while d's held (+0.22 of its own spread, CI 0.07 to 0.36). d's score also tracks text style, rising with the density of digits and "$" in the window before it. So the small rise may be partly text style rather than anything about finding the answer.

**M2 measured a planning heading, not concluding.** Of the 24 "concluding" events, 22 matched "final answer" and 2 matched "the answer is". None was a box or the end of the thinking block, and 23 of the 24 still ran to the cap. The phrase comes from the model's habit of writing outlines ("5. Review and Format: ... final answer"). That's why a random direction "predicted" it (AUC 0.69). Those events sit earlier in the run and in denser formatting. A strict label has no positive cases in unedited traces, so this question needs traces that actually finish, from an edited arm.

**M3 couldn't fail.** Qwen splits numbers into single digits, and the answer's last digit is usually followed by "$", "." and a line break, so no doubt word can start within 3 tokens. With a 10-token window, 24 answers have one, and dropping them leaves M1 about where it was.

**What it means.** d's activity is slightly higher where the model first writes the right answer, at a size close to what text style alone produces. This study gives no evidence that d's level predicts what the model does next. The agenda question "does the direction's activity predict behavior on held-out text" stays open. Answering it would need traces that conclude, raw as well as baselined h·d, and many random directions.

**My predictions.** I predicted M1 at 0.3 to 0.8, M2 at AUC 0.65, and no shift for the random direction. All three were wrong. The plan's design was weaker than I thought when I wrote it. Its event windows mixed checking text into the comparison, its conclude label caught outline headings, and its M3 window was shorter than the digit tokenization allows.

Reviewed by Claude (Fable), which re-tokenized all 192 traces and matched every event index, then recomputed every comparison above from the saved rows.

## Study 28: is removing self-doubt more than banning doubt words?

Plan in [PLAN_BAN.md](PLAN_BAN.md) (plan hash ec8046d5ddfcd379), written before any generation or logit-lens measurement. **Banning the 11 whole-word tokens for "wait", "hmm" and "actually" at the output got 13 of 64 right, against 4 unedited and 32 for removing self-doubt. It recovers about a third of the edit's gain and almost none of its finishing. 2 banned runs finished under the cap, against 17 with self-doubt removed. Self-doubt removal beats the ban by 30 points (CI 17 to 42), 21 problems to 2.**

The ban sets those tokens' logits to minus infinity at every step, on the unedited model, before temperature and top-k. The tokens never appear in the 64 outputs, and none of rows 800 to 863 use these words in the problem text, so the ban cost no ordinary uses. Generation matches study 20: rows 800 to 863, seed 20, 3,000-token cap. The run had no restarts. Part of it ran several times slower because something outside the job was sharing the GPU, which doesn't change outputs.

| Arm | Correct | Finished | Closes its thinking | Answer written but run cut | "I will" per 1,000 tokens |
|---|---:|---:|---:|---:|---:|
| unedited | 4 | 0 | 0 | 42 | 0.18 |
| doubt words banned | 13 | 2 | 3 | 45 | 1.07 |
| self-doubt removed | 32 | 17 | 19 | 32 | 0.62 |

| Bar | Result | Verdict |
|---|---|---|
| B1. The ban works | doubt words 0.10 per 100 | pass |
| B2. Does the ban help? | +14.1 points, CI 4.7 to 23.4 | yes |
| B3. Self-doubt beats the ban | +29.7, CI 17.2 to 42.2 | pass |
| B4. Stuck runs (no box, right number in the text) | 24 against 26 unedited and 11 self-doubt | not comparable, see below |
| B5. Intact | 0 collapsed runs | pass |

B1's 0.10 is entirely words the ban didn't cover: "re-read" (81), "double-check" (12) and one "WAIT". The banned words themselves are at zero. B3 also passes against self-doubt's replication level of about 27.

**B4 was a poor measure.** It counts runs with no box anywhere, so an arm that boxes more scores lower whether or not it stops. Reviewed by Claude (Fable): better measures all say the same thing. Runs that write the right number but get cut at the cap are 42 unedited, 45 banned and 32 with self-doubt removed. After the first right box, the median run goes on for 1,896 tokens unedited, 1,631 banned and 690 with self-doubt removed. On every measure the ban sits with unedited.

**What the model said instead.** My prediction was that other checking phrases would rise sharply. That was wrong. The planned substitute phrases fell (0.64 per 1,000 tokens, against 1.01 unedited). The direction's other top words from Part A didn't take over either. "reconsider" and "rethink" were near zero, and no arm wrote any Chinese. What rose was checklist talk. "I will" went from 0.18 to 1.07 per 1,000 tokens ("I will check the LaTeX syntax. I will make sure the explanation is clear"), "a typo for" appeared 68 times, and some runs re-derived the whole solution. The model with the ban doesn't doubt in new words. It keeps going in other words.

**Part A, logit lens.** Pushed through the final norm and unembedding, d's top token is "wait", ranked first of 248,320. Its top 30 also include "WAIT", "reconsider", "rethink", "hesitation", and Chinese for "review the question", "carefully" and "look again". The tokens it pushes down most include "Consequently", Chinese "therefore", "remaining", "respectively" and the end-of-text token. On 4 unedited traces, removing d cut the 11 banned tokens' probability 2 to 8 times. Output in `runs/research/ban_lens.json`.

**What it means.** Banning the words gets part of the gain. Removing the direction does something more that a word ban can't. The extra may not be doubt. Reviewed by Claude (Fable): d may be better described as a "keep thinking, don't conclude" direction. Its bottom tokens are concluding words and the end-of-text token. Self-doubt removal closes the thinking block in 19 runs, against 3 for the ban and 0 unedited. Doubt words may top the lens because they are this model's commonest way to keep a thought going. The data fit either a doubt state or a "conclude" signal, and this study can't tell them apart.

**What this does not show.**

- "Correct" means a right box appears somewhere. 11 of the ban's 13 correct boxes, and 15 of self-doubt's 32, sit inside thinking that was cut at the cap and never ended. So self-doubt removal doesn't end the stuck state either. Half its correct runs box and then run to the cap, some in a verbatim loop.
- Any edit at block 19 acts upstream of the output, so "more than a word ban" alone isn't evidence of a doubt state.
- One seed, one model, 64 problems. The ban leaves variants such as "WAIT" and "Hmmm".

My predictions: the ban's 13 fell inside the predicted 6 to 14 and B3 passed. The prediction that substitute checking phrases would rise sharply was wrong.

Reviewed by Claude (Fable), which checked the ban processor's place in generation, re-tokenized all outputs, checked the scoring rule for inflation and computed the alternative stuck measures. I reran its thinking-closed, cut-with-answer and "I will" counts myself, and they match.

## Study 27: does a working learned direction with no self-doubt in it also help?

Plan in [PLAN_REPORTH.md](PLAN_REPORTH.md) (plan hash f5ca075d749ee2ab). The plan says it was written "about 06:35". That is wrong. The file was saved at 05:57:40, nine seconds before generation began at 05:57:49, so it still came before any data. The correction is logged rather than edited into the plan. **Removing repetition with its self-doubt part taken out got 7 of 64 right, against 4 unedited and 32 for removing self-doubt. It cut verbatim repetition hard, but the model stayed stuck. Self-doubt removal beats it by 39 points (CI 28 to 52), and won all 25 problems where the two disagreed.**

This is the control study 25 should have been. Repetition (`dir_neurospeak`) is a learned direction that demonstrably changes GSM thinking (study 14) but overlaps self-doubt (cosine 0.42). `dir_both_rep` is that direction with self-doubt projected out (cosine 0.000 with self-doubt, 0.908 with raw repetition). Both were removed at block 19's output on rows 800 to 863, seed 20, 3,000-token cap. Study 20's unedited and self-doubt arms are the baselines, run with the same rows, seed and cap. All 128 new runs were checked for those settings.

| Arm | Correct | Finished | Repeats (8-gram) | Repeats (4-gram) | Runs in a loop | Doubt words per 100 |
|---|---:|---:|---:|---:|---:|---:|
| unedited | 4 | 0 | 46.7 | 62.7 | 28 | 1.42 |
| repetition-only | 7 | 4 | 17.4 | 36.3 | 5 | 2.06 |
| raw repetition | 12 | 8 | 10.2 | 26.6 | 2 | 0.83 |
| self-doubt | 32 | 17 | 21.2 | 37.4 | 11 | 0.23 |

| Bar | Result | Verdict |
|---|---|---|
| R1. The control works | repeats 46.7 to 17.4, cut by well over a third | pass |
| R2. Does it help? | +4.7 points, CI 0.0 to 10.9 | not shown |
| R3. Self-doubt beats it | +39.1, CI 28.1 to 51.6, 25 to 0 | pass |
| R4. Raw minus repetition-only | +7.8, CI −3.1 to 18.8 | unresolved |
| R5. Intact | 0 collapsed runs in every arm | pass |

Repetition-only's 7 correct problems are all among self-doubt's 32, and they include all 4 of unedited's.

**R1 passes only on the measure the plan named.** The plan's repetition measure counts repeated 8-word runs. Counting 4-word runs instead, repetition-only (36.3) and self-doubt (37.4) are tied. Reviewed by Claude (Fable): 3-word runs and repeated whole sentences are tied too. The traces show why. In rows 801, 824 and 847 the model holds the right number and keeps cycling through "Wait, is there a trick? Let's re-read" in fresh wording. 26 of repetition-only's runs never box an answer but do write "= <the right answer>" somewhere, as many as unedited (26) and more than self-doubt (11). Re-read phrases and "wait" both rise, and so do doubt words.

**What it means.** Breaking verbatim repetition isn't enough to fix the math. The edit stopped the model from copying itself, and it went on second-guessing in new words instead. Removing self-doubt ends that stuck state, and this control does not. So self-doubt is specific against a direction that works on its own behavior, which study 25 could not show.

**What this does not show.**

- It does not test whether ending the stuck state by other means would be enough. Breaking verbatim loops is not the same as ending the stuck state. The penalty arms of studies 18 and 19 are the closer comparison for that.
- Dose is not matched, and the repetition edit is large. On unedited text the repetition directions have a mean h·d of 2.71 with spread 1.17, the same sign on 99.9% of tokens. Self-doubt's mean is 0.50, spread 0.91. Removing 2.71 from every token is more than block 19's whole MLP output (norm 2.13) against |h| of 7.27. Its varying part (1.17) also exceeds self-doubt's. So damage and loop-breaking can't be separated here. A strength-matched or centered repetition arm would be needed to say anything about size.
- R4 doesn't settle whether raw repetition's gain rides on its self-doubt part. Raw repetition still beats unedited (+12.5, CI 3.1 to 21.9), the same direction as study 14 but lower (12 against 22, on different rows with a different seed).
- One seed, one model, one layer, 64 problems. Self-doubt's 32 is high against its replications (about 27 of 64), and R3 still passes against that.

My predictions were repetition-only 8 to 16 and raw repetition 18 to 24. Both came in low.

Reviewed by Claude (Fable), which recomputed the repetition measures, read traces from each arm and checked the scorer, baselines and directions. The 4-gram figures and the held-answer count were rerun by me and match its finding.

## Study 26: which parts of the model write the self-doubt signal?

Plan in [PLAN_WRITERS.md](PLAN_WRITERS.md) (plan hash 5df63df25d1144b8), written before the decomposition was run. **No small set of parts writes the signal. The top 3 parts carry 26% of its token-to-token variation, under the 30% the plan set in advance, so the planned test of blocking them was skipped. 79% of the variation is written in blocks 13 to 19. It takes 7 parts to carry half and 14 to carry 80%.**

Each block adds its token mixer output (attention or linear attention) and its MLP output to the residual stream. So the self-doubt projection at block 19's output, h·d, is exactly the sum of 41 terms: the embedding, plus a mixer and an MLP term for each of blocks 0 to 19. `research/writers.py` splits it on the completion tokens of the first 12 unedited traces from study 20, 36,000 tokens in all. The 41 terms reproduce h·d to under 1e-5 relative error. Each term's "variance share" is cov(term, h·d) / var(h·d). The shares sum to 1. Output is in `runs/research/writers.json`.

| Blocks | Share of variation |
|---|---:|
| Embedding | 0.01 |
| 0 to 12 | 0.20 |
| 13 to 19 | 0.79 |

These are signed shares. Nine small MLP terms in blocks 2 to 11 are negative (together −0.066), which is why blocks 13 to 19 can carry 79%. Measured by absolute share instead, blocks 13 to 19 carry 70%.

The top parts are the MLP at block 18 (10.0%), attention at block 15 (8.4%), the MLP at block 16 (7.9%), linear attention at block 16 (7.3%) and the MLP at block 19 (7.2%). Mixers carry 62% in all and MLPs 38%. h·d runs 1.62 higher on tokens just before a doubt word (the next 3 tokens contain one) than elsewhere, about 1.7 standard deviations. Every top part is higher there too. This part is close to circular. d was built from example pairs whose doubt side reads "Wait, let me double-check. Hmm, actually... re-read" (`investigate/behaviors.py:241-245`), the same words the doubt counter matches, so it is a sanity check that re-finds study 3, not a new finding. "Before" is also loose, since tokens inside a doubt phrase count when the next 3 tokens hit.

**What it means.** The self-doubt signal is built gradually over the last seven blocks before block 19, by many parts at once. This fits study 21, where removing the direction helped at block 15 but little at blocks 7 and 11. No single block component (a whole mixer or whole MLP) writes it. This split can't rule out one head inside attention 15 or a few neurons inside MLP 18 carrying their part's share.

**What this does not show.** Variance share isn't causal. A part can carry variation that it only passes along from earlier parts. d was built at block 19, so projections of early parts onto it may mean something different. 12 traces of one arm. My prediction was a top 3 of 30 to 50% in blocks 12 to 19. The location was right, but the concentration was lower than predicted. The top 3's 26% is close to the 30% bar, and that bar was a judgment call, so skipping Part B is per plan rather than a strong negative.

Reviewed by Claude (Fable) after it was first written. The corrections above (absolute share, the circular doubt-word check, "block component" rather than "head or neuron") come from that review.

## Study 25: does removing other learned directions do the same?

Bars in [PLAN_OTHERDIR.md](PLAN_OTHERDIR.md) (plan hash de8a62f8c8183995, written before data). **Removing three other learned directions at L19 doesn't help: bullet lists 5, Chinese 3, restating 4, against 4 unedited and 32 for self-doubt. But the review found these directions do nothing measurable on math text at all. So the study only shows that inert directions don't help. It is a weak control.**

| Arm | Correct | Last box not empty | Finished and right | Doubt per 100 words | "wait" per 1,000 tokens | Dose against self-doubt |
|---|---:|---:|---:|---:|---:|---:|
| Unedited | 4 | 2 | 0 | 1.42 | 6.8 | |
| Self-doubt removed | **32** | 28 | 17 | **0.23** | 1.0 | 1 |
| Bullets removed | 5 | 4 | 0 | 1.31 | 5.8 | 0.70 (0.68) |
| Chinese removed | 3 | 1 | 1 | 1.54 | 7.0 | 0.47 (0.47) |
| Restating removed | 4 | 3 | 0 | 1.77 | 8.5 | 0.89 (0.73) |

Dose is mean |h·d|/|h| at L19 on unedited math text, from `research/otherdir_dose.py`, 8 traces. In brackets is Claude (Fable)'s re-measurement on 12 other traces. Restating's dose is the least stable.

| Bar | Result |
|---|---|
| N1. Self-doubt beats each | **pass**: +42.2, +45.3, +43.8, each lower bound 29.7 or more, 0 problems worse |
| N2. Others against unedited | +1.6 (−6.3 to 9.4), −1.6 (−7.8 to 4.7), 0.0 (−6.3 to 6.3). None helps |
| N3. Doubt cut is specific | **pass**: self-doubt 0.23, each other arm above the 0.95 line |
| N4. Intact | no run collapsed in any arm |

My prediction was that N1 passes and all three land at 2 to 8, which held.

**Why it's weak.** I chose these directions without checking the project's own record. [investigate/RESULTS.md](../investigate/RESULTS.md) shows that removing bullets and restating didn't change those behaviors even on the text they were built from. Bullets went the wrong way. Claude (Fable) measured the named behaviors in the study-25 traces.
- **Chinese.** No completion in any arm contains Chinese, so that direction has nothing to remove.
- **Bullets.** Bullet-line share went from 0.84 unedited to 0.75 with bullets removed, but to 0.675 with self-doubt removed.
- **Restating.** Restated 3-gram share went from 0.79 to 0.77, but to 0.66 with self-doubt removed.

So self-doubt's removal moves each control's own measure more than the control does. Chinese is also under the plan's "under half the dose" line for a weak control. The defensible reading is narrow: three other directions built from example pairs at L19, at 0.5 to 0.9 of self-doubt's dose, none with a measurable effect here, don't reproduce the gain. Study 27 tests a learned direction that does work on this text.

**Checks.** Claude (Fable) confirmed the three directions rebuild exactly from their example pairs by the same path as self-doubt (cosine 1.000). Those pairs are non-math, short answers with no reasoning. The prompts and answers are byte-identical across all five arms. The scorer checks run settings only for the three new arms. Study 20's traces carry no run settings, so their seed, rows and cap rest on `research/run_layers.sh` and its log. The scorer was written after the bullets arm finished, with its bars fixed by the plan. `research/` isn't in git yet, so the plan hashes aren't anchored to a commit.

## Studies 23 and 24: how much removal does it take, and which part does the work?

Bars in [PLAN_SPLIT.md](PLAN_SPLIT.md) (study 23, plan hash e72230fd9b528c47) and [PLAN_LEVEL.md](PLAN_LEVEL.md) (study 24, plan hash 75ed937dc17bfdc7), each written before its arms ran. **The gain is graded and steep near full removal: removing a third, two thirds and all of self-doubt's projection gets 7, 13 and 32 of 64, against 4 unedited. Two-thirds removal still beats study 22's random controls, so their smaller late dose doesn't explain their failure. Flattening the signal while holding it at a constant gets 19 to 21. Removing only a constant isn't shown to do anything.** So most of the gain comes from flattening how the signal rises and falls, not from a steady push. Whether full removal beats flattening alone is not settled, because this run's full arm (32) is high against its own replications.

All arms remove the self-doubt direction d at block 19's output, every token. The problems and settings match studies 20 to 22 (GSM8K test rows 800 to 863, seed 20, 3,000-token cap, thinking on). Study 20's unedited and full arms are reused. "Centered at c" is h − ((h·d) − c)d, which flattens the projection to the constant c. "Offset c" is h − c·d, which removes a constant c and leaves the variation.

Study 23 set c to the mean of block 19's output over general text, μ·d = 0.169. Before writing it up I measured the signed mean of h·d on 12 unedited math traces (`research/gsm_offset.py`, `runs/research/split_gsm_offset.json`). It is 0.49 overall, rising from 0.23 in the first 500 tokens to 0.60 after 2,000. So study 23's offset arm removed about a third of the real constant. Study 24 reran both arms with c = 0.49.

| Arm | Correct | Last box not empty | Finished and right | Doubt per 100 words | "wait" per 1,000 tokens |
|---|---:|---:|---:|---:|---:|
| Unedited (study 20) | 4 | 2 | 0 | 1.42 | 6.8 |
| Gain 0.35 | 7 | 4 | 3 | 0.88 | 4.1 |
| Gain 0.65 | 13 | 9 | 2 | 0.53 | 2.5 |
| Full removal (study 20) | **32** | 28 | 17 | 0.23 | 1.0 |
| Offset 0.169 | 2 | 2 | 0 | 1.37 | 6.0 |
| Offset 0.49 | 9 | 3 | 1 | 0.87 | 3.7 |
| Centered 0.169 | 21 | 19 | 8 | 0.33 | 1.4 |
| Centered 0.49 | 19 | 14 | 8 | 0.40 | 1.6 |

| Bar | Result |
|---|---|
| S1. Two-thirds beats study 22's pooled controls | **pass**: +13.7 (5.5 to 22.7) |
| S2. Strength curve | gain 0.35 minus unedited +4.7 (0.0 to 10.9), minus full −39.1. Gain 0.65 minus unedited +14.1 (4.7 to 23.4), minus full −29.7 (−42.2 to −18.8) |
| S3. Flattening beats unedited | **pass**: centered 0.169 +26.6 (15.6 to 37.5), 17 better, 0 worse |
| S4. Offset 0.169 against unedited | −3.1 (−9.4 to 3.1) |
| S5. Which part | centered 0.169 minus offset 0.169 +29.7 (18.8 to 40.6). But the offset arm was undersized, so this compares sizes as much as parts |
| L1. Offset 0.49 against unedited | +7.8 (0.0 to 15.6), 6 better, 1 worse. Not shown above unedited |
| L2. Offset 0.49 against gain 0.65 | −6.3 (−17.2 to 3.1) |
| L3. Centered 0.49 against full | −20.3 (−32.8 to −7.8) |
| L4. Centered 0.49 against centered 0.169 | −3.1 (−15.6 to 9.4) |
| S6, L5. Intact | no run collapsed in any arm |

My predictions were gain 0.65 near 20 (got 13), gain 0.35 near 10 (got 7), centered near full (got 21 against 32), and offset 0.169 near unedited (right). For study 24 I predicted offset 0.49 at 4 to 8 (got 9) and centered 0.49 near 25 (got 19).

**What it means.**

- The effect is a steep curve, not a dial. Two thirds of the removal buys about a third of the gain (9 of 28 problems). The last third buys the rest. Doubt words fall more smoothly (1.42, 0.88, 0.53, 0.23), so the drop in doubt words starts before the drop in failures.
- Two-thirds removal beats the random controls by 14 points. On late text three of study 22's controls removed 61 to 68% of self-doubt's dose. So the "specific to self-doubt" result from study 22 doesn't depend on that shortfall.
- Flattening carries most of it. Holding the projection at a constant, whether 0.169 or 0.49, gets 19 to 21 and cuts doubt words to a quarter. Removing a right-sized constant (0.49 every token, about as much per token as gain 0.65) gets 9, which isn't shown above unedited. So the edit works mainly by removing how the signal moves from token to token, not by pushing steadily along d. This is the reading earlier studies assumed. It now has a test behind it.
- Flattening looks like a weaker form of the same effect, not a separate part. 17 of centered 0.169's 21 credits and 16 of centered 0.49's 19 are problems full removal also solved.
- The gap between flattening and full removal is not settled. L3's −20 uses this run's full arm, which got 32. Its two earlier replications got 23 and 26 (81 of 192 pooled, about 27 of 64). Against 27, flattening's shortfall is 6 to 8 points, which 64 problems can't resolve. So "setting the projection to exactly zero matters" is suggestive at most, and so is any claim that the two parts add up.

**What this does not show.** At this cap "correct" means the model boxed an answer within 3,000 tokens. Claude (Fable) found one wrong boxed answer in 576 runs across all nine arms. Most credits are runs that were still going when cut (full 15 of 32, offset 0.49 8 of 9, gain 0.65 11 of 13). The edits change whether the model gets to an answer, not whether its answer is right. "Finished and right" is the stricter column. On it, offset 0.49 gets 1 and the centered arms get 8, against 17 for full. Centered at 0.49 holds the projection at 0.49 from the start, while the real mean early in a run is about 0.35. So for the first 1,000 or so tokens it adds a little self-doubt on average. One constant was used per arm, while the real mean drifts through a run. One seed, one model, one layer, 64 problems.

**Checks.** Claude (Fable) reviewed both studies.
- They confirmed `whyfix/generate.py` implements each arm as the plans state. d has norm 1. The constant is applied in float32 to bf16 states. On 3,600 real block-19 states the realized removal matched the intended one within 0.4%.
- They read the six offset 0.49 credits that have an empty last box (rows 803, 813, 819, 834, 855, 862). Each boxed the right number in its thinking, then echoed the instruction's empty box. So the rule isn't crediting wrong answers.
- Disclosure: study 23's centered, offset and gain 0.35 arms ran before `--level` was added to `generate.py`, and that version isn't in git. Old and new compute the same constant (0.1699 against 0.1693), but the traces carry no `level` key. `research/score_split.py` was extended for study 24 after the offset 0.49 traces existed. Its bars match PLAN_LEVEL.md as written. Study 20's arms carry no run settings, so their match is by behaviour.

## Study 22: is self-doubt special, or would removing any direction that size help?

Bars in [PLAN_DOSECTL.md](PLAN_DOSECTL.md) (plan hash 7006aa4b28ec2303, logged before any direction was built). **Removing self-doubt at L19 beats each of four random directions of about the same size, by 39 to 48 points. Not one control solved a problem that self-doubt missed. Removing a same-size random direction doesn't help at all. The gain isn't explained by removing a random direction that size.** On the late text where doubt runs, three of the four controls remove only about two-thirds as much as self-doubt. The fourth removes as much and still gets 7 of 64.

Every earlier control was one random direction that removed a third as much as self-doubt (study 21). The four new controls were drawn at random with the shape of the residual's variance at L19 (C^α z, with α between 0.63 and 0.88). Each had self-doubt and the residual mean projected out. Each was tuned so its mean |h·d|/|h| on the build text equals self-doubt's. The same 64 problems and settings as studies 20 and 21. Study 20's unedited and L19 arms are reused, and Claude (Fable) confirmed the prompts are byte-identical and the run flags match.

| Arm | Correct | Last box not empty | Finished and right | Doubt per 100 words | "wait" per 1,000 tokens |
|---|---:|---:|---:|---:|---:|
| Unedited | 4 | 2 | 0 | 1.42 | 6.8 |
| Self-doubt removed | **32** | 28 | 17 | **0.23** | 1.0 |
| Control 101 | 7 | 5 | 1 | 1.30 | 5.6 |
| Control 102 | 1 | 1 | 1 | 1.67 | 7.7 |
| Control 103 | 2 | 0 | 0 | 1.47 | 6.6 |
| Control 104 | 7 | 4 | 0 | 1.79 | 8.5 |

| Bar | Result |
|---|---|
| D1. Beats each control | **pass**: +39.1, +48.4, +46.9, +39.1, each lower bound 28 or more, 0 problems worse |
| D2. Beats the controls pooled | **pass**: +43.4 (32.4 to 54.7). Under the non-empty-box rule +39.8, finished and right +25.8 (15.2 to 36.7) |
| D3. Controls don't help | **holds**: pooled controls minus unedited +0.4 (−3.9 to 4.3) |
| D4. Doubt cut is specific | **pass**: self-doubt 0.23, every control above the 0.95 line |
| D5. Intact | no control run collapsed, nor any unedited run |

My predictions were D1, D2 and D4 pass, with controls at 3 to 10. All held, with controls at 1 to 7.

**What it means.**

- The gain is specific to the self-doubt direction compared with random directions of the same size, under every scoring rule. A large random removal at L19 leaves accuracy, doubt words and "wait" where they were.
- The dose match is weaker late in a run. The controls were matched on the build text, which is the prompt plus about the first 1,000 characters of thinking. Claude (Fable) measured dose on 10 runs per arm, using the first 768 tokens and the last 512 of each. On early text the controls remove 92 to 96% of self-doubt's dose. On late unedited text, controls 101 to 103 remove 61 to 68% of it, and control 104 removes 108%. Control 104 still got 7 of 64, and it had the most doubt words of any arm. So the late shortfall doesn't explain the gap. Control 104 also leans on dimension 0, a dimension with outsized activations, more than self-doubt does, so it's not a perfect stand-in.
- The mean isn't the explanation either, but this study didn't test that. Self-doubt overlaps the residual mean a little (cosine 0.045 to 0.064), and its projection is mostly one sign. The controls had the mean removed. The seed-1 control of study 20 had self-doubt's mean overlap and scored 4. Removing the mean outright breaks the model (whyfix/RESULTS.md C2). So same-mean-random does nothing, and same-dose-no-mean does nothing. Splitting self-doubt into its constant part and its varying part would settle it.
- At this cap, "correct" mostly means the model commits to an answer within 3,000 tokens. Unedited finished 0 of 64, and all 4 of its credits are boxes written mid-thought. The rule is the same for every arm, so it doesn't bias the comparison.

**What this does not show.** The test only rules out random directions. A non-random direction, such as one built from other behaviors, might also shorten thinking. Four controls, and they aren't fully independent draws (pairwise cosines up to 0.32 in size). One seed, one model, one layer, 64 problems. Removal strength wasn't varied, so how much of self-doubt's own gain survives at two-thirds of its dose is unknown.

**Checks.** Claude (Fable) reproduced every count from the traces with the project rule. They confirmed the plan hash, that the controls have no overlap with self-doubt or the mean, and that the generation path matches study 20's. The dose check is `research/dosectl_dose_check.py`. It takes its output path as an argument, and its output is in `runs/research/dosectl_dose_check.json`.

## Study 21: is it the depth or the direction?

Bars in [PLAN_TRANSFER.md](PLAN_TRANSFER.md) (plan hash d6d6a0a97e90d414, logged before launch). **Depth limits the gain. The original L19 vector, removed at L15, works almost as well as at L19. At L23 it gives a smaller gain, at a smaller dose. At L7 and L11 it removes about 80% as much of the residual as it does at L19, yet gains only +9 to +12 points, against +44 at L19, and it isn't shown to beat a random removal there.** Whether study 20's early directions were also worse vectors isn't settled: the head-to-head comparison had no power to tell.

The same 64 problems, seed and settings as study 20, whose unedited, L19 and per-layer arms are reused for comparison. Every row records the layer it ran at, and the scorer checks it.

| Arm | Correct | Last box not empty | Finished and right | Doubt per 100 words |
|---|---:|---:|---:|---:|
| Unedited (study 20) | 4 | 2 | 0 | 1.42 |
| L19 vector at L7 | 10 | 10 | 1 | 1.97 |
| L19 vector at L11 | 12 | 10 | 3 | 1.66 |
| L19 vector at L15 | 26 | 23 | 10 | 0.40 |
| L19 vector at L19 (study 20) | 32 | 28 | 17 | 0.23 |
| L19 vector at L23 | 16 | 12 | 5 | 0.24 |
| Random control at L7 / L11 / L15 / L23 | 4 / 6 / 6 / 7 | 3 / 5 / 3 / 5 | 2 / 2 / 0 / 1 | 1.59 / 1.43 / 1.65 / 1.67 |

| Bar | L7 | L11 | L15 | L23 |
|---|---|---|---|---|
| T1. Beats unedited | +9.4 (1.6 to 18.8), pass | +12.5 (3.1 to 21.9), pass | **+34.4 (21.9 to 46.9)** | +18.8 (7.8 to 29.7), pass |
| T2. Beats the control | +9.4 (0.0 to 18.8), fail | +9.4 (0.0 to 20.3), fail | **+31.3 (20.3 to 42.2)** | **+14.1 (4.7 to 23.4)** |
| T3. Against study 20's direction built at that layer | +6.3, none shown | +7.8, none shown | −6.3, none shown | −3.1, none shown, contaminated |
| T4. Against L19 | −34.4, lower | −31.3, lower | −9.4 (−21.9 to 3.1), none shown | −25.0, lower |
| T5. Intact | pass | pass | pass | pass |

My predictions were that L15 passes and L7 and L11 fail T1. L15 passed, but L7 and L11 passed T1 too, both narrowly.

**What it means.**

- Depth matters. Claude (Fable) measured on CPU, over 2,688 tokens of unedited text, how much of the residual the vector catches at each layer. As a share of the residual's length it is 6.6% at L7, 6.8% at L11, 7.8% at L15, 8.3% at L19 and 5.1% at L23. So at L7 and L11 the edit is nearly as large as at L19 and does a third as much. L23's weaker result comes with the smallest dose, so at L23 depth and dose are mixed together.
- L7 and L11 fail T2 at the boundary, so they're "not shown to beat the control", not "the same as random". Exact sign tests, exploratory: L7 8 better against 2 worse, p = 0.11. L11 9 against 3, p = 0.15. Under the non-empty-box rule, L7's T2 is +10.9 (1.6 to 20.3). The 10 problems the vector solves at L7 include 9 of L19's 32, but only 2 of the control's 4. That looks like a weak real effect. But 9 of the 10 early credits at L7 and 9 of 12 at L11 are boxes written mid-thought. Counting only finished runs, the early arms got 1 and 3 right, against the controls' 2 and 2.
- T3 can't settle whether study 20's early directions were worse vectors. With arms at 6 to 12 correct of 64, a 4 to 8 point gap is out of reach. They differ in kind, though. Study 20's own L7 direction raised "wait" by +11.9 per 1,000 tokens over the control (+6.1 to +19.8), and its own L11 direction by +9.0 (+3.5 to +16.7). The L19 vector at L7 and L11 raised it by only +1.4 and +1.3, with both CIs spanning 0. The control didn't raise it either. So the "wait" pathology at early layers belongs to the directions built there, not to early removal as such. The raw totals I first read (1,700 against 1,303) were driven by a few looping runs.
- T3 at L23 is contaminated. Study 20's L23 direction was built after the final norm. Under the non-empty-box rule, T3 at L23 is +6.3 (−4.7 to 17.2) in the L19 vector's favor.

**The control is one direction.** Every "matched random control" in studies 20 and 21 is the same random direction. Its pairwise cosines across all ten are 0.97 to 1.0, which I checked. `matched_control` uses seed 1 each time, and the self-doubt vector's overlap with every layer's mean is tiny (0.005 to 0.034), so the random draw dominates. It also catches only a third as much of the residual as the vector, so it isn't matched for dose. No verdict changes, because that control did nothing at every layer (1 to 7 correct). But "beats its control" means "beats one random direction that removes less", not "beats chance". Testing against several seeds and dose-matched random directions is the next fix.

**What this does not show.** One seed, one model, 64 problems, four layers. The generator was changed during these studies and isn't committed, so study 20's and study 21's arms came from different working copies. The smoke test covers the shared path.

**Checks.** Claude (Fable) reproduced every count and interval from the traces with the project rule. They confirmed that problem IDs, prompts and gold answers are identical across all 14 arms, and that every study 21 row ran at its stated layer. I reran the dose measurement and got the same numbers. It's saved as `research/measure_dose.py`, with output in `runs/research/xfer_dose.json`.

## Study 20: at which layers does removing self-doubt help?

Bars in [PLAN_LAYERS.md](PLAN_LAYERS.md) (plan hash logged in the decision trail before launch). **Directions built at L15 and L19 each pass every bar, by the same margin. The direction built at L23 passes as scored, but most of its gain comes from a scoring rule, and it was built at a slightly different point in the model than where it was removed. Directions built at L3, L7 and L11 don't help.** Each layer's direction comes from the same 4 pairs. So this tests direction and depth together, not depth alone.

GSM8K test rows 800 to 863, 64 new problems, seed 20, 3,000-token cap, no penalty. Controls were run only at layers passing Y1, as the plan said.

| Arm | Correct | Correct, last box not empty | Finished and right | Doubt per 100 words | Mean tokens |
|---|---:|---:|---:|---:|---:|
| Unedited | 4 | 2 | 0 | 1.42 | 3,000 |
| L3 | 7 | 4 | 1 | 2.24 | 2,975 |
| L7 | 6 | 5 | 1 | 4.06 | 2,975 |
| L11 | 7 | 4 | 1 | 3.32 | 2,986 |
| L15 | 30 | 29 | 18 | 0.27 | 2,677 |
| L19 | 32 | 28 | 17 | 0.23 | 2,596 |
| L23 | 18 | 8 | 7 | 0.24 | 2,909 |
| L15 control | 4 | 3 | 2 | 1.50 | 2,940 |
| L19 control | 4 | 2 | 1 | 1.38 | 2,967 |
| L23 control | 1 | 1 | 0 | 1.74 | 3,000 |

"Correct" is the project's rule: the last non-empty box. "Last box not empty" and "finished and right" are exploratory. The first was added after review: it doesn't credit a run whose final box is the empty `\boxed{}` the model sometimes echoes from the instructions.

| Bar | L3 | L7 | L11 | L15 | L19 | L23 |
|---|---|---|---|---|---|---|
| Y1. Beats unedited, CI above 0 | +4.7 (0.0 to 10.9), fail | +3.1, fail | +4.7, fail | **+40.6 (28.1 to 53.1)** | **+43.8 (31.2 to 56.2)** | **+21.9 (10.9 to 32.8)** |
| Y2. Beats its control | not run | not run | not run | **+40.6 (28.1 to 53.1)** | **+43.8 (31.2 to 56.2)** | **+26.6 (15.6 to 37.5)** |
| Y3. Doubt at most 2/3 of unedited | fail, higher | fail, higher | fail, higher | pass | pass | pass |
| Y4. Against L19 | lower | lower | lower | −3.1 (−14.1 to 7.8), no winner | | lower, −21.9 |
| Y5. Intact | pass | pass | pass | pass | pass | pass |

**What it means.**

- L15 and L19 do the same job. Each takes the model from 4 to about 30 right, cuts doubt words about sixfold, and beats a mean-matched random removal at its own layer by about 40 points under every scoring rule. They solved 21 of the same problems, 8 only at L15 and 7 only at L19.
- L23 is weak and uncertain. 10 of its 18 credits (corrected from 11 after study 21's review; the table gives 18 − 8) are runs whose last box is the empty echoed box: the run boxed an answer, then looped on "the user wants the final answer within \boxed{}". Not crediting those, it beats unedited by +9.4 (1.6 to 18.8) and its control by +10.9 (3.1 to 18.8). The library also returns the state after the final norm as the last layer's hidden state, so the L23 direction was built after that norm and removed before it. Fable found the mapped direction still has cosine 0.96 with the intended one, so this is a modest distortion, but it is a build error.
- Directions built at L3, L7 and L11 don't help, and runs with them have more doubt words, almost all "wait" (1,303 unedited, 3,717 at L7), and more loops (27 to 48 runs at L7). No early-layer control was run, so this doesn't show that removing these directions causes doubt. A random removal there might do the same.
- Early failure isn't explained by a small dose. Fable measured the removed share of the residual on unedited text: 13 to 14% at L3 to L11, against 7 to 9% at L15 to L23. But early directions remove only 1.1 to 1.4 times what their random controls remove, against 2.8 to 4.6 times at L15 to L23. So the pairs barely separate at early layers, which points to the direction, not the depth. Study 21 revises this. The L19 vector also does little at L7 and L11, at about 80% of its L19 dose, so depth alone limits the gain.
- L19 got 32 here, above the 23 and 26 of the earlier two replications. Pooled over all three sets, it's 81 of 192.

**What this does not show.** It doesn't separate depth from direction: removing the L19 direction at L15 or L7 wasn't tried. The six layers tested are all full-attention blocks (every 4th from L3). The model's linear-attention blocks weren't tested. One seed, one model, 64 problems, six layers. L15 and L19 survive any correction for six tests. L23's strict result, with a low of 1.6, doesn't.

**Checks.** Fable (Claude) reproduced every count and confirmed the code sets and hooks the right layer. A smoke test showed the new `--layer 19` path gives the same text as the old L19 arm on 8 problems. The traces didn't record which layer each arm ran, so the layer is shown by code and behavior, not by the files. `whyfix/generate.py` now writes its arguments into every row.

## Study 19: is study 18's 61 from the direction or from the penalty?

Bars in [PLAN_NOPEN.md](PLAN_NOPEN.md). **N1, N2 and N5 pass, N3 shows no interaction, and N4 favors self-doubt removal over MIT's Astra edit, all under the pre-registered scoring rule. Removing self-doubt helps without the penalty under every scoring rule. But without the penalty it often writes the right answer and then can't stop: 18 of its 46 credits come from runs that boxed the answer, never closed thinking, and repeated the box until the cap. Counting only finished runs, it ties Astra, and the direction and the penalty contribute about equally.**

One new arm, self-doubt removed at L19 with no penalty, on study 18's 96 problems (rows 704 to 799), seed 18, 9,000-token cap. The other arms are study 18's traces. It ran in one try with no restarts. Its comparator, the unedited arm, finished 14 minutes before it started.

| Arm | Correct at 3,000 | Correct at 9,000 | Finished by 9,000 | Finished and right | Right but capped | Mean tokens | Loops |
|---|---:|---:|---:|---:|---:|---:|---:|
| Unedited | 5 | 8 | 2 | 2 | 6 | 8,845 | 93 |
| Unedited, penalty | 5 | 16 | 13 | 13 | 3 | 8,411 | 71 |
| Self-doubt removed | 35 | 46 | 30 | 28 | 18 | 7,035 | 63 |
| Self-doubt removed, penalty | 33 | 61 | 72 | 60 | 1 | 5,298 | 16 |
| Astra edit, penalty | 14 | 31 | 33 | 31 | 0 | 7,398 | 6 |

"Correct" uses the project's value-based rule: the last box in the text, wherever it is, as in every earlier study. "Finished and right" also needs the run to close thinking before the cap. That column and the "strict" figures below are exploratory and weren't pre-registered ([explore_nopen.py](explore_nopen.py)).

| Bar | Needed | Result | Verdict |
|---|---|---|---|
| N1. Direction without the penalty | removed minus unedited, CI above 0 | +39.6, CI 29.2 to 50.0 | **Pass** |
| N2. Penalty on top of the direction | with penalty minus without, CI above 0 | +15.6, CI 6.3 to 25.0 | **Pass** |
| N3. Interaction | penalty's gain with the direction removed minus its gain unedited, CI excludes 0 | +7.3, CI −5.2 to 19.8 | No interaction shown |
| N4. Our setting against theirs | removed, no penalty, minus Astra with penalty | +15.6, CI 6.3 to 26.0 | **Favors self-doubt removal** |
| N5. Intact | at most 2 more runs with 200 spaces in a row, or collapsed, than unedited | 0 and 0 | **Pass** |

**What it means.**

- The direction works on its own. Without the penalty it reached 46 against 8 by the pre-registered rule, and 28 against 2 counting only finished runs (+27.1, CI 17.7 to 36.5). It cut doubt words from 14.4 to 0.77 per 1,000 tokens (counted by Fable).
- What the penalty adds is mostly the ability to stop. Without it, 63 of 96 runs loop, and 18 runs box the right answer but then repeat "Final Answer: \boxed{50}" and similar lines until the cap, never closing thinking. With it, 16 loop and 1 is right but capped. Finished runs rose from 30 to 72.
- That stopping isn't free. Finished but wrong went from 2 of 30 to 12 of 72, so accuracy among finished runs fell from 93% to 83%.
- How the credit splits depends on the scoring rule. By the pre-registered rule, the arm without the penalty has 46 of the 61 (75%), and N3 shows no interaction. By the strict rule it has 28 of 60 (47%). The direction adds about +26 and the penalty about +32, and they compound: the strict interaction is +21.9, CI 10.4 to 33.3, exploratory.
- N4's win over Astra exists only under the pre-registered rule. Strictly, it's 28 against 31 (−3.1, CI −13.5 to 7.3), with no winner shown. By the pre-registered rule, both arms solved 26 problems, 20 only ours and 5 only Astra's, so study 18's finding that "every problem theirs solved ours also solved" needs the penalty.

**What this does not show.** It doesn't show why the penalty lets the model stop. A likely reason is that the penalty pushes down tokens the model has already written, including a repeated answer line, but this study didn't test that. One seed, one model, 96 problems, four bars with no correction. Same seed only pairs runs by problem. The edits change the logits from the first token, so the samples differ anyway.

**Checks.** Fable (Claude) reproduced every number from the traces. It found the IDs, prompts and gold answers byte-identical across arms. The runner passes no penalty, and the arm's behavior matches: its distinct-token share in the first 2,000 tokens is 0.131, against 0.121 unedited and 0.18 to 0.20 for the penalty arms. All 18 capped credits lack `</think>`, 14 were already right at 3,000 tokens, and the repeated boxes agree in value. The per-batch pattern of correct answers appears in every arm, so it reflects problem difficulty, not drift.

**Deviation.** `research/` isn't tracked by git, so the only record that the plan came before the data is its file time, 05:47, the same minute the run started. The plan was written before launch in this session's transcript.

## Study 18: our edit against MIT's, on the same problems

Bars in [PLAN_HEAD.md](PLAN_HEAD.md), with its amendment. **Every bar passes: H1, H2, H3, H5, H7 and H8, and H4 favors self-doubt removal.** Under MIT's sampling and at a 9,000-token cap, removing the self-doubt direction got 61 of 96 right, against 31 for MIT's Astra weight edit. Every problem their edit got right, ours also got right. This is one cap, one seed and one set of 96 problems, and it says nothing about their result at 32,768 tokens.

GSM8K test rows 704 to 799, 96 new problems, seed 18, 9,000-token cap, L19, value-based scoring. Every arm with "penalty" uses MIT's presence penalty of 1.5. Each run is also scored as if cut at 3,000 by truncating its text.

| Arm | Correct at 3,000 | Correct at 9,000 | Finished by 3,000 | Finished by 9,000 | Right when finished | Mean tokens | Loops |
|---|---:|---:|---:|---:|---:|---:|---:|
| Unedited | 5 | 8 | 2 | 2 | 2 of 2 | 8,845 | 93 |
| Unedited, penalty | 5 | 16 | 2 | 13 | 13 of 13 | 8,411 | 71 |
| Astra edit, penalty | 14 | 31 | 13 | 33 | 31 of 33 | 7,398 | 6 |
| Self-doubt removed, penalty | 33 | 61 | 29 | 72 | 60 of 72 | 5,298 | 16 |
| Both, penalty | 43 | 63 | 47 | 81 | 63 of 81 | 4,295 | 0 |
| Matched random, penalty | 3 | 14 | 2 | 12 | 12 of 12 | 8,493 | 64 |

"Correct at 3,000" can exceed "finished by 3,000" because it credits a right box written inside thinking that hadn't closed by 3,000 tokens.

| Bar | Result | Verdict |
|---|---|---|
| H1. Their fix reproduces: Astra minus unedited, penalty, at 9,000 | +15.6, CI +8.3 to +22.9 | pass |
| H2. Self-doubt adds to their fix: both minus Astra, at 9,000 | +33.3, CI +24.0 to +42.7 | pass, and above the +10 useful size |
| H3. Faster: self-doubt minus Astra, at 3,000 | +19.8, CI +12.5 to +28.1 | pass |
| H4. Who wins given 9,000 tokens: self-doubt minus Astra | +31.3, CI +21.9 to +40.6 | self-doubt |
| H5. Our lead holds under their sampling: self-doubt minus unedited, penalty, at 3,000 | +29.2, CI +19.8 to +38.5 | pass |
| H6. Penalty alone at 9,000, no bar | +8.3, CI +2.1 to +15.6 | estimate |
| H7. Intact: a blind Claude reader finds at most 2 of 20 excerpts garbled in each arm | 0 garbled in every arm | pass |
| H8. Control: self-doubt minus matched random, at 3,000 | +31.3, CI +22.9 to +40.6 | pass |

**What it means.**

- **Their edit reproduces here at the size their own logs predict.** Rescored at 9,000 tokens, their logs give 32.5% correct and 39.5% finished. This study gives 32.3% and 34.4%. So the setup can show their result, and the comparison is fair at this cap.
- **At 9,000 tokens, removing self-doubt beats their edit by about 31 points, and it isn't only speed.** It leads at 3,000 (H3) and still leads at 9,000 (H4). The overlap is one-sided: 31 problems both solve, 30 only ours solves, none only theirs.
- **The gain comes from finishing, with some loss of accuracy.** Self-doubt removal finished 72 runs against 33, but was right in 83% of those against 94% for their edit. It commits to more answers, and a larger share of them are wrong. That gap is also one way their edit could catch up with more tokens.
- **The two edits add up.** Applying both gets 63, two more than self-doubt removal alone. It finishes more and never loops, but its accuracy when finished drops again, to 78%. So most of what their edit does at this cap, self-doubt removal already does.
- **The penalty matters, and this design can't say how much.** The penalty alone doubles the unedited model, from 8 to 16. There's no self-doubt arm without the penalty here. In study 17, on different problems and without the penalty, self-doubt removal got 44% right at 9,000 and finished 33%. Here it got 64% and finished 75%. Either these problems are easier, or the penalty and self-doubt removal work together. Only "self-doubt removal given the penalty" is measured.

**What this does not show.**

- Anything about MIT's full result. At 9,000 tokens their edit shows about 43% of its final gain, and their median run is 11,200 tokens. Their higher accuracy when finished suggests they may close the gap with more tokens.
- That the two edits differ as methods. This compares one permanent weight edit against one run-time removal.
- That their edit stays intact under this penalty. In 6 to 10 of their 63 runs that hit the cap, depending on threshold, the text fills with ever-longer runs of spaces, up to hundreds of thousands of characters. No other arm does this except 2 runs each of the unedited and random-control arms. The loop counter and the reader both split text into words, so neither saw it. A likely cause is that the penalty taxes each indentation token once it's used, so the model reaches for a longer one. None of these runs was correct, so no score changes. It may not happen under vLLM.

**Deviations.**

- An outside kill at 03:49, probably a GPU reset, stopped the Astra arm at 48 of 96. I added `--resume` to `whyfix/generate.py` and a retry loop to `run_head.sh`. Resume keeps whole batches of 8 and reseeds the rest with seed plus rows done. Batch 6 of that arm stalled for about 6 hours across 5 attempts before completing at 09:49. Its result, 2 of 8, is in range of the other batches.
- I added a progress printer to the generator mid-arm. It returns scores unchanged.
- The versions of `run_head.sh` and `generate.py` from before the restarts weren't committed, so the code that made Astra's first 48 rows can't be shown. Both halves of that arm behave alike: 15 and 16 correct, 17 and 16 finished.
- The planned Sonnet reader was stopped by a safety classifier before it wrote anything, so Haiku did the labels.

**Checked.** A Fable review reproduced every count and CI from the traces with its own scorer, with no row-level disagreement. Its other checks:

- **Scoring.** It hand-checked 14 correct runs and every boxed-but-wrong run.
- **Integrity.** It confirmed the IDs and gold answers are the same in every arm, and that the blind key regenerates exactly.
- **Readability labels.** It spot-checked 10 of the Haiku labels.
- **Presence penalty.** It confirmed against current vLLM source that the penalty applies to output tokens only, before temperature, as the plan assumed. MIT's vLLM version is still unchecked.
- **Edits applied.** The traces don't record which edit each arm ran, so the review checked this from behavior. The arms behave as expected: the matched random direction does almost nothing, Astra matches MIT's own logs, and the combined arm differs from both of its parts.
- **Its own findings.** It found the whitespace failure and the penalty confound.

Two more numbers come from it. First, counting only runs that finished under 3,000, H3's arms are 27 and 13. Second, self-doubt removal gets 6 credits for boxes written inside thinking at 3,000, against 1 for Astra. Both are disclosures, and no bar changes. I confirmed the plain last-box rescoring, the one-sided overlap and the whitespace runs myself.

Limits: one seed, one model, 96 problems, no correction across bars, no factorial design for the penalty, and our penalty may differ in detail from the vLLM version MIT used.

Scripts: `run_head.sh`, `score_head.py`, `excerpts_head.py`. Data: `runs/research/head_results.json`, `head_reader.json`, traces `head_{none,none_pp,astra_pp,doubt_pp,astra_doubt_pp,matched_pp}_traces.jsonl`.

## Study 17: given 9,000 tokens, does removing both directions get more right?

Bars in [PLAN_FINISH.md](PLAN_FINISH.md), with its amendment. **E2, E3, E4, E5 and E7 pass, and E1 and E6 fail.** Given 9,000 tokens, removing both directions got 42 of 64 right, against 28 for self-doubt removed and 31 for repetition removed. Study 14's finish gap at 3,000 tokens did not replicate. E7, the pair against repetition alone, is the thinnest pass and depends on how "correct" is counted.

GSM8K test rows 640 to 703, 64 new problems, seed 17, 9,000-token cap, L19, value-based scoring. Each run is also scored as if cut at 3,000 by truncating its text.

| Arm | Correct at 3,000 | Correct at 9,000 | Of which cut at the cap | Finished by 3,000 | Finished by 9,000 | Broken |
|---|---:|---:|---:|---:|---:|---:|
| Repetition removed | 17 | 31 | 1 | 10 | 36 | 0 |
| Self-doubt removed | 22 | 28 | 11 | 9 | 21 | 0 |
| Both removed | 18 | 42 | 3 | 11 | 47 | 1 |

| Bar | Result | Verdict |
|---|---|---|
| E1. Finished by 3,000, both minus self-doubt | +3.1, CI -7.8 to +14.1 | fail |
| E2. Correct at 9,000, both minus self-doubt | +21.9, CI +7.8 to +35.9 | pass |
| E3. Both, correct at 9,000 minus at 3,000 | +37.5, CI +26.6 to +50.0 | pass, but a within-arm bar that was nearly certain to be positive |
| E4. Both finishes at least 32 | 47 | pass |
| E5. Broken, both at most self-doubt plus 2 | 1 against 0 | pass |
| E6. Finished by 3,000, repetition minus self-doubt | +1.6, CI -7.8 to +12.5 | fail |
| E7. Correct at 9,000, both minus repetition | +17.2, CI +4.7 to +29.7 | pass |

**What it means.**

- **Given time, the pair is the better edit.** It beats self-doubt alone by 22 points and repetition alone by 17. At 3,000 tokens the three arms are close (17, 22, 18), which matches study 14's tie. The extra tokens turn into answers mainly when both directions are removed.
- **Counting only finished answers, E2 grows and E7 shrinks to the edge.** 11 of self-doubt's 28 correct runs wrote the right box inside thinking that never closed, then kept going to the cap (study 2's pattern). The plan counts those, so the verdicts stand. Counting only runs that finished with a right answer gives both 39, repetition 30, self-doubt 17. On that count E2 is +34.4 (CI +20.3 to +48.4) and E7 is +14.1 (CI 0.0 to +28.1). E7 is also the thinnest bar at 99% (low end +1.6). So "the pair beats repetition alone" is supported only under the pre-registered count, and only just.
- **Self-doubt removal does grow with time, just less.** It goes from 22 to 28 (+9.4, CI +3.1 to +17.2). The pair grows 28 points more than self-doubt does (CI +14 to +42). Repetition's growth is not separated from self-doubt's (difference +12.5, CI 0.0 to +25.0), and the two single edits tie at 9,000 (+4.7, CI -4.7 to +15.6). My first draft said self-doubt's removal "saturates". That was wrong.
- **The "different jobs" reading is the plan's, and a bigger edit explains it as well.** The two directions have cosine 0.42, so each single edit removes a large part of the other. The pair removes the plane they span. No plane-sized control ran at 9,000, so "two jobs" and "a larger edit" aren't told apart here.
- **The finish gap at 3,000 didn't replicate, partly from a floor.** Only 9 to 11 of 64 finished by 3,000 in any arm here, against 19 and 22 for repetition and both in study 14. E1's CI excludes study 14's +18.8, so the failure is fair. By 9,000 the pair finished 47, against 21 for self-doubt and 36 for repetition (no bar).

**Deviation.** An outside SIGTERM, not from this loop, killed the pair's arm after 56 of 64 problems. I kept those 56 and regenerated rows 696 to 703 alone with seed 17 and appended them. That last batch had the same 8 problems and the same padding as in the original run, and only the random stream differs. The tail went against the pair: 4 of 8 correct, 0 finished by 3,000, and it holds the arm's one broken run. On the first 56 alone, E2 is +26.8 and E7 +21.4, so the 64-problem numbers above are the conservative ones.

**Checked.** A Fable review reproduced every count and CI from the traces. It found truncation by retokenizing safe: no run's finished-by-3,000 status changes, and no cut run had a stray wrong box. It also caught the overreach in my first reading, which the bullets above now correct. I checked the strict-count numbers and the 0.42 cosine myself.

Limits: one seed, one model, 64 problems, 7 bars with no correction. No unedited arm. Study 1 found the unedited model at 2 of 32 from 3,000 to 10,000 tokens.

Scripts: `run_finish.sh`, `score_finish.py`. Data: `runs/research/finish_results.json`, traces `finish_{rep,doubt,both}_traces.jsonl`, and the split `finish_both_traces_first56.jsonl` plus `finish_both_tail.jsonl`.

## Study 16: is one tilted axis enough to free a stuck run?

Bars in [PLAN_TILT.md](PLAN_TILT.md), second amendment. **T1, T2 and T3 fail, and T4, T5, T6 and T7 pass. The axis I froze at 42 degrees is not confirmed.** Repetition-only at equal push beat it by 14.7 points. Pushing in this plane does free runs, and the sign matters. After the results, a one-axis fit moves to about 33 degrees and fits these arms better, but only because it used this data. Fitted without this data it recovers about a third of T3's gap, so T3 is unexplained and not resolved.

102 stuck runs, cut at loop onset, none edited in any rescue before. Seed 16, 1,500 new tokens.

| Arm | Exits | Finished | Correct, cut runs included | Broken | Next-token KL dose |
|---|---:|---:|---:|---:|---:|
| Unedited | 2 | 0 | 0 | 74 | |
| Repetition-only, gain 1 | 44 | 1 | 1 | 33 | 0.0192 |
| Repetition-only, equal push, gain 1.11 | 59 | 3 | 3 | 27 | 0.0261 |
| Tilted 42 degrees | 44 | 6 | 6 | 43 | 0.0107 |
| Both removed | 56 | 3 | 4 | 35 | 0.0167 |
| Perpendicular, equal push | 4 | 0 | 0 | 68 | 0.0073 |
| Random, constant shift | 2 | 0 | 0 | 75 | 0.0022 |

| Bar | Needed | Result | Verdict |
|---|---|---|---|
| T1. Signed | tilted minus repetition-only, CI above 0 | 0.0 points (CI -10.8 to +10.8) | **Fail**, and not the reverse either |
| T2. Tie at equal push | tilted minus equal-push repetition-only within 5, CI top below +15 | -14.7 (CI -25.5 to -3.9) | **Fail**, repetition-only ahead |
| T3. Plane adds nothing | both minus tilted at most +5, CI top below +15 | +11.8 (CI +2.0 to +21.6) | **Fail** |
| T4. Study 11 replicates | both minus repetition-only, CI above 0 | +11.8 (CI +1.0 to +22.5) | Pass |
| T5. Direction in the plane | tilted minus perpendicular at least +20, CI above 0, perpendicular at most 8 | +39.2 (CI +28.4 to +50.0), 4 of 102 | Pass |
| T6. Not displacement | tilted minus random at least +20, CI above 0 | +41.2 (CI +31.4 to +51.0) | Pass |
| T7. Intact | at most 3 more broken than unedited | all at or below unedited's 74, except random at 75 | Pass, but see below |

**What the plan's readings say.** T2 failing with repetition-only ahead reads as an axis nearer repetition-only than 42 degrees. T3 failing reads as "a second direction matters". These two conflict, and the rest of this section is why I don't take the second at face value yet. The plan predicted 37, 48, 48 and 1 exits for repetition-only, tilted, both and perpendicular, against 44, 44, 56 and 4 observed. An axis fitted on the earlier studies predicts equal-push minus tilted and repetition-only minus tilted of 0 and 0 at 42 degrees, and +23.6 and +12.6 at repetition's own 24.7 degrees. The plan's "about 12" was the second of those, wrongly attached to the equal-push arm. Observed were +14.7 (CI +3.9 to +25.5) and 0.0 (CI -10.8 to +10.8). So T2's CI excludes the 42-degree prediction and T1's excludes the 24.7-degree one, which puts the axis between them without any refit. Exits per source are in `runs/research/tilt_results.json` and show nothing odd. Unedited exited 2 of 102, near 0 as expected.

**Post hoc refit, no bars.** `research/fit_axis_post.py` fits one axis to these six in-plane arms and to the 11 from studies 7, 8 and 11. It gives 33 degrees, with a bootstrap over prefixes of 26.2 to 38.7. That excludes the frozen 42, and the original repetition direction's own 24.7 by a small margin when pooled, though 24.7 stays inside study 16's own band. Fitted on study 16 alone it gives 32, with a likelihood band of 24 to 40, and that band ignores the pairing of arms by prefix. Predicted against observed exits at 33 degrees: repetition-only 44.5 and 44, equal-push 55.1 and 59, tilted 41.6 and 44, both 52.0 and 56, perpendicular 2.8 and 4. **Pooling flatters those numbers.** A fit on studies 7, 8 and 11 alone, whose best angle is 39, predicts this study as repetition-only 38.7, equal-push 49.5, tilted 44.4, both 48.2 and perpendicular 1.4. That is a summed error of 26.5 exits against 12.4 for the pooled fit, and it gives both minus tilted as +3.8 against the observed +11.8. So held out, one axis recovers about a third of T3's gap, and 39 degrees is inside study 16's own band. **T3 is inconclusive.** Its +11.8 (CI +2.0 to +21.6) is consistent with +3.8 from one axis fitted earlier, with +10.4 from one axis refitted, and with a real second direction. A logistic linear in both pushes is itself a one-axis model, so this fit could only contradict one axis through an interaction term. That term adds 0.4 to the log likelihood with a bootstrap CI of -0.13 to +0.29, which is about ±2 logits at typical pushes and uninformative. The fit also overpredicts repetition-only in studies 8 and 11, 24.3 against 19 and 29.1 against 22, and underpredicts `both`, 28.7 against 33.

**Self-doubt's part.** Perpendicular pushes repetition-only about as far as tilted does, 2.38 against 2.63, and raises self-doubt where tilted lowers it, and it frees 4 runs against 44. So the sign of the self-doubt change matters. The caveat from the plan stands: study 9 found adding self-doubt induces loops, so this may be harm from adding self-doubt and not help from removing it. A fit with separate slopes for lowering and raising self-doubt didn't beat two plain slopes (log likelihood gain 0.9), so these data can't separate those.

**How far "exit" goes.** A blind Claude reader (Sonnet) judged 86 excerpts taken at the point of exit, all six exits from the perpendicular and random arms and 20 sampled from each of the others. It labelled 0 garbled, 45 coherent and 41 repetitive. Coherent per sampled arm: repetition-only 10 of 20, equal-push 12, tilted 11, both 9, perpendicular 2 of 4, random 1 of 2. So the exit rule isn't counting garbled text, and that audit passes. But about half of what it counts as an exit still reads as restating a template, so an exit doesn't mean real recovery. This isn't a strict upper bound, because the reader never saw non-exits and the rule may miss some recoveries. With 20 per arm and one reader, arms can't be compared on this.

**T7 had almost no teeth.** The broken rule counts low distinct-word text, and an unedited stuck run is that by nature, so the limit was 77 of 102. Differences in the broken column are mostly exits restated: among exits, tilted and equal-push have 3 broken each. The reader's zero garbled is the evidence of intact text. An intactness bar for stuck runs should leave out runs still looping.

**Random and dose.** T6 rules out equal mean displacement, but the constant shift has a KL dose about a fifth of tilted's, so it does not rule out an edit of equal effect on the next-token distribution. KL doesn't order the arms by itself either: tilted at 0.0107 frees 44 and repetition-only at 0.0192 also frees 44. These KLs come from 8 prefixes.

**Finishing is a different question.** Exits rarely became answers. Finished runs were 0 to 6 of 102 and correct 0 to 6. Tilted's 6 correct against equal-push's 3 is +2.9 points, CI -2.0 to +7.8, no difference shown, and it had no bar. These continuations get only 1,500 tokens.

**Limits.** One seed and one model, and no correction across bars. Excerpts were judged by one Claude reader. T2 shows the 42-degree angle is wrong, and T3's bar was set at that angle, so whether one axis is enough stays open. A frozen angle on new stuck runs is what would test that.

Files: `research/build_tilt.py`, `research/run_tilt.sh`, `research/score_tilt.py`, `research/audit_tilt.py`, `research/fit_axis_post.py`, `runs/research/tilt_results.json`, `runs/research/tilt_audit.json`, `runs/research/tilt_fit_post.txt`, `runs/research/tilt_reader.json`.

## Study 15: does the before-the-answer lead replicate on MobileLLM-R1-360M?

Bars in [PLAN_REP_PORT2.md](PLAN_REP_PORT2.md). **R1 and R4 pass, and R2 and R3 fail. The port is not shown against unedited.** Removal beat the matched control on first-box answers, missed against unedited by a CI low of -0.8, and didn't reduce repeats before the answer against the control.

128 unused problems, rows 512 to 639, seed 12, 1,500 tokens, L12, plain text prompts, directions frozen from study 13. The base checkpoint only. The first box means the first containing a digit. Nearly every run hit the cap, 126, 127 and 122 of 128.

| Arm | First box correct | Last box correct | Has a box with a digit | Repeats before first box | Broken before box | Broken, whole run |
|---|---:|---:|---:|---:|---:|---:|
| Unedited | 31 | 33 | 102 | 18.5 | 5 | 10 |
| Repetition removed | 41 | 39 | 113 | 14.6 | 5 | 14 |
| Matched random removed | 26 | 26 | 108 | 15.2 | 4 | 7 |

| Bar | Needed | Result | Verdict |
|---|---|---|---|
| R1. Against control, answers | at least +5, CI above 0 | +11.7 points (CI +3.1 to +20.3) | Pass |
| R2. Against unedited, answers | at least +5, CI above 0 | +7.8 points (CI -0.8 to +16.4) | **Fail** |
| R3. Against control, repeats | at most -5, CI below 0 | -0.6 points (CI -8.5 to +7.0) | **Fail** |
| R4. Intact | at most 3 more broken than unedited | 5 against 5 before the first box | Pass |

The plan's reading for R1 passing and R2 failing was "the matched edit does harm and removal doesn't help". The data don't say that cleanly. Matched minus unedited is -3.9 points, CI -11.7 to +3.9, so harm from matched isn't shown either. What holds is narrower: removal differs from its control and can't be told apart from no edit.

The plan says all measures use the text before the first box, and my scorer first applied R4 to the whole run, which gave 14 against 10 and a fail. A review caught that, and I fixed the scorer. On whole runs, R4 would fail by one.

**R3 fails, so the fall in repeats against matched didn't replicate.** With about half the runs at zero and about 170 words before the box, this measure can't show a mechanism either way. Study 13's gap came mostly from matched being high there.

**After the box, the edit does the opposite of its name.** Whole-run repeats are 80.4 unedited and 90.2 removed, +9.7, CI +3.8 to +15.6, and the median run writes about 73 boxes with a digit against 29. Removing this direction makes this model commit and then restate. So the direction built from repetition examples isn't simply a repetition control here.

No bars on the rest:

- On the 115 problems where no arm is broken before the box, removal minus unedited is +7.8, CI -1.7 to +17.4, and removal minus matched is +12.2, CI +2.6 to +21.7.
- Pooling studies 13 and 15, 192 problems, first-box correct is 45 unedited, 61 removed, 36 matched, removal minus unedited +8.3, CI +1.6 to +15.1. That isn't independent evidence. Study 13's part is where the hypothesis came from, and the replication on its own failed R2.
- This model boxes in about 80% of runs and is right in about 30% of them. Unedited Qwen boxes in under 15% of runs and is right in most of those. So the edit here acts on a model that commits freely and is often wrong, where on Qwen it acts on one that rarely commits.

The control bar has now passed twice, +15.6 and +11.7, and the bar against unedited has missed narrowly twice, +9.4 and +7.8. A third 128-problem run would settle R2. I'm stopping for cost and not because the data closed the question: the Qwen questions teach more per GPU hour.

Scripts: `run_rep_port2.sh`, `score_rep_port2.py`, `audit_rep_port2.py`. Data: `runs/research/rport2_results.json`, `rport2_audit.json`.

## Study 14: is removing both directions from the start better than either alone?

Bars in [PLAN_BOTH.md](PLAN_BOTH.md). **B1 and B2 fail, and B3 and B4 pass. Removing both is not better than removing self-doubt on correct answers: 28 of 64 each.** By the plan's reading, adding repetition's removal doesn't improve on self-doubt's, and study 11's finish count may be noise or specific to stuck runs.

64 unused problems, rows 448 to 511, seed 11, 3,000-token cap, value-based scoring.

| Arm | Correct | Boxed | Right when boxed | Finished | Entered a loop | Repeats | Broken |
|---|---:|---:|---:|---:|---:|---:|---:|
| Unedited | 7 | 8 | 7 of 8 | 2 | 31 | 49% | 0 |
| Self-doubt removed | 28 | 32 | 28 of 32 | 10 | 10 | 25% | 0 |
| Repetition removed | 22 | 25 | 22 of 25 | 19 | 0 | 7% | 0 |
| Both removed | 28 | 30 | 28 of 30 | 22 | 0 | 5% | 0 |
| Two random removed | 10 | 11 | 10 of 11 | 1 | 29 | 46% | 0 |

| Bar | Needed | Result | Verdict |
|---|---|---|---|
| B1. Beats self-doubt | at least +5, CI above 0 | 0 points (CI -12.5 to +12.5), 8 better and 8 worse | **Fail** |
| B2. Beats repetition | at least +5, CI above 0 | +9.4 points (CI -1.6 to +20.3) | **Fail** |
| B3. Beats its control | at least +10, CI above 0 | +28.1 points (CI +17.2 to +40.6) | Pass |
| B4. Intact | at most 2 more broken than unedited | 0 broken | Pass |

The rest of this section had no bars and was chosen after looking.

- **Self-doubt's effect replicated again.** Self-doubt minus unedited was +32.8 points, CI +21.9 to +43.8, on new problems. Repetition minus unedited was +23.4, CI +12.5 to +34.4.
- **The edits changed how often a box appears, not how often a box is right.** Every arm is right on about 88% to 93% of its boxes, including unedited, though unedited has only 8. All five arms also tie within small counts, so this is a description and not a test.
- **Finishing and loops follow repetition's removal, not the pair.** Repetition alone finished 19 and both 22, +4.7 points, CI -6.2 to +15.6, so the pair-over-repetition finish gain seen in study 11 did not carry. Repetition alone beats self-doubt on finishing, +14.1, CI 0 to +28.1, and both reach 0 loops. I first credited this to the pair and a review corrected it.
- **So the pair combines two separate effects and adds nothing new I can show.** It reaches self-doubt's 28 correct with about repetition's finishing. Of self-doubt's 28 correct runs, 19 hit the cap, with the box followed by 52 to 1,422 more words. Of the pair's 28, 21 finished, and repetition alone finished 17 of its 22. 20 correct problems are shared between the pair and self-doubt, with 8 on each side.
- **Runs that hit the cap with both removed aren't in a verbatim loop, but they don't settle.** Of the 42, 34 never box, 7 box the right value inside thinking and keep writing, and 1 boxes a wrong value. 4 of the 42 had closed their thinking and were cut mid-answer, so "hit the cap" overstates non-finishing a little. A Fable review read them more widely than I did: having computed a candidate, they enumerate readings of the wording ("Interpretation A/B", "Is there a trick?"), re-derive the same number under each, and don't close thinking. 25 of the 34 unboxed contain the right value, 24 of those as a computed "= value". Doubt words by the project's list, in the last 500 words of capped runs, are 2.5 unedited, 0.3 self-doubt, 0.9 repetition, 0.5 both, 2.3 random, so that wording falls while the indecision continues. That reading comes from reading samples and has no test yet.
- **Doubt words in the first 500 words**, as the plan said would be reported, per 100 words: 0.68 unedited, 0.20 self-doubt, 0.50 repetition, 0.32 both, 0.78 random. The pair removes less of this wording than self-doubt alone.

Limits: one seed, one model, 64 problems, no correction across bars. Removing a plane is a bigger edit than removing one direction, and this says nothing about its cost on other text. Study 17 tests whether time helps these arms, with repetition alone included.

Scripts: `run_both.sh`, `score_both.py`, `audit_both.py`. Data: `runs/research/both_results.json`, `both_audit.json`.

## Study 13: does the repetition method carry to MobileLLM-R1-360M?

[Plan](PLAN_REP_PORT.md). 64 new problems, seed 10, 1,500 tokens, L12, plain text prompts. A review found before I read any result that this model answers in about 200 words and then repeats itself, so the plan was amended to rest its reading on text before each run's first box. The name of the pairs was also wrong in the build script, which stopped the run after its unedited arm, and it was fixed and resumed.

| Arm | Repeats, whole run | Runs entering a loop | Repeats before first box | First box correct | Last box correct | Broken |
|---|---:|---:|---:|---:|---:|---:|
| Unedited | 85.0 | 45 | 17.2 | 14 | 12 | 5 |
| Repetition removed | 88.1 | 47 | 12.7 | 20 | 18 | 5 |
| Matched random removed | 80.9 | 44 | 22.9 | 10 | 11 | 2 |

| Bar | Result | Verdict |
|---|---|---|
| Original gate | repeats 85%, 45 loops | pass, but it mostly measures after the answer |
| Amended gate, at least 20% repeats before the first box | 17.2% | **fail** |
| Q1. Repeats fall | 88.1 against 85.0, +7.3 from matched | **fail** |
| Q2. Loops fall | +4.7 from matched, CI -10.9 to +20.3 | **fail** |
| Q3. Answers, last box, at least +5, CI above 0 | +9.4, CI -1.6 to +20.3 | **fail** |
| Q4. Control is quiet | CI includes 0 | pass |
| Q5. Intact | 5 and 5 | pass |

**The plan's reading: this model doesn't loop enough while reasoning to test this, and whole-run bars only measure repeating after the answer. The method did not pass on this model.**

**A lead, from after the amendment and with no bar.** Before the first box, repeats were 12.7 against 22.9 for matched, -10.2, CI -19.3 to -1.2. First-box correct was 20 against 10 for matched, +15.6, CI +4.7 to +26.6, and against 14 unedited, +9.4, CI -3.1 to +21.9. The first box here means the first containing a digit, after a review found placeholders like `\\boxed{answer}` being echoed, which moved these figures from -12.4 and +17.2. So removal may help the part of the run that counts on a second model. These measures were chosen after a look at how this model behaves, and the matched arm got worse as much as removal got better, so this needs a test with bars set first. Study 15 is that test.

## Study 12: does the method carry to another model?

[Plan](PLAN_PORT.md). **The gate failed, so the study stopped there and no direction was built.** On 32 problems, MobileLLM-R1-360M, a base model given plain text, used 0.004 doubt words per 100, with 1 of 32 runs having any, against a needed 0.3 and 10. Runs do reason, all 32 past 100 words, and it boxed 25 and got 7 correct. Its text is `## Step 1:` lists and repeated LaTeX lines, with no "wait" or "hmm". This model doesn't show this behavior under this prompt, so self-doubt can't be tested on it. It's a limit on the self-doubt story and doesn't test the method. A review adds that this is a base checkpoint, and words like "wait" tend to come from later training, so this says nothing about the trained version of this model, which wasn't tried. It does show heavy-looking repetition, which is the subject of study 13.

## Study 11: what about repetition's edit frees a stuck run?

[Plan](PLAN_AXIS.md), written before any arm ran. Same 56 stuck runs, seed 7, 1,500 new tokens.

| Arm | Push on repetition's axis | Self-doubt activity after, mean | Left the loop, of 56 | Still out at the end | Finished | Correct |
|---|---:|---:|---:|---:|---:|---:|
| Unedited | 0 | +0.86 | 0 | 0 | 0 | 0 |
| Repetition, gain 0.25 | 1.07 | +0.41 | 1 | 0 | 0 | 0 |
| Repetition, gain 0.5 | 2.14 | -0.03 | 4 | 1 | 0 | 0 |
| Repetition, gain 0.75 | 3.20 | -0.48 | 17 | 7 | 1 | 1 |
| Repetition, gain 1 (study 7) | 4.27 | -0.93 | 33 | 17 | 1 | 2 |
| Repetition-only, gain 1 (study 8) | 3.91 | +0.86 | 19 | 8 | 0 | 0 |
| Repetition-only, gain 1.09 | 4.27 | +0.86 | 22 | 10 | 0 | 0 |
| Repetition-only and self-doubt both removed | 4.27 | 0 | 33 | 21 | 5 | 5 |

| Bar | Result | Verdict |
|---|---|---|
| G1. Gain 0.75 minus 0.25, at least +20 | +28.6, CI +17.9 to +41.1 | pass |
| G2. Gain 0.25 minus unedited, at least +10 | +1.8, CI 0 to +5.4 | **fail** |
| G3. Repetition minus repetition-only at the same push, at least +15 | +19.6, CI +3.6 to +35.7 | pass |
| G4. Both removed minus repetition-only at the same push, at least +15 | +19.6, CI +5.4 to +33.9 | pass |
| G5. Intact | no collapsed run in any arm | pass |

**The plan's prescribed reading was an interaction, "neither one axis nor two separate routes". A review shows that goes too far, so I'm not claiming it.** What holds is narrower: **repetition's axis alone doesn't account for the exits.** With the same push on that axis, exits were 22 with self-doubt left in place and 33 with it at zero. But one single axis tilted about 40 to 45 degrees from repetition-only toward self-doubt, which is close to repetition itself, orders every arm by push, including self-doubt alone. So a single axis other than repetition's exact one fits most of these arms. It doesn't fit all of them: a later fit at 42 degrees reproduces the dose curve and the repetition-only arms but predicts 26 exits of 56 for both removed, where 33 were observed. These arms can't separate one axis from an interaction, and study 16 tests it on new runs.

The evidence is also not strong. G3 is 18 better and 7 worse, p 0.043 by exact test, and G4 is 15 and 4, p 0.019, on 4 uncorrected bars. Two near-identical arms, repetition-only at gain 1 and 1.09, gave 19 and 22 with only 17 exits in common, so counts move by 3 to 5 from resampling alone. The response is also steep enough that a 15% error in the estimated push would produce the whole 11-exit gap, and those pushes are estimates. Also, the activity figures are mean absolute projections, so the "self-doubt after" column assumes all tokens share a sign. Only the 0 for "both" is certain.

**The response has a threshold.** Exits go 1, 4, 17, 33 as push rises, so they accelerate, and a push of 1.07 does almost nothing.

**Removing both had the most useful outcome, with no bar on it:** 5 finished and 5 correct of 56, against 1 and 2 for repetition alone, and 21 still out at the end against 17. Reading them, they're real solutions, re-derived and not copied, though all 5 prefixes already contained the answer's number, as did 28 of all 56. Small counts, which need a test on new runs, which is why study 14 exists.

**Limits.** Pushes and activities are means, and tokens vary. The interaction could be nonlinear in ways two points can't describe. Study 8's review already noted that self-doubt alone at a safe gain barely pushes on repetition's axis, so "alone it does nothing" holds only at the doses tried. 56 runs, one seed, one model, 4 bars with no correction, and the CI lows for G3 and G4 are not far above 0.

## Study 9: does adding the direction cause the behavior?

[Plan](PLAN_ADD.md), written before data and amended before any edited arm generated, after a Fable review. 48 new problems, seed 8, 3,000-token cap. The edit is `h + a d` at L19 at every token.

| Arm | Amount | Dose | Doubt per 100, first 500 words | Whole run | Runs entering a loop | Median entry, words | Answered | Correct | Collapsed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Unedited | | | 0.69 | 1.28 | 23 | 1,150 | 5 | 5 | 0 |
| Self-doubt, low | 1 | 0.012 | 1.45 | 3.01 | 37 | 1,050 | 1 | 1 | 0 |
| Random, low | 2 | 0.026 | 0.67 | 1.22 | 30 | 975 | 1 | 1 | 0 |
| Self-doubt, high | 3 | 0.096 | 5.01 | 7.43 | 48 | 550 | 0 | 0 | 0 |
| Random, high | 4 | 0.110 | 0.48 | 0.46 | 38 | 650 | 2 | 2 | 0 |
| Repetition, high | 3 | 0.081 | 0.81 | 1.18 | 45 | 650 | 0 | 0 | 1 |

| Bar | Result | Verdict |
|---|---|---|
| A1. Self-doubt high minus random high, at least +1.0 | +4.53, CI +2.79 to +7.06, 47 better and 0 worse | pass |
| A2. Self-doubt low minus random low, at least +0.5 | +0.78, CI +0.52 to +1.03 | pass |
| A3. High minus low, above 0 | +3.56, CI +1.85 to +6.03 | pass |
| A4. Self-doubt high minus repetition high, at least +0.5 | +4.20, CI +2.48 to +6.73 | pass |
| A5. Intact, at most 2 extra collapsed runs in any arm | at most 1 | pass |
| A6. Repetition high minus random high on loops, at least +20 | +14.6, CI 0 to +29.2 | **fail** |

**The plan's reading: adding self-doubt causes doubt words, specifically and with dose, so the direction works in both ways.** Removing it lowers them and adding it raises them. Self-doubt at the lower dose raised them even though its dose was under half of random's, and repetition at about the same high dose barely moved them.

**Repetition didn't pass as a cause of loops.** Adding it took loops to 45 of 48, but adding random at a similar dose took them to 38, against 23 unedited. So a large addition of a random direction causes many loops on its own, +31 points, CI +14.6 to +47.9. This matters for reading all removal results: this model tips into loops easily when disturbed. Adding self-doubt was the strongest loop inducer here, 48 of 48 with a median entry at 550 words, more than repetition at about the same dose. That had no bar.

**A review found broken runs that the collapse check misses.** Two self-doubt-high runs are garbage almost from the start, for example `I'm a wait, I'm a wait,`, and they supply 396 of the 998 uses of "wait". The bars stand, but the +4.53 is inflated. A stricter check I then ran, under 20% distinct words in the first 200, finds 3 such self-doubt-high runs and 1 repetition-high run. Dropping those 4 problems gives A1 +3.08, CI +2.63 to +3.58, A2 +0.72, A3 +2.20 and A4 +2.75, so every bar's verdict holds. Also, 21 of the 48 self-doubt-high runs enter a loop before word 500, so the amendment's assumption that the first 500 words come before loops doesn't hold for this arm. The new-context count below covers that.

**"Specific" has a limit.** Repetition has cosine 0.42 with self-doubt, so repetition at amount 3 carries 1.26 units of self-doubt, more than self-doubt low's 1. Yet it gave 0.81 against 1.45. So it's not a clean control, and the effect isn't linear in the self-doubt component alone.

**What the doubt reads like.** Reading typical self-doubt-high runs, it's mostly doubt about reading the problem: "Wait, hold on... Let me re-read... Maybe I'm missing something". It's less often doubt about arithmetic.

**Checks, no bars.** The rise isn't just repeated text. Counting only doubt words in new context, self-doubt high minus random high is +2.31, CI +2.0 to +2.63. In the first 200 words it's +3.74, CI +2.08 to +6.11. Most of the rise is the word "wait", 998 uses against 79 for random high. Loop entry and answers also moved, with no bars: all 48 self-doubt-high runs entered a loop, and none answered.

**Blind read, set after the counts.** One Claude reader (Haiku), blind to arm, compared the first 200 words of paired runs on the same problem and picked which second-guessed its own work more. It picked self-doubt high over random high on 20 of 24 problems, p 0.002, which passes that check's bar. It picked self-doubt low over random low on 16 of 24, p 0.15, which reaches the 60% rate but **fails** the p requirement. So the high dose reads as more doubt in meaning, and the low dose isn't shown to. One reader, Claude only, and the check was added after seeing the counts.

**Limits.** Doubt words are a count of words, and most are "wait". Amounts are coarse, so low doses differ by about 2 times between self-doubt and random, in the direction that works against self-doubt. A fixed amount at every token isn't how the model's own activity behaves. One seed, one model, 48 problems, no correction for 6 tests.

## Study 10: does removing self-doubt reason better, or commit sooner?

[Plan](PLAN_COMMIT.md), written before computing anything. It reads existing traces on the CPU, with no new generation. 119 of the 128 replication problems count, because the other 9 have an answer that's already in the problem statement or isn't a plain number.

| Of 119 problems | Right value appears in the text | Answered in a box | Correct |
|---|---:|---:|---:|
| Unedited | 81 | 12 | 11 |
| Self-doubt removed | 96 | 51 | 46 |
| Matched random removed | 80 | 11 | 10 |

| Bar | Result | Verdict |
|---|---|---|
| C1. Unedited reaches it at least twice as often as it's correct | 81 against 11 | pass |
| C2. Self-doubt removal changes reaching by at most +10 | +12.6, CI +5.0 to +20.2 | **fail**, and the CI clearly excludes 0 |
| C3. Among problems reached in both, correct rises at least +20 | +40.8 of 76 problems, CI +28.9 to +52.6, 33 better and 2 worse | pass |
| C4. Random's same difference includes 0 | -1.4, CI -11.1 to +8.3 | pass |

**The plan's reading for this pattern: more right values are found and more are stated, so it isn't only about committing.** But the committing part is much the larger of the two. Unedited runs had the right value in their text on 81 problems and stated it as an answer on 11. Removal added 15 problems reaching the value and 35 correct.

So 69 of the unedited model's 81 reaching runs never box anything, and 33 of the 37 new correct answers came on problems unedited already reached. That supports committing as the larger part. But a review shows "found it and failed to box it" fits only about half of those 69: the right value is the last `= value` in 34 of them, and the rest went on to other candidates. So I'd say the unedited model often reaches the right value and doesn't settle on it, which is weaker than saying it had the answer.

**Checks.** A different problem's answer appears in a run's text by coincidence 10.5% to 14.5% of the time, against about 68% to 81% for the right answer, so most reaching isn't coincidence. That varies by size. In the unedited arm it's 33.3% for answers under 20, on 25 problems, 7.5% for 20 to 100, and 2.7% for 100 or more. The value first appears at a similar point in each arm, a median of 239 words unedited and 225 removed.

**Limits.** Reached is a word match, not a judgment that the reasoning was sound: the value may appear as a step toward something else and then get abandoned. Conditioning on problems reached in both arms selects them. More of a run's text being free of loops gives it more room to reach a value, which could explain the +12.6. This reads old traces, so it's a new look at data already used to choose this hypothesis, and it needs testing on new runs. One model.

## Study 8: with matched doses, what frees a stuck run?

[Plan](PLAN_DOSE.md), written before any arm ran. Same 56 stuck runs, seed 7, 1,500 new tokens, edits from the prefix's last token on. Unedited and repetition are reused from study 7. Target dose was repetition's own, 0.0176 nats per token.

| Arm | Gain | Dose | Left the loop, of 56 | Still out at the end | Finished | Correct | Collapsed |
|---|---|---|---:|---:|---:|---:|---:|
| Unedited | | | 0 | 0 | 0 | 0 | 0 |
| Random with repetition's overlap with the mean | 1 | 0.0007 | 0 | 0 | 0 | 0 | 0 |
| Repetition | 1 | 0.0176 | 33 | 17 | 1 | 2 | 0 |
| Repetition with self-doubt projected out | 1 | 0.0217 | 19 | 8 | 0 | 0 | 0 |
| Random | 24 | 0.0160 | 5 | 0 | 0 | 0 | 0 |
| Self-doubt | 2 | 0.0260 | 2 | 0 | 0 | 1 | 0 |
| Self-doubt with repetition projected out | 2 | 0.0061 | 1 | 0 | 0 | 0 | 0 |

| Bar | Result | Verdict |
|---|---|---|
| F1. Repetition beats its control, at least +20 | +58.9, CI +46.4 to +71.4 | pass |
| F2. Repetition beats random at dose, at least +20 | +50.0, CI +37.5 to +62.5 | pass |
| F3. Self-doubt beats random at dose, at least +20 | -5.4, CI -14.3 to +3.6 | **fail** |
| F4. Repetition without self-doubt beats random, at least +20 | +25.0, CI +14.3 to +37.5 | pass |
| F5. Self-doubt without repetition beats random, at least +20 | -7.1, CI -16.1 to 0 | **fail** |

**The plan's reading for this pattern: repetition has an effect self-doubt can't produce at this dose.** That is the claim this study supports. A random direction at this dose freed 5 runs and repetition freed 33, so direction matters here, for the one random direction and one dose tried.

**What this doesn't show.**

- **I first wrote that this rules out one axis. It doesn't.** Equal KL isn't equal push along repetition's axis. Using activity inside loops (repetition 4.27, self-doubt 0.86, cosine 0.42), push along repetition's axis was about 4.27 for repetition, 3.91 for repetition without self-doubt, 0.72 for self-doubt at gain 2, and 0 for self-doubt without repetition and for random. Exits were 33, 19, 2, 1 and 5. So a story where only repetition's axis frees loops mostly fits, and self-doubt could simply be too weak on that axis to show anything at a safe gain.
- Repetition without self-doubt still has cosine 0.91 with repetition, so F4 doesn't test that axis being absent.
- The fall from 33 to 19 has more than one explanation. Plain repetition removal also pushes self-doubt's activity past zero to about -0.93, and the projected version leaves it at about +0.86.
- Random at gain 24 left the loop 5 times, and none of those 5 were still out at the end. Reading them, they're real reasoning for 150 to 400 words and then a new loop, and 3 of the 5 contain garbled phrases. Repetition's 17 lasting exits are the larger difference.
- Leaving a loop rarely gave an answer, 1 finished and 2 correct of 33, within 1,500 tokens.
- One self-doubt exit decays into `Total = 82. *` repeated, which the collapse check misses, and two sit near the 0.5 threshold at 0.47 and 0.41.

**Limits.** The dose match is coarse because the curves are steep. Self-doubt reached 1.5 times the target and self-doubt-only reached 0.35 times, which passes the plan's rule but makes F5 the weaker of the two failures. Gain 3 would have overshot, 0.055 for self-doubt-only. Doses were calibrated on teacher-forced text and not measured during generation. Gains above 1 reverse a component instead of removing it. KL is one definition of dose. 5 bars with no correction, one seed, one model.

## Study 7: can removing a direction free a run that's already stuck?

Bars in [PLAN_RESCUE.md](PLAN_RESCUE.md). **E1, E4 and E5 pass, and E2 and E3 fail. Removing repetition moved 33 of 56 stuck runs out of their loop and removing self-doubt moved 1, but this was not a fair comparison, so it doesn't show that the two directions do different jobs.**

| Arm, 56 stuck runs, 1,500 more tokens | Left the loop | Still out at the end | Finished | Correct | Identical to unedited |
|---|---|---|---|---|---|
| Unedited | 0 | 0 | 0 | 0 | n/a |
| Self-doubt removed | 1 | 0 | 0 | 0 | 49 |
| Repetition removed | 33 | 17 | 1 | 2 | 8 |
| Matched random removed | 0 | 0 | 0 | 0 | 56 |

| Bar | Needed | Result | Verdict |
|---|---|---|---|
| E1. Loops hold on their own | unedited escaped at most 10% | 0 of 56 | Pass |
| E2. Self-doubt frees | at least +20 points, CI above 0 | +1.8 (CI 0 to +5.4) | Fail |
| E3. Self-doubt beats matched random | CI above 0 | +1.8 (CI 0 to +5.4) | Fail |
| E4. Repetition frees | at least +20 points, CI above 0 | +58.9 (CI +46.4 to +71.4) | Pass |
| E5. Intact | at most 2 more collapsed | 0 collapsed in every arm | Pass |

A cross-model review found these limits, and I checked the first two against the raw files:

- **The doses aren't matched.** Inside loops, the part of the residual along repetition averages about 4.2, against about 0.65 for self-doubt. So removing repetition is a push about 6 times larger, and repetition also has cosine 0.39 with the residual mean. E4 had no control of its own: the random direction with the same overlap with the mean wasn't run. So "repetition frees" can't yet be told apart from "a big enough push frees".
- **Identical outputs aren't a bug, they show how sure a loop is.** The next token is over 99% certain at 96% of loop positions. Same seed plus the same top token gives the same text. So E2's null means this edit was too weak to change a token that certain. It doesn't mean self-doubt plays no part.
- **My own check agrees (`loop_confidence.py`, 7 runs, last 30 tokens of each prefix).** The top token averages 98.8% and is over 95% at 96% of positions. Removing self-doubt or matched random changed the top token at 0% of positions, and repetition at 1.4%. Next-token KL was 0.0056, 0.00007 and 0.020. Repetition's activity is 1.8 before loops and 4.3 inside them, and self-doubt's is 0.66 and 0.86.
- **"Left the loop" isn't "free".** Of the 33, 17 are still out at the end, and many of the rest fall back in or switch to another loop. Only 1 finished. It breaks the loop more often than it reaches an answer.
- **One axis may explain both directions.** They have cosine 0.42, and an [earlier test](../whyfix/RESULTS.md) lost most of self-doubt's gain when that overlap was removed. So "self-doubt delays and repetition holds" is one reading, and "both push the same axis by different amounts" is a rival that this study can't rule out.
- **The prefix stays unedited in every arm**, so this tests only edits made after the loop has formed.

What holds: stuck runs stay stuck under fresh sampling, and a single direction removal can move many of them out of the loop without collapsing the text. Study 8 tests the rest with matched doses.

Scripts: `rescue.py`, `score_rescue.py`, `audit_rescue.py`. Data: `runs/research/rescue_results.json`, `rescue_audit.json`.

## Study 3: does the direction predict doubt before it happens?

Bars in [PLAN_ACTIVITY.md](PLAN_ACTIVITY.md). **All five pass, with one bar fixed after a void first run. On normal runs the direction reads as "about to write a doubt word", though a good part of its raw score comes from where sentences start.** Nothing was edited here. It reads L19 on 64 unedited traces no direction was built from.

| AUROC, mean within a trace | Lag 1 | Lag 5 | Lag 1, only words opening a sentence or line |
|---|---|---|---|
| Original self-doubt | **0.935** | 0.747 | **0.688** |
| Writer 2 | 0.887 | 0.779 | 0.670 |
| Writer 3 | 0.880 | 0.756 | 0.644 |
| Repetition direction | 0.649 | 0.610 | 0.572 |
| Matched random | 0.573 | 0.516 | 0.597 |

| Bar | Needed | Result | Verdict |
|---|---|---|---|
| D1. Predicts the next token | at least 0.70, CI above 0.5 | 0.935 (CI 0.927 to 0.943) | Pass |
| D2. Beats control | at least +0.10, CI above 0 | +0.362 (CI +0.329 to +0.396) | Pass |
| D3. Not just sentence starts, fixed rule | at least 0.60, CI above 0.5, at least 20 traces | 0.688 (CI 0.652 to 0.724), 44 traces | Pass |
| D4. Leads the word | lag 5 at least 0.60, CI above 0.5 | 0.747 (CI 0.697 to 0.792) | Pass |
| D5. Different from repetition | CI above 0 | +0.286 (CI +0.264 to +0.308) | Pass |

What this adds:

- **Reading and removing agree.** The direction that reduces doubt words when removed is also high just before them when left alone, in text it wasn't built from.
- **Most of the raw score is position.** The score falls from 0.935 to 0.688 once only sentence and line openers are compared, and matched random is at 0.597 there. So it adds +0.091 over a random direction in the same spot (CI +0.037 to +0.147). That's the fair size of the effect, and it's modest.
- **Independent writers mostly agree.** Directions from pairs written by other Claude models predict nearly as well on the raw score. At sentence openers, writer 2 adds +0.074 over random (CI +0.019 to +0.130), and writer 3's +0.048 isn't clear (CI −0.007 to +0.105).

Limits:

- D3 was first run with a rule that left it resting on 1 trace, so that run is void. I fixed the rule after seeing the other four bars' numbers, without knowing this result.
- Lag 5 isn't controlled for position, so D4 may also partly reflect line structure.
- The word pattern misses doubt written other ways. Predicting a word doesn't prove the model uses this direction to write it, though the removal tests point that way.

Scripts: `activity.py`, `score_activity.py`. Data: `runs/research/activity_results.json`, `activity_traces.json`. The void first run is kept as `*_v1.json`.
