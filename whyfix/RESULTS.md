# Can `why` find a fix for a real failure?

**Where this ended:** word lists failed, and removing one direction passed on three seeds and on 64 new problems. See the end of this file.

**Read this first.** The first two rounds below used a bad loop detector, found after they were scored. Of its 94 "loop onsets", 80% repeat 8 tokens or fewer and 1% start a 64-token repeat. Most are the model quoting the problem. So the why, active, and contrast-free numbers below are about short repeats, not loops, and they aren't a fair test of `why` on loops. The corrected test is at the end of this file.

Run on 2026-11-04. Bars are in [PLAN.md](PLAN.md), written before generation. 16 test problems, same sampling as the README table, 3,000-token cap.

| Arm | Repeated 8-grams | Correct | Problems better / worse than unedited |
|---|---|---|---|
| Unedited | 45.9% | 3/16 | |
| Hand-written concept edit (README) | 11.4% | 3/16 | |
| Astra's edit (README) | 11.8% | 3/16 | |
| Random edit (README) | 46.1% | 2/16 | |
| **why words**: remove top 16 from `why` at 94 loop onsets | **64.1%** | 2/16 | 5 / 10 |
| Random words, matched on firing rate | 41.1% | 3/16 | 9 / 7 |
| Active words: 16 most often active at onsets | 27.1% | 3/16 | 13 / 3 |
| f976 only: the word edits turn off most | 36.2% | 1/16 | 12 / 4 |

| Bar | Needed | Result | Verdict |
|---|---|---|---|
| F1. Fewer loops | at most 30.6% | 64.1% | **Fail** |
| F2. At least 10 points below random words | below 31.1% | 64.1% vs 41.1% | **Fail** |
| F3. Below active words | below 27.1% | 64.1% | **Fail** |
| F4. Accuracy: lose at most 1 of 3 | at least 2/16 | 2/16 | Pass |
| F5. f976 alone | at most 30.6% | 36.2% | **Fail** |

**What I first concluded, now in doubt because of the detector.** `why` tells the truth about one token: remove these words and this next token changes. That truth didn't add up to a fix. Words that push one repeated token are about the content of that token (names, units, numbers). Removing them everywhere didn't stop the model repeating, and made it repeat more on this sample.

Looping looks like a state of the model, not something that happens at one token. That would explain why simply-active words did better than attributed ones, and why a direction chosen from examples of looping text (11.4%) still does far better than any word set here. The second-best idea in this table was the naive baseline, which is a useful warning about `why`.

Limits: 16 problems, one sampling seed, and active words' 27.1% had no bar of its own. Removing words with the SAE at every token also isn't the same kind of change as a weight edit.

## Repeat at seed 1

| Arm | Repeated 8-grams | Correct | Better / worse than unedited | Change vs unedited (CI) |
|---|---|---|---|---|
| Unedited | 41.4% | 3/16 | | |
| Active words | 23.4% | 3/16 | 10 / 4 | −18 points (−36 to −1) |
| Diff words | 32.5% | 2/16 | 8 / 8 | −9 points (−25 to +7) |
| Random words | 32.5% | 2/16 | 10 / 5 | −9 points (−24 to +4) |

| Bar | Needed | Result | Verdict |
|---|---|---|---|
| F6. Active words | at most 27.6%, and 10 points below random | 23.4%, 9.1 below | **Fail**, narrowly |
| F7. Diff words | at most 27.6%, and 10 points below random | 32.5%, same as random | **Fail** |

Active words came in below unedited on both seeds, by similar amounts, and I'm not treating that as a pass. Random words also vary a lot: 41.1% and 32.5%. At n=16, seed-to-seed noise is about as big as these effects, so only the concept and Astra edits, at 11–12%, are clearly separated from it.

Diff words failing is informative. The words an edit turns off aren't the edit: removing the 16 that fall most under Astra's edit is no better than removing random words. The edit's effect isn't held in those words.

## The corrected test, seed 2

This is the fair test, using the fixed loop detector and the bars written before generation. Words came from the mining traces only. The same 16 problems were used. Intervals are 95%, paired by problem.

| Arm | Repeated 8-grams | Correct | Better / worse than unedited | Change vs unedited (CI) |
|---|---|---|---|---|
| Unedited | 34.2% | 5/16 | | |
| Contrast words | 33.7% | 2/16 | 10 / 6 | −0.5 points (−17 to +18) |
| Why words | 51.6% | 1/16 | 6 / 8 | +17 points (+2 to +35) |
| Random words | 57.3% | 1/16 | 3 / 12 | +23 points (+5 to +41) |
| Active words | 70.5% | 0/16 | 3 / 12 | +36 points (+14 to +57) |

| Bar | Needed | Result | Verdict |
|---|---|---|---|
| F9. Why words | at most 22.8%, and 10 points below random | 51.6%, 5.7 below | **Fail** |
| F10. Active words | at most 22.8%, and 10 points below random | 70.5%, 13.3 above | **Fail** |
| F11. Contrast words | at most 22.8%, and 10 points below random | 33.7%, 23.6 below | **Fail**: under random, but no better than unedited |

None passed, so by the plan this line of tests stops and the concept edit stays the recommended fix.

What this means:

- None of the corrected word lists fixed loops, and none came near the concept and Astra edits' 11–12%.
- Three of the four lists made loops worse. The corrected active list is not the earlier one. The earlier list, from the flawed detector, reached 27.1% and 23.4% at seeds 0 and 1, so its result is still open. See the follow-up below.
- These results are exploratory beyond the bars. Random words' effect ran from −9 points to +23 across seeds, and each seed's list differs. So one list of 16 words can shift this number a lot with no connection to loops, and only large gaps like the concept edit's can be read at n=16.
- Correct answers fell from 5/16 to 2/16 or fewer in every word arm. That is small, was not a bar, and is worth a look only because it points the same way as the loop numbers.
- It doesn't show that loops have no words in this dictionary. It shows that removing up to 16 of them, at L19, with this method, isn't how to stop loops. One explanation, measured below: the working edits remove one direction, and that direction is spread across many words.

Scorer: `whyfix/score_seed2.py`. Data: `runs/whyfix/seed2_*_traces.jsonl`, `runs/whyfix/seed2_results.json`.

## Follow-up: the earlier active list at seed 2

Added 2026-11-04 after the results above and before running it. The plan's stop rule is met, so this is outside it. The first active list dropped about 18 points at both seeds it ran on, which is the one consistent result in this line, and it missed its only bar narrowly. It has never run at seed 2, where unedited and random2 already exist.

| Bar | Needed |
|---|---|
| F12. Earlier active list, seed 2 | at most 22.8%, and at least 10 points below random2 (57.3%) |

| Result | 33.9% repeats, 4/16 correct. Change vs unedited −0.3 points (CI −15 to +15), 9 better and 7 worse. 23.4 points below random2 (CI −40 to −8). |
|---|---|

**F12 fails.** It meets the margin over random2, but it's no better than unedited, which is the part that matters. So the earlier 18-point drops did not repeat. Across three seeds the same list gave −19, −18 and 0 points, against unedited rates of 46%, 41% and 34%. It may help more when a run loops more, but this test can't say so, and three seeds can't support it.

This closes the line. No word list has beaten the concept edit, or passed a bar.

## Why word lists can't carry it: the direction is spread

Bars in [PLAN_SPREAD.md](PLAN_SPREAD.md), written before the measurement. No text generated. On the 24 mining traces (67,320 tokens) at L19, each residual's projection on an edit's direction splits into one part per word plus the dictionary's error.

| | NeuroSpeak edit | Astra's edit | 20 random directions |
|---|---|---|---|
| Top 16 words' share of absolute projection | 30.7% | 33.3% | 10–14% |
| Dictionary error's share | 9.3% | 9.1% | 13–15% |
| Words needed for half of the words' part | 60 | 50 | 159–204 |
| Words needed for 90% | 857 | 849 | 1,356–1,494 |
| Closest word, cosine | f14407, 0.61 | f976, 0.68 | 0.12–0.14 |

| Bar | Needed | Result | Verdict |
|---|---|---|---|
| S1. Spread | top 16 under 50% | 30.7%, 33.3% | Holds |
| S2. Not just any direction | outside the random range | 2.2 and 2.4 times the top of the range | Holds |

So the direction sits between the two simple stories. It isn't one word, and it isn't noise. A list of 16 can hold only about a third of it, and half takes 50 to 60 words. The dictionary sees the direction well, with less error than for a random direction. Two independent edits' closest words are f976 and f14407, which are also the top two words in the diff report.

Limits. This fits the explanation and doesn't prove it. Shares add absolute values, so they ignore parts that cancel. Only L19 was measured, and both edits also change L15. And removing a word takes out its whole vector, not just its part along the direction, so a word list also changes other things. That may be why three lists made loops worse, and it hasn't been tested. The next test would remove only the direction during generation, with no words, to separate the direction from the method.

Script: `whyfix/spread.py`. Data: `runs/whyfix/spread.json`.

## Removing only the direction, no words

Bars in [PLAN_SPREAD.md](PLAN_SPREAD.md), written before generation. Each arm subtracts the projection on one unit direction from L19's output at every token. No weights change, and L15 isn't touched. Seed 2, same 16 problems.

| Arm | Repeated 8-grams | Correct | Cut at cap |
|---|---|---|---|
| Unedited | 34.2% | 5/16 | 93.8% |
| NeuroSpeak edit's direction | **10.3%** | 7/16 | 68.8% |
| Astra edit's direction | **13.0%** | 6/16 | 81.2% |
| Random direction | 39.2% | 1/16 | 100% |

| Bar | Needed | Result | Verdict |
|---|---|---|---|
| F13. dir_neurospeak | at most 22.8%, and 10 points below random | 10.3%, 28.9 below | Pass |
| F14. dir_astra | at most 22.8%, and 10 points below random | 13.0%, 26.2 below | Pass |

Passes on one seed, and replication is below. Word lists holding a third of this direction failed, and the direction itself passes. That fits the spread explanation.

Replication, seed 0: dir_neurospeak 11.5% and 5/16 correct, dir_random 44.9% and 1/16, unedited 45.9% and 3/16. **F15 passes** (needed at most 30.6%, and 10 points below random; it is 33.4 below).

Replication, seed 1: dir_neurospeak 8.8% and 4/16 correct, dir_random 40.0% and 1/16, unedited 41.4% and 3/16. **F16 passes** (needed at most 27.6%, and 10 points below random; it is 31.2 below).

| dir_neurospeak vs random, paired by problem | Difference in points (CI) | Better / worse |
|---|---|---|
| Seed 0 | −33.4 (−49 to −19) | 16 / 0 |
| Seed 1 | −31.2 (−44 to −18) | 14 / 2 |
| Seed 2 | −28.9 (−44 to −15) | 13 / 3 |

All three seeds pass, on the same 16 problems. These 16 are the problems the concept edit was first judged on, and a new set is the next check. Correct answers across the three seeds: 16/48 with the direction removed, 11/48 unedited, 3/48 with a random direction. That wasn't a bar.

## Accuracy on 64 new problems

Bars in [PLAN_SPREAD.md](PLAN_SPREAD.md), written before this ran. GSM8K test problems, a new seed (3), the same 3,000-token cap. None of these problems were used to find the direction, and the direction comes from an edit built on four hand-written pairs.

| | Unedited | NeuroSpeak edit's direction removed |
|---|---|---|
| Repeated 8-grams | 39.9% | 9.2% |
| Correct | 3/64 (4.7%) | 10/64 (15.6%) |
| Finished before the cap | 2/64 | 7/64 |
| No boxed answer | 61/64 | 52/64 |

| Bar | Needed | Result | Verdict |
|---|---|---|---|
| A1. Correct answers | at least 5 points up, CI above 0 | +10.9 points (CI +3.1 to +18.8), 7 gained and 0 lost | Pass |
| A2. Repeats | at most 26.6% | 9.2% | Pass |

What this shows:

- On new problems, removing one direction at one layer at runtime cut repeats by three quarters and more than tripled correct answers. No weights changed.
- It's mostly about finishing. 6 of the 7 runs that finished were right, and 52 of 64 still had no answer at the cap. The direction helps the model stop going in circles, and it's still too slow for this cap.

Limits:

- 10 correct answers is a small count. The interval's low end is +3 points.
- This test had no random-direction arm. In the 16-problem test, the random arm stayed within 5 points of unedited on repeats (39.2, 44.9 and 40.0 against 34.2, 45.9 and 41.4) and got fewer right (3/48 against 11/48), so a random direction isn't expected to explain this, though it wasn't run here.
- One seed, one model, one kind of problem, and a 3,000-token cap. It doesn't reproduce Astra's MATH-500 result, which used far longer runs. It's not compared here with the weight edits on the same problems.
- The model's answer to gsm-test-56 is scored wrong because the scorer doesn't strip a backslash before `$`. The true count for this arm is 11/64. The bar was judged at 10/64, and passing doesn't depend on it.
- Harm to normal text was measured later: see below.
- It explains where the effect sits, not what the direction means. Its closest words have untested names.

Scorer: `whyfix/score_accuracy.py`. Data: `runs/whyfix/gsm64_*_traces.jsonl`, `runs/whyfix/gsm64_results.json`.

## Where this line ended

Word lists can't fix loops. Their direction can. A direction spread over dozens of words works at runtime, and any 16 of those words do not. `brain.without` is the supported way to do it, and it joins the concept edit as a tested fix.

## Review found a weak control

Written before the controls below ran. A cross-model review computed that dir_neurospeak overlaps the residual's constant part, and one random direction removes almost nothing. So the random-direction margin in F13–F16 can't tell a repetition direction from any shrinking of L19's mean. Also, F14 isn't independent of F13, since the two directions have cosine 0.70 and were both built against repetition. Bars C1 and C2 test this.

| Seed 2, same 16 problems | Repeated 8-grams | Correct | Collapsed into gibberish |
|---|---|---|---|
| Unedited | 34.2% | 5/16 | 0/16 |
| NeuroSpeak edit's direction | 10.3% | 7/16 | 0/16 |
| Random direction, same overlap with the mean | 43.3% | 4/16 | 0/16 |
| The mean itself | 10.9% | 0/16 | 16/16 |

| Bar | Needed | Result | Verdict |
|---|---|---|---|
| C1. dir_matched | above 22.8% | 43.3% | Pass: the shared mean part doesn't explain the effect |
| C2. dir_mean | above 22.8% | 10.9% | **Fail by the letter, and the number is an artifact** |

C2 needs its reason stated. Removing the mean destroys the model: all 16 traces become one character repeated, loss on Wikipedia rises 3.6 nats, and 0/16 are right. The repeat metric counts word 8-grams and returns near zero when text has no spaces, so it scored the collapse as a low repeat rate. The bar's stated meaning, that removing the mean also stops loops, is not what happened. This also shows a real blind spot in the metric, so `whyfix/degenerate.py` checked all arms. Every other arm has 0 collapsed traces, except one trace in seed 1's active arm. No direction result depends on the blind spot.

## Harm to normal text

Bars N1 and N2 in [PLAN_SPREAD.md](PLAN_SPREAD.md), written before measuring. Rise in next-token loss with each direction removed, in nats per token, with 95% intervals for the first row.

| Removed at L19 | Wikipedia, 160 chunks | GSM8K solutions, 117 chunks |
|---|---|---|
| NeuroSpeak edit's direction | +0.028 (0.025 to 0.031) | +0.017 (0.015 to 0.019) |
| Astra edit's direction | +0.028 | +0.029 |
| Random direction | +0.003 | +0.001 |
| Random, same overlap with the mean | +0.045 | +0.025 |
| The mean | +3.63 | +1.42 |

N1 and N2 pass, needing at most 0.05. The cost isn't zero, and it's about the size of the weight edits' 0.03. The matched random direction costs as much or more and doesn't help loops, so the cost may come from the overlap with the mean, while the loop effect doesn't. That reading is inferred, not tested.

## Final state of this line

Word lists don't fix loops. One direction does, at runtime, with a small cost. The evidence holds up against the random control, the matched control, and new problems. Still unproved: that it's specific to repetition, since no direction from an unrelated concept was tried, and only one matched random draw ran. It also explains where the effect sits, not what the direction means.

Data: `runs/whyfix/degenerate.json`, `runs/whyfix/harm.json`. Scripts: `whyfix/degenerate.py`, `whyfix/harm.py`.

## Self-doubt direction on 64 problems

Bars in [PLAN_SELF_DOUBT.md](PLAN_SELF_DOUBT.md), written before generating. **All five pass. It gets more answers out under the cap, and this test doesn't show better math.** The same 64 GSM8K test problems, seed 3 and 3,000-token cap as the repetition direction's test. Counts use the scorer fixed before this run, so they differ from the table above.

| | Unedited | Matched random | Self-doubt | Self-doubt without its repetition overlap | Repetition direction |
|---|---|---|---|---|---|
| Correct | 4/64 | 4/64 | **23/64** | 7/64 | 13/64 |
| Boxed an answer | 4/64 | 4/64 | 26/64 | 7/64 | 15/64 |
| Finished before the cap | 2/64 | 1/64 | 10/64 | 2/64 | 7/64 |
| Finished and correct | 2/64 | 1/64 | 9/64 | 2/64 | 6/64 |
| Repeats | 39.9% | 43.8% | 17.5% | 42.9% | 9.2% |
| Doubt words per 100 | 1.44 | 1.51 | 0.26 | 0.99 | 0.71 |
| Collapsed runs | 0 | 0 | 0 | 0 | 0 |

| Bar | Needed | Result | Verdict |
|---|---|---|---|
| S1. Correct | at least +5 points, CI above 0 | +29.7 points (CI +18.8 to +40.6), 19 gained, 0 lost | Pass |
| S2. Repeats | at most 26.6% | 17.5% | Pass |
| S3. Control | beats matched random on both | correct +29.7 (CI +18.8 to +40.6), repeats −26.3 (CI −33.9 to −18.9) | Pass |
| S4. Intact | at most 2 collapsed runs | 0 | Pass |
| S5. On target | at most 0.96 doubt words per 100 | 0.26 | Pass |
| O1. Effect without the repetition overlap | that arm passes S1 and S2 | +4.7 points (CI −4.7 to +14.1), repeats 42.9% | **Fail** |

Against the repetition direction, with no bar: correct +15.6 points (CI +4.7 to +28.1), repeats +8.3 (CI +3.6 to +13.1). So self-doubt got more answers right and left more repetition.

What the audit found (`audit_self_doubt.py`):

- **It's about committing to an answer.** Of the boxed answers, 23 of 26 are right, against 4 of 4 unedited and 13 of 15 for the repetition direction. Four answers are too few to compare accuracy, so this shows no rise and can't rule one out. The gain in count comes from answering at all. In the runs I read, the unedited model often reaches the right number and keeps circling. Longer runs might close the gap, and this test can't say, since the cap was 3,000 tokens.
- **Most of these runs still don't stop.** 14 of the 23 correct runs hit the cap, 14 boxed inside the thinking block, and some keep repeating lines such as "(Note: 35% is the answer)" after boxing. Only 9 runs finished and were right, against 2 unedited, a much smaller count than 23.
- **The count isn't sensitive to which box is scored.** Taking the first box gives 24, and in 21 of the 23 every box agrees. One answer, "48 pieces", is right but counted wrong under the frozen scorer, so the true count is at least 24.

Limits:

- One seed, and the scorer changes don't decide it: the old scorer gives 3, 3, 21, 4 and 10 correct across the table's columns. These problems were also used in the repetition direction's test, though they played no part in finding the self-doubt direction or picking this lead.
- Removing the 0.42 overlap wiped out the effect, even though the result still has cosine 0.91 with the original. This suggests the shared part matters, and the repetition direction alone reached only 13/64. So why self-doubt does better is unknown. That arm's overlap with the residual mean also changed, from −0.005 to −0.18, so it isn't a clean comparison.
- The words "self-doubt" describe the examples, not a proven meaning for the direction. It lowers doubt words, and it could be doing something else as well.
- Cost on normal text comes from the earlier run on different text (0.007 nats) and wasn't remeasured here. The other arms' numbers can't be compared across seeds.

Scorer: `whyfix/score_self_doubt.py`. Data: `runs/whyfix/gsm64_dir_self_doubt*_traces.jsonl`, `runs/whyfix/self_doubt_results.json`.

## Replication on unused problems

Bars in [PLAN_REPLICATE.md](PLAN_REPLICATE.md), committed in `33d385f` before any output existed. 64 GSM8K test problems no earlier test used, a new seed (4), the same 3,000-token cap, and the same saved directions. **All seven bars pass. The self-doubt result replicates. It boxes more right answers than the repetition direction, but that lead isn't clear on the stricter count. As before, this is about answering within the cap, and it doesn't show better math.**

| | Unedited | Matched random | Self-doubt | Repetition direction |
|---|---|---|---|---|
| Correct | 4/64 | 5/64 | **26/64** | 14/64 |
| Correct, numbers compared by value | 7/64 | 6/64 | 26/64 | 15/64 |
| Boxed an answer | 8/64 | 7/64 | 28/64 | 17/64 |
| Finished before the cap | 0/64 | 1/64 | 14/64 | 10/64 |
| Finished and correct | 0/64 | 1/64 | 14/64 | 9/64 |
| Repeats | 48.8% | 46.4% | 26.1% | 8.1% |
| Doubt words per 100 | 1.77 | 1.32 | 0.23 | 0.72 |
| Collapsed runs | 0 | 0 | 0 | 1 |

| Bar | Needed | Result | Verdict |
|---|---|---|---|
| R1. Correct | at least +5 points, CI above 0 | +34.4 points (CI +23.4 to +45.3), 22 gained, 0 lost | Pass |
| R2. Repeats | at most 32.5% | 26.1% | Pass |
| R3. Control | beats matched random on both | correct +32.8 (CI +20.3 to +45.3), repeats −20.3 (CI −28.4 to −12.5) | Pass |
| R4. Intact | at most 2 collapsed runs | 0 | Pass |
| R5. On target | at most 1.18 doubt words per 100 | 0.23 | Pass |
| R6. Stricter count | finished and correct, CI above 0 | +21.9 points (CI +12.5 to +32.8) | Pass |
| R7. Against repetition | correct, CI above 0 | +18.8 points (CI +6.2 to +31.2) | Pass |

**Against the repetition direction, only R7's count is clear.** Counting only runs that finished and were right, comparing numbers by value, self-doubt's lead over repetition has intervals that include 0 on both seeds: +6.2 points (CI −6.2 to +18.8) here and +4.7 (CI −4.7 to +15.6) at seed 3. Repeats went the other way: self-doubt left 18.0 points more than the repetition direction (CI +11.7 to +24.6). Across both tests, comparing numbers by value, correct answers out of 128 are 11 unedited, 10 matched random, 49 self-doubt and 28 repetition direction.

**A scorer miss found after scoring.** The frozen scorer marks `12.00` wrong against 12. Three of these are in the unedited arm, one in the matched arm and one in the repetition arm, so it slightly favors self-doubt. The bars stand as scored, as the plan says. Comparing numbers by value (`sensitivity_replicate.py`) gives self-doubt minus unedited +29.7 points (CI +17.2 to +42.2) and against repetition +17.2 (CI +4.7 to +31.2), so every comparison keeps its interval above 0. It doesn't change the seed 3 counts.

What this adds:

- **It stops some runs.** 14 runs finished here and all 14 were right, against none unedited. 12 of the 26 correct runs still hit the cap, and all 12 boxed inside the thinking block.
- **It's still about answering.** A rough text check finds the right number after an equals sign in 36 of 64 unedited runs and 46 of 64 self-doubt runs. A different check by the reviewer gave 42 and 51, so treat these as approximate. Most unedited runs never box it. Whether longer runs close the gap isn't tested.
- **Less repetition isn't the whole story.** Across both seeds the repetition direction leaves fewer repeats and fewer right answers. So repeats alone don't predict correct answers here.

Limits:

- Two seeds and 128 problems of one kind, on one model, at one cap.
- Self-doubt's repeats were 26.1% here, against 17.5% at seed 3. R2 passes because it's relative to unedited, which also rose, to 48.8%.
- The commit showing the plan came first exists only on this Mac, so its timestamp isn't independent proof. `directions.pt` isn't in git, and its unchanged state rests on a file time of 08:39 that morning.
- The direction's meaning is still untested. "Self-doubt" names the examples it came from. The overlap-removed arm wasn't rerun, so its failure rests on seed 3 alone.
- Cost on normal text wasn't remeasured.
- I wrote the pairs, scores and bars, and one person's small set of hand-written pairs produced this direction.

Scorer: `whyfix/score_replicate.py`, `whyfix/sensitivity_replicate.py`, `whyfix/audit_replicate.py`. Data: `runs/whyfix/gsm64b_*_traces.jsonl`, `runs/whyfix/replicate_results.json`, `runs/whyfix/replicate_sensitivity.json`.
