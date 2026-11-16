# Study 18: our edit against MIT's, on the same problems

Written on 2026-11-06 before any data from these arms, and before study 17 was scored. We have never run self-doubt removal and Astra's weight edit on the same problems. Our results and theirs differ in cap, problems and sampling, so no comparison between them means anything yet.

**What their logs predict.** I rescored the 200 GSM8K samples in their published logs (`exp8_math/eval`) as if cut at each cap, using a plain last-box comparison. This is post hoc and on their problems.

| Cap | Their edit correct | Unedited correct | Their edit finished | Unedited finished |
|---:|---:|---:|---:|---:|
| 3,000 | 9.5% | 5.5% | 9.5% | 4.0% |
| 9,000 | 32.5% | 12.0% | 39.5% | 10.5% |
| 32,768 | 60.0% | 12.0% | 94.0% | 26.0% |

So at 9,000 their edit should show about half its final gain, and at 3,000 very little. Self-doubt removal scored 38% at 3,000 without their presence penalty, on different problems. That contrast is why this test is worth running, and it is the thing that could vanish once the settings match. A 32,768 cap would take about 11 hours per arm here, so it isn't practical.

**Runs.** GSM8K test rows 704 to 799, 96 unused problems. Seed 18, 9,000-token cap, thinking on, value-based scoring, the same prompt as theirs. Each run is also scored as if cut at 3,000 tokens by truncating its text, as in study 17.

**Their sampling.** Temperature 0.6, top_p 0.95, top_k 20 and presence penalty 1.5, which we have never used. `whyfix/generate.py --presence 1.5` subtracts 1.5 from the logits of every token already generated, before temperature. Prompt tokens aren't penalized. This is my reading of vLLM's behavior from memory, and I can't run vLLM here to confirm it. The arm `none` keeps our usual sampling.

**Arms**, run in this order so a stop leaves the most useful ones.

| Arm | Weights | Run-time edit | Presence penalty |
|---|---|---|---|
| `astra_pp` | Astra's edit (`data/astra_p50c_edit.npz`) | none | 1.5 |
| `doubt_pp` | base | self-doubt removed at L19 | 1.5 |
| `astra_doubt_pp` | Astra's edit | self-doubt removed at L19 | 1.5 |
| `none_pp` | base | none | 1.5 |
| `none` | base | none | 0 |

| Bar | Needed | Why |
|---|---|---|
| H1. Their fix reproduces | correct at 9,000, `astra_pp` minus `none_pp`, at least +10 points, CI above 0 | is our setup able to show their result? Predicted about +20 |
| H2. Self-doubt adds to their fix | correct at 9,000, `astra_doubt_pp` minus `astra_pp`, at least +10, CI above 0 | does our lead add anything beyond their edit? |
| H3. Faster | correct at 3,000, `doubt_pp` minus `astra_pp`, at least +10, CI above 0 | the contrast above, with settings matched |
| H4. Who wins given time | correct at 9,000, `doubt_pp` minus `astra_pp`, CI excludes 0 | signed; a CI across 0 is reported as no winner shown |
| H5. Our lead holds under their sampling | correct at 3,000, `doubt_pp` minus `none_pp`, at least +15, CI above 0 | everything of ours was without the penalty |
| H6. What the penalty does alone | correct at 9,000, `none_pp` minus `none`, CI above 0 | is the setting itself part of their baseline? |
| H7. Intact | a blind Claude reader judges 20 sampled excerpts per arm; at most 2 of 20 garbled | a presence penalty can push text toward odd words. Bars using an arm that fails are void |

The reader judges excerpts from the middle of each run, labelled coherent, repetitive or garbled, not knowing the arm. The study 14 broken count is also reported, but carries no bar, because study 16 showed it mostly restates looping.

**How to read it.**

- H1 fails: this setup doesn't reproduce their edit at this cap, and H2 to H4 can't speak to it. Report that first.
- H1 and H2 pass: self-doubt adds something their edit doesn't contain, which supports a second lever beyond their one direction.
- H1 passes, H2 fails: at this size, no added value is shown on top of their fix. With the CI top under +10 it reads as their edit already covering what self-doubt removal does.
- H3 passes, H4 shows no winner or favors theirs: self-doubt removal gets answers sooner and their edit catches up with time. Speed, not capability.
- H3 and H4 both favor self-doubt: our edit is better at this cap on matched settings.
- H5 fails: our gain depended on our sampling, and every earlier number needs rereading.

Also with no bars: finished, boxed, right-when-boxed, loops by the study 14 rule, tokens per run, and the bar differences among runs that finished.

**Limits known in advance.** One seed, one model, 96 problems, no correction across bars. Their edit is a permanent weight change and ours a run-time removal, so this compares two edits and not two methods of finding edits. Self-doubt's direction was built on the unedited model. `astra_doubt_pp` applies it to edited weights without rebuilding it. At 9,000 their edit shows about half its gain, so H4 is not a verdict on their full result. Our presence penalty may differ in detail from vLLM's. There's no `astra` arm without the penalty, so an effect of the penalty on their edit can't be separated.

## Amendment, 2026-11-06, after a Fable review and before any arm

These override anything above that disagrees. Simulated by the reviewer at n=96 with paired bootstrap CIs, using the discordance seen in earlier studies. I have not rerun the simulation.

**Why the bars change.** "At least +10 and CI above 0" passes only about 40% of the time when the true effect is +10, 14% at +5 and 78% at +15. Study 14's closest analogue was +9.4. So H2 as written would likely fail even if it were true, and a CI half-width of about 10 points can't show their edit "already covers" self-doubt.

| Bar | Now needed | Reading of a fail |
|---|---|---|
| H1. Their fix reproduces | unchanged: correct at 9,000, `astra_pp` minus `none_pp`, at least +10, CI above 0. Power about 1.0 at the predicted +21 | |
| H2. Self-doubt adds to their fix | CI above 0. A difference of at least +10 is reported separately as the practically useful size | not shown. No claim that their edit covers it |
| H3. Faster | correct at 3,000, `doubt_pp` minus `astra_pp`, CI above 0 | |
| H4. Who wins given time | unchanged and signed. At a true +8 this excludes 0 only about 28% of the time, so "no winner" is the likely outcome whatever the truth | no winner shown |
| H5. Our lead holds | unchanged. Power 0.82 to 1.0 at prior effect sizes | our gain depends on the sampling, or the penalty raised the baseline. This design can't separate the two, because there's no `doubt` arm without the penalty |
| H6. Penalty alone | no bar. Reported as an estimate, since their logs predict about +6 and power is about 44% | |
| H7. Intact | unchanged | |
| H8. Control | correct at 3,000, `doubt_pp` minus `matched_pp`, CI above 0 | new. Every earlier study had a matched random control |

**New arm.** `matched_pp`, removing `dir_self_doubt_matched` at L19 with the penalty.

**New order.** `astra_pp`, `none_pp`, `doubt_pp`, `astra_doubt_pp`, `matched_pp`, `none`. The first two judge H1, which gates H2 to H4, so they go first.

**Readings replaced.** "H1 passes, H2 fails" now reads as "no added value shown at this size". The rest stand.

**Corrections to the table above.** The reviewer's recount using our own scorer gives correct, edit and unedited, of 10.0 and 6.0 at 3,000, 33.5 and 12.5 at 9,000, and 61.0 and 12.5 at 32,768, within a point of mine. "Finished at 32,768: 94.0 and 26.0" was wrong, because re-encoding the text drifts past the cap. With a 50-token margin it is 90.5 and 12.0, matching their recorded 92.8 and 12.5. The gain over unedited is +4 at 3,000, +21 at 9,000 and +48.5 at 32,768. The "38% at 3,000" for self-doubt mixes earlier runs that ranged from 27% to 44%.

**Limits added.** 9,000 tokens gives their edit about 43% of its final gain, and their median run is 11,200 tokens. So H4 says nothing about their full result. The penalty matches vLLM's order as I remember it, and the reviewer also could not check vLLM's source here.

