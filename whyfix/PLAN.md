# Can `why` find a fix for a real failure?

Written on 2026-11-04 before any of these results. `why`'s words are causally real, but their names didn't help readers. This asks a practical question that needs no name:
if `why` points at the words behind a real failure, does removing them fix it?

The failure: Qwen3.5-0.8B falls into reasoning loops. Table to beat, from the README: repeated 8-grams 45.9% unedited, 11.4% with the hand-written concept edit, 11.8% with Astra's, 46.1% with a random control.

## Finding the words

Only from problems not used for the test: `data/practice_sample.jsonl` lines 17–24 of the base run, plus new unedited traces for lines 25–48. A loop onset is a token where an 8-gram repeating
an earlier one begins (at most 3 per trace, at least 200 tokens apart). `why` runs on each onset for the repeated token's log probability, on the last 512 tokens, at L19.
**why words**: the 16 words with the largest total positive attribution across onsets.
Controls: **active words**, the 16 words most often active at onsets; **random words**, 16 random words matched to the why words' firing rates in `runs/wordtest/scan.pt`.

## The test

The first 16 problems, same seed, 3,000-token cap, same sampling as the README table. An arm removes its 16 words from L19 at every token during generation.

| Bar | Needed |
|---|---|
| F1. Fewer loops | why words reach at most 30.6% repeated 8-grams, two thirds of unedited |
| F2. Not just any words | why words at least 10 points below random words |
| F3. Beats a naive tool | why words below active words |
| F4. Accuracy holds | at most 1 of 16 lost from the unedited 3 of 16 |

With 16 problems, this is a strong signal at most, as in the README. If F1 passes, the words' names and evidence get reported, read only after the result. Bars are fixed; changes go below, dated, before generation.

## Amendment, 2026-11-04, before mining or generation

The edit diff found that L19.f976 falls most under both real loop edits (0.80 to 0.05 per token under Astra's), and it fires heavily on repeated text. A new arm tests it alone: **f976 only**, removing just that word during generation.

| Bar | Needed |
|---|---|
| F5. One word is enough | f976 alone reaches at most 30.6% repeated 8-grams |

The same arm also tells whether a word found by reading an edit agrees with one found by `why`: report whether f976 is among the why words.

## Amendment, 2026-11-04, after the first results

Active words reached 27.1%, under the 30.6% mark, with no bar of their own. Before believing it, repeat it with a new seed (1), on the same 16 problems, with unedited and random-word arms at that seed. New arm **diff words**: the 16 words falling most under Astra's edit, measured on the 24 mining traces, not the test problems.

| Bar, at seed 1 | Needed |
|---|---|
| F6. Active words reach two thirds of that seed's unedited rate or less | yes, and at least 10 points below random words |
| F7. Diff words do the same | yes, and at least 10 points below random words |

## Amendment, 2026-11-04, after seed 1

A last arm. Loops look like a state and not a token, so instead of asking which words pushed one token, ask which words are active more around loop onsets than in the same traces well before them. `brain.contrast(loop_texts, other_texts)` gives **contrast words**: the top 16 by difference in activation, on mining traces only. Seed 2, with unedited and random-word arms.

| Bar, at seed 2 | Needed |
|---|---|
| F8. Contrast words | at most two thirds of that seed's unedited rate, and at least 10 points below random words |

This arm was stopped before any result: see the next amendment.

## Amendment, 2026-11-04: the onset detector was wrong

Found before any contrast result. The onsets were any repeated 8-gram, and 80% of them repeat 8 tokens or fewer, mostly quoted problem text. The earlier arms' words are not about loops. Their results stand as measured, and don't test the plan's question.

**New onset.** A token p, after the prompt, where at least 48 of the next 64 8-grams have already appeared. At most 3 per trace, at least 200 tokens apart. Before running: report how many, and read a sample.

**New words**, all on mining traces only: **why2** (same method on new onsets), **active2** (most often active at new onsets), **contrast2** (active more in the 128 tokens before new onsets than in text at least 128 tokens earlier), **random2** matched to why2's firing rates.

Seed 2, same 16 problems, arms: unedited, why2, active2, contrast2, random2.

| Bar, at seed 2 | Needed |
|---|---|
| F9. why2 | at most two thirds of unedited, and at least 10 points below random2 |
| F10. active2 | same |
| F11. contrast2 | same |

If none passes, the concept edit stays the recommended fix and this line of tests stops.
