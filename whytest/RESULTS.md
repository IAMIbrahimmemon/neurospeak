# `why` test results

Run on 2026-11-04 on Qwen3.5-0.8B. Bars are in [PLAN.md](PLAN.md), written before these results, with two dated amendments.

**Short version.** `why` finds the words that really move a decision. Nothing shows yet that those words' names help a reader. The mechanics pass clearly, and the test of whether names help a reader predict the model failed twice.

## Mechanics: does removing the listed words change the decision?

300 decisions per layer from text no labeler read. Each arm removes 5 words at one layer.

| | L19 | L15 |
|---|---|---|
| Top choice changes, why's words | **83%** | **66%** |
| Top choice changes, 5 largest activations on the last token | 48% | 44% |
| Top choice changes, 5 largest activations anywhere | 2% | 2% |
| Top choice changes, 5 random active words | 4% | 1% |
| Top choice changes, random direction at why's positions | 9% | 5% |
| Median lead drop, why (median lead 1.03 logits) | 2.30 | 1.41 |
| Median lead drop, random direction | −0.03 | +0.02 |

| Bar | Needed | L19 | L15 | Verdict |
|---|---|---|---|---|
| M1. Beats active-last | ≥70%, CI low above 50% | 93% (90–96) | 91% (87–94) | Pass |
| M1. Beats active-any | same | 100% | 100% | Pass |
| M1. Beats random-word | same | 100% | 100% | Pass |
| M2. Changes the top choice, at least 3× random direction | ≥30% | 83% vs 9% | 66% vs 5% | Pass |
| M3. Removing a listed word lowers the lead | ≥80% of words | 99.4% | 99.2% | Pass |
| M4. Estimate ranks measured effect (Spearman) | median ≥0.5 | 0.95 | 0.93 | Pass |

How many words does it take? On L19, removing 1 word changes the top choice in 46% of decisions, 2 in 62%, 5 in 83%, 10 in 90%, 20 in 92%, 50 in 94%. For decisions with a lead above 2 logits, 5 words change the choice 57% of the time at L19 and 22% at L15.

Limits of these numbers:

- These bars were easy. The pilot changed 20 of 20 decisions, and the attribution is built from the same gradient that removal then follows. So they show the list is real, not that it's a good explanation.
- A flip can be close to adversarial: a median lead of 1 logit doesn't take much to overturn.
- 97% of top words at L19 sit on the last token. `why` says what the model's state was at that moment, and rarely reaches back to earlier text. `trace` exists for this.

## Reading: do the names tell a reader anything?

| Test | Bar | Result | Verdict |
|---|---|---|---|
| R2. Predict the model's top choice: real sentences vs no words | +5 points, CI above 0 | 75% vs 72%, +3 (CI −6 to +13) | **Fail** |
| R3, new decisions. Measured token lists vs no words | +5 points, CI above 0 | 58% vs 62%, −4 (CI −13 to +5) | **Fail** |
| R3. R2 repeated: sentences vs no words | reported | 58% vs 62%, −4 (CI −14 to +6) | Same result as R2 |
| R3. Another decision's lists vs no words | within 5 points | 60% vs 62%, −2 (CI −11 to +7) | OK |
| R1. Pick the real driver from two names | ≥65%, floor 45–55% | 76%, floor 59% | **Void** |
| R1b, matched on labeler confidence | same | 76.8%, floor 59.6% | **No result**, gate failed again |

So word names, as they stand, don't help a reader predict the model. That's true of sentences and of measured effect lists, on two sets of decisions. Reader scores differ between rounds (72% vs 62% with no words), so these decisions vary a lot in difficulty. A real effect of a few points can't be ruled out at this sample size.

R1 is the one positive hint: real names beat shuffled names by 17 points in both runs. I can't count it. Both floors came in at about 60%, above the gate, so drivers must look more like drivers than decoys do whatever the context. The gate failing twice with almost the same numbers suggests it's built into the test and isn't chance. This wasn't investigated further.

A simple automatic reader (no bar) that adds up each word's general effect on the two options chose right 60% of the time, against 50% for another decision's words, on 100 decisions. So general effect lists carry a little signal about a specific decision, and less than the context alone gives a reader.

## What this means for `why`

`why` is a reliable pointer to what to change: remove these words and the answer moves. It reports the measured push on each word, which needs no name. Reading the list as a story isn't supported. A plausible story from `why`'s word names shouldn't be trusted without checking it by removal.

Readers and labelers are all Claude models. At least six labelers in round 1 ran a harmless size check or `head` on their batch, against the brief. Round 2 briefs repeated the rule.
Data: `runs/whytest/`. Scripts: `whytest/`. Names now exist for 953 L19 words in `runs/words/`.
