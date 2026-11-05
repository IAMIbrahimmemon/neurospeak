# `why` test plan

Written on 2026-11-04 before any result on these decisions. Only prompts typed by hand were looked at first ("The capital of France is"
and two others), to check that the code ran.

**The claim.** `brain.why(text)` lists dictionary words that pushed one next-token decision. Two questions decide if it opens the box:
are those words what really drove the decision (mechanics), and do their human-language names tell a reader anything true (reading)?

## Decisions

300 decision points per layer, from the test half of `runs/wordtest/scan.pt` (the half no labeler read): 150 wiki, 75 GSM8K, 75 traces.
Context is the 128 tokens before the point. Kept only if the model's top choice has p between 0.1 and 0.9. "Lead" is the logit of the top
choice minus the logit of the runner-up. The model runs in float32 here because bfloat16 rounds logits to about 0.12.

## Mechanics: what removal does to the lead

Each arm removes 5 words from the residual and reruns the model.

- **why**: top 5 by gradient times activation for the lead, positive only.
- **active-last**: the 5 largest activations on the last token. This is what a plain "active features" tool would show.
- **active-any**: the 5 largest activations anywhere in the context.
- **random-word**: 5 random active words from the last 8 tokens.
- **random-dir**: at why's positions and sizes, a random direction instead of the word.

| Bar | Needed |
|---|---|
| M1. why drops the lead more than each of active-last, active-any, random-word | in at least 70% of decisions, each with CI low above 50% |
| M2. why changes the top choice | at least 30% of decisions, and at least 3 times random-dir |
| M3. Listed words are real: removing each of the top 8 alone lowers the lead | at least 80% of words |
| M4. The estimate ranks them: Spearman of estimate vs measured over the top 8 | median at least 0.5 |

Reported with no bar: flips and lead drop when removing 1, 2, 5, 10, 20, and 50 words, which says how many words an honest explanation
needs. Also each result by source and by lead size. Failing any bar is reported as a failure of `why`, and the code's output would have to say so.

## Reading: do the names mean anything?

Run only if M1 and M3 pass, on L19, with labels written blind by Claude subagents from the labeling half using the existing brief.

**R1. Which word caused it?** A reader sees the context, the answer, the alternative, and two names: the word whose removal lowered the
lead most (at least 0.25) and a decoy active on the same token, of at least half the driver's activation, whose removal moved the lead
under a tenth as much. Pick the driver. A/B balanced.
Floor: same trials with names swapped for another unit's. Bars: at least 65%, floor within 45–55%, otherwise void.

**R2. Can a reader predict the model better?** A reader sees the context and the model's top two choices and picks the model's choice.
Three arms per decision, with different readers: no words; the real words (top 5 by absolute attribution for the lead, sign hidden, so
selection doesn't leak the answer); another decision's words. Bars: real at least 5 points above no-words with a paired CI above 0,
and another-decision's words within 5 points of no-words.

Bars are fixed. Any change goes in an amendment below, dated, with its reason, before scoring.

## Limits known in advance

One token ahead. Removal at one layer. Claude readers only. Removal can overstate a word's role if the model never sees that kind of residual.

## Amendment, 2026-11-04, after R1 and R2 were scored

R1 scored 76% with its floor at 59%, so it is void. R2 failed: real words 75%, no words 72%, difference +3 (CI −6 to +13). Neither result changes.
New tests on decisions not used so far, with fresh readers:

**R1b.** R1 again, with driver and decoy matched on the labeler's confidence, 120 trials. Same bars and gate. If the gate fails again, R1 has no result.

**R3. Do measured effects carry what sentences don't?** A word's measured effect list may tell a reader more than a sentence written from its firings. Each
word's list is the 8 tokens it most raises and the 8 it most lowers when added at 20 shared neutral contexts from the labeling half, at 3 times its typical strong firing, so the list doesn't depend on this decision.
Same reader task as R2, 100 decisions, four arms per decision with different readers: no words; sentences (R2 repeated); lists; another decision's lists.
Bars: lists at least 5 points above no words with a paired CI above 0; another decision's lists within 5 points of no words. Sentences minus no words is reported as R2's repeat.

Also with no bar: a simple automatic reader that sums each word's activation times (effect on A minus effect on B) over the same lists and picks the higher.

Second amendment, same day, before any R1b reader ran: exact confidence matching left only 43 of 120 trials, too few to test the bar. Matching is relaxed to within 1 confidence point, preferring an exact match. Results for the exact-match trials are also reported.
