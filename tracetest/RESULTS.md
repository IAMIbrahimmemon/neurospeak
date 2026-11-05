# `trace` test results

Run on 2026-11-04 on Qwen3.5-0.8B. Bars are in [PLAN.md](PLAN.md), written before these results, with dated amendments.

**Short version.** The connections are real when measured by removal. No valid test shows that their names tell a reader which word feeds which.

## Mechanics, on 150 decisions

For each decision, the child is the L19 word that moved the lead most, and parents are the L15 words that `trace` says feed it.

| Bar | Needed | Result | Verdict |
|---|---|---|---|
| C1. Removing 3 parents lowers the child more than the 3 largest activations on its token | ≥70%, CI low above 50% | 95% (91–98) | Pass |
| C1. Same, against 3 random active words | same | 100% | Pass |
| C2. Removing each top-5 parent alone lowers the child | ≥80% | 100% | Pass |
| C3. Holding the child fixed undoes more of the lead drop than holding a random word | CI above 0, child wins ≥60% | 0.65 vs 0.00 logits (CI on the difference 0.58–0.75), wins 99% | Pass |

Removing 3 parents lowers the child's activation by a median 55% (3 largest activations: 19%; random: 0%). Removing 5 parents drops the lead by a median 0.84 logits, and holding the child at its original value returns 0.65 of that. So one L19 word carries most of the effect of those parents. These parents were chosen for this child, which makes C3 partly built in.

No bar: 99% of children and 91% of top parents sit on the last token, and only 8% of children have an earlier token as their strongest source of text. `trace` mostly shows a chain within one token's computation. It doesn't yet show how the model uses earlier text, and token replacement is a blunt way to ask.

## Reading: which of two L15 words feeds this L19 word?

| Build | Real names | Control | Verdict |
|---|---|---|---|
| C4, with the text shown | 39% (CI 30–49) | Whole triple in the wrong text: 75%. Child's name swapped: 28% | **Void**: floor out of range, and real failed its 65% bar anyway |
| C4b, names only | 71% (CI 62–80) | Child's name swapped: 63% (CI 53–72) | **Void**: floor out of range |

I stopped there. Three versions of this kind of reading test now have floors that won't hold, and more variants would be fishing for a pass.

What the numbers suggest, as a hypothesis only:

- Showing the text misleads. The same parent-decoy pairs drop from 63% with no text to 28% with the real text and an unrelated child name. The decoy usually fits the text better, because it's named for what's on the page.
- Parents look like parents regardless of the child, since 63% holds with an unrelated child. The child's name adds about 8 points at most, and that gap isn't clearly above noise.
- What a word's name describes and what a word does for a later word aren't the same thing. One hand-checked trial: the later word fires on "times", and the true parent's name describes indentation in reasoning text, while the decoy's name is about the word "times". A reader picks the decoy, and removal says otherwise.

That last point agrees with the `why` result: the names describe where a word fires, and readers can't get the causal structure from them.

Data: `runs/tracetest/`. Scripts: `tracetest/`. One label file had a missing quote, fixed by hand with no change in content.
