# `trace` test plan: how one word feeds another

Written on 2026-11-04 before any of these results. `why` showed which L19 words push an answer, and that 97% of top words sit on the last
token. That names the parts but doesn't show how the model computes. `brain.trace` asks the next question: for an L19 word, which L15
words feed it, and which earlier tokens of text feed those?

## Data

150 decisions from `runs/whytest/L19.jsonl` (test half), first in a fixed shuffle. Child word: the single L19 word with the largest measured
lead drop. Parents: L15 words ranked by gradient times activation for the child's pre-activation at its token, positive only.

## Arms (each removes 3 L15 words, reruns, and measures the child's pre-activation)

- **parents**: top 3.
- **active-same**: the 3 largest L15 activations on the child's token.
- **random-word**: 3 random active L15 words from the last 8 tokens.

| Bar | Needed |
|---|---|
| C1. parents lower the child more than each other arm | in at least 70% of decisions, CI low above 50% |
| C2. Removing each top-5 parent alone lowers the child | at least 80% of parents |
| C3. The path goes through the child: with 5 parents removed, holding the child at its original value undoes more of the lead drop than holding a random active L19 word on the same token | paired median difference with CI above 0, and the child wins in at least 60% of decisions |
| C4. Reading: given the child's name and two L15 names (top parent; a decoy active on the same token, activation at least half the parent's, whose removal moved the child under a tenth as much), a blind reader picks the real parent | at least 65%; floor with another trial's names within 45–55%, otherwise void |

Reported with no bar: for each child, the earlier token whose replacement (with a same-context neutral token) most lowers the child, and how often
that token isn't the last one. C3 is expected to be a small share per word, since one word is one of many paths. The bar compares against a control for that reason.

C4 runs only if C1 and C2 pass. Bars are fixed; changes go below, dated, before scoring.

Amendment, 2026-11-04, after C1–C3 were scored and before C4 was built: because R1's floor failed twice in the `why` test, C4 adds a third arm that keeps the two parent names and swaps only the child's name. It has no bar and shows whether the reader uses the child's name at all. C1 to C3 results: `runs/tracetest/mechanics.json`.

## Amendment, 2026-11-04, after C4 was scored

C4: real 39%, floor 75%, child swapped 28%. It fails its 65% bar and its floor is outside the gate, so it is void. The floor was badly built: it moved the whole parent-child triple to a different context, keeping the pairing it was meant to break.

Reading the three numbers together suggests a cause, which this new test checks. Readers who see the real context pick the word that fits the text, which is usually the decoy (child swapped: 72% decoy). Readers shown a triple out of context use the names, which point to the parent (75%).

**C4b.** The same 100 trials with no context shown, fresh readers. Real child name vs. another trial's child name. Only the child changes, so this is a true floor.
Bars: real at least 65%, swapped within 45–55%, otherwise void. A pass would mean names carry true information about how words connect, and that showing the text misleads readers.
