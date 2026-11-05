# Word test on distilgpt2

Run on 2026-11-04 with the bars in [PLAN.md](PLAN.md). Same scripts and briefs as the Qwen run. The only code changes were two hard-coded layer lists in the scorers.

**Short version.** The results look much like Qwen's on a very different model: the words predict effects, they beat neurons, and they miss the 75% read bar in the same way.

| | distilgpt2 L4 | Qwen3.5-0.8B L15, L19 |
|---|---|---|
| Features tested | 100 | 200 |
| FVU, Wikipedia | 19.4% | 17.9%, 18.8% |
| FVU, GSM8K | 15.1% | 17.3%, 18.1% |
| Loss recovered | 93.5% | 97.2%, 95.9% |

| Bar | Needed | distilgpt2 | Qwen | Verdict for distilgpt2 |
|---|---|---|---|---|
| Read: balanced accuracy | ≥75% | 70.9% (CI 68.6–73.0) | 71.5% | **Fail** |
| Floor: shuffled labels | near 50% | 51.7% | 50.2% | OK |
| Better than neurons | CI above 0 | +8.0 (CI +3.7 to +12.0) | +13.0 | Pass |
| E1: steering, real sentence | ≥65% | 86.5% (CI 81.0–91.5) | 79.6% | Pass |
| Gate: null, floor | 45–55% | 53.0%, 45.5% | 50.3%, 48.7% | Pass, floor close to the edge |

Other numbers: strongest firings found 88%, weak firings 22%, same-token rejected 73%, same pattern as Qwen. Of 100 features, 78 got both steering trials right, 17 one, and 5 neither. Neurons' steering sentences scored 68.3% against a null of 60.0% on only 60 trials, so they can't be separated here either.

What this shows: the method's results carried over to a second architecture, in the one way tested. What it doesn't: distilgpt2 is a small, old model trained on web text. A pass on two small models doesn't mean any model. Scorers leaned to A a bit in the null (59%), though answers were balanced.

Scorers and labelers are Claude models. Several ran a `head` or used Bash to write their file, against the brief. They report judging from the batch file only, which can't be verified from here.
