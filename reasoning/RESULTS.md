# Reasoning dictionary results

Run on 2026-11-04, using the bars in [PLAN.md](PLAN.md), which were written before training. Both dictionaries are scored by `reasoning/score.py`.

| Variance not reconstructed at L19 | Old dictionary (12M tokens) | New dictionary (6M tokens, half traces) | Bar | Verdict |
|---|---|---|---|---|
| R-1. 24 unseen traces | 26.6% | 22.4% | at most 22% | **Fail**, narrowly |
| R-2. Held-out Wikipedia | 18.8% | 23.9% | at most 21% | **Fail** |

**The old dictionary stays the default.** The new one is at `runs/sae/Qwen__Qwen3.5-0.8B/reasoning/L19.pt`, with no named words.

What this shows:

- More trace text does help on reasoning, by 4.2 points, and that came with half the training tokens.
- It cost 5.1 points on Wikipedia, more than the plan expected. Half the tokens and half the Wikipedia share means about 3M Wikipedia tokens against about 10M, so this cost is expected to shrink with more training. That is untested.
- It doesn't show that a dictionary can't do both. Training was stopped at the planned 6M tokens, and its running trace-and-Wikipedia FVU was still falling.

Limits. Traces came from 216 problems, not the planned 252, because generation ran slow. A pause of about 9 hours and other jobs on the same GPU don't affect the result, though Adam's state did survive the pause because the process was frozen, not restarted. I am not retraining to clear these bars, since that would be choosing a stopping point after seeing the result. A longer run would be a new test with its own bars.
