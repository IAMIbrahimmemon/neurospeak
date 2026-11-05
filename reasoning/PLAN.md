# A dictionary that covers reasoning text

Written on 2026-11-04 before retraining. `why` and `trace` matter most on reasoning text, where the model loops, and the dictionary is weakest there. Measured tonight on 24 traces it has never seen: FVU 25.9% at L19 and 26.6% at L15, against about 18% on Wikipedia. The old 24% figure was partly on traces in the training set.

**Plan.** Generate unedited traces for problems 49–300 of `data/practice_sample.jsonl`, which gives about 750k new tokens. Train a new L19 dictionary with the same settings, from scratch, 6M tokens, half from traces and half from the existing mix. The old dictionary is kept: the new one goes to `runs/sae/Qwen__Qwen3.5-0.8B/reasoning/L19.pt`.
The 24 unseen traces in `runs/whyfix/mine_traces.jsonl`, and the 16 test problems, stay out of training.

| Bar | Needed | Today |
|---|---|---|
| R-1. FVU on the 24 unseen traces | at most 22% | 25.9% |
| R-2. FVU on the Wikipedia held-out text | at most 21% | 18.8% |

The trade-off is expected: half as many tokens in total and more of them on reasoning. If R-1 fails, the likely reason is too little trace text, and the old dictionary stays the default either way until words are named in the new one.

## Amendment, 2026-11-04, before training

`reasoning/score.py` is the ruler for both dictionaries. On the old dictionary it gives 18.8% on Wikipedia, matching the word test, and 26.6% on the 24 unseen traces, not the 25.9% written above. The earlier figure came from a one-off measurement. The bars do not change: R-1 at most 22%, R-2 at most 21%.

Trace generation is slower than planned. Training starts at 07:30 with whichever trace files have finished, from problems 49 on, and the results will say how many that was. Training uses half traces and half Wikipedia (`--wiki-share 0.5 --no-gsm8k`), not "the existing mix", which is a small change from the plan.
