# Word test results

Run on 2026-11-03 on Qwen3.5-0.8B, using the bars in [PLAN.md](PLAN.md), which were written before labeling.

## Dictionary quality on held-out text

| | L15 | L19 |
|---|---|---|
| Variance not reconstructed (FVU), Wikipedia | 17.9% | 18.8% |
| FVU, GSM8K | 17.3% | 18.1% |
| FVU, reasoning traces (also in training) | 24.9% | 23.5% |
| Next-token loss with the reconstruction swapped in (original 2.77 nats) | 2.96 | 3.00 |
| Loss recovered, relative to mean-only ablation | 97.2% | 95.9% |
| Features active at least once / eligible for the test (of 16,384) | 16,129 / 12,528 | 16,140 / 12,149 |

## Pass bars

| Bar | Needed | Result | Verdict |
|---|---|---|---|
| Read: balanced accuracy, real labels | at least 75% | 71.5% (CI 69.9–73.2) | **Fail** |
| Floor: shuffled labels | near 50% | 50.2% | OK |
| Hard read: same-token negatives rejected | at least 65% | 83% | Pass |
| Better than neurons | CI above 0 | +13.0% (CI +9.3 to +16.4) | Pass |
| Causal strength: beats a random direction | at least 80% of units | 84% | Pass |
| Causal strength: beats another word's direction | at least 50% of units | 79% | Pass |
| Causal reading: two-list test | at least 65% | 93% | **Passed, but void** (see below, replaced by the effect test) |

| Real labels | n | Balanced | Strongest firings found | Weak firings found | Same-token rejected | Random rejected |
|---|---|---|---|---|---|---|
| SAE features | 200 | 71.5% | 84% | 18% | 83% | 99% |
| MLP neurons | 60 | 58.5% | 50% | 4% | 86% | 92% |

## The causal reading test does not measure what it claimed

The list test passed its bar at 93%. A control added after the results (not in the plan) used the same labels with lists from
removing a **random direction** at the same tokens. Scorers did just as well: 184/198 for features and 50/60 for neurons,
the same totals as with the real lists. That tie is a coincidence: the two runs disagree on 18 feature units and 14 neuron units, evenly split in each direction. The real and random lists share about 3 of 8 tokens.

So the lists mostly show what comes next in that context, not what the unit does. This test gives no evidence that a
label predicts a unit's effect. A valid version must compare against the context's own likely next tokens.

Causal strength stands as measured: removing a feature moved the next token distribution by a median KL of 0.030 nats, against 0.016
for a random direction and 0.017 for another word's direction. This shows that the direction matters at those tokens, not that
its label is right.

## Effect test (replaces the void test)

Plan and bars: [PLAN_EFFECT.md](PLAN_EFFECT.md). Scorers see only a label's effect sentence, and pick between two token lists from the
same contexts. Random-direction lists act as a built-in check on the test.

| Steering replication, 2 trials per unit, A/B balanced | Features (398 trials) | Neurons (120 trials) |
|---|---|---|
| Real effect sentence | **79.6%** (CI 75.1–84.2) | 58.3% (CI 48.3–67.5) |
| Null: random-direction lists | 50.3% (CI 45.2–55.3) | 57.5% (CI 47.5–66.7) |
| Floor: another unit's sentence | 48.7% (CI 44.0–53.8) | 44.2% (CI 35.0–52.5) |

| Bar for features | Needed | Result | Verdict |
|---|---|---|---|
| Gate: null and floor | 45–55% | 50.3%, 48.7% | Pass |
| E1: steering, real sentence | at least 65% | 79.6% | Pass (replication) |
| E2: removal vs a random direction at the same tokens | at least 60% | 60.4% (CI 53.6–67.2), floor 49.0% | Pass, barely |
| E3: list stability beats random | interval above 0 | 5.24 vs 4.21 of 8 tokens (difference CI 0.78–1.29) | Pass |

How the first run went:

- The first run of steering scored 71.9%, but its floor scored 57.8%, outside the gate, so the plan's rule voided it. The replication
  was declared before scoring, with the same bars.
- Scorers leaned hard to option A when nothing separated the lists: 74–76% in the first run. The replication balanced answers exactly
  and told scorers A and B are equally likely. This change may also explain part of the rise from 71.9% to 79.6%.
- E2 was not replicated with balanced answers. In its real condition, 55% of correct answers were A and scorers said A 58% of the time,
  which may inflate it slightly above its 60% bar. Treat it as weak.

What this does and doesn't show:

- A feature's effect sentence, written with no sight of these lists, usually picks the right one of two. That is evidence that
  words predict what features do. For neurons, real sentences (58.3%) can't be told from the null (57.5%).
- It is a two-way choice. It doesn't show that a sentence is precise, or complete.
- Effects are one token ahead, at one strength, chosen in a pilot that looked only at stability.
- 19 of 199 features failed both trials, and 43 failed one. Some words are wrong: L19.f3890's sentence says it predicts "as",
  and its list holds "ing, ings, ating".
- 10% of feature lists repeat a token the feature fires on. Those few score no worse, and the rest are not explained by echoing.

## Other exploratory findings, not in the plan

- Scored only on strongest firings and all negatives, features reach 87.6% (CI 85.3–89.9) and neurons 69.8% (CI 64.2–75.4).
  Labels describe a feature's strong firings. Its weak half is mostly unexplained: either labels miss something or weak firings
  are noise. This test can't tell which.
- Reader confidence predicts accuracy: 50% at confidence 1, 76% at confidence 5.

## Limits

- 200 of about 24,700 eligible features. About 25% of features were not eligible, mostly rare ones, so none of this covers them.
- Labelers and scorers are the same model family. Each labeler ran a harmless size check on its batch, against the brief.
- The neuron threshold (top 1%) is a choice. Neurons are also judged on positive firings only.
- Labels are from one reader, with no second opinion per unit.
- No generation test: nothing here shows what happens to whole answers.
- Scorers got no second opinion per trial, apart from two trials per unit in the replication.
- One model. No other model has been through this test.

Scripts: `scan.py` → `build.py` → labelers → `make_trials.py` → scorers → `score.py`; `causal_control.py` for the control;
`effect.py` → `make_effect_trials.py` → `score_effect.py`, and `make_effect_replication.py` → `score_replication.py` for the effect test.
Data: `runs/wordtest/` holds units, labels, scores, keys, `results.json`, and `control_result.json`.
