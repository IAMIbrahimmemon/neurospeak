# Blind-read test: can a neuron be read from its weights alone?

**Setup.** We sampled 60 random MLP neurons from Qwen3.5-0.8B: 20 early (L1–3), 20 middle (L11/13/15) and 20 late
(L19/20/22). For each neuron we built a dossier from the weights only. It lists the raw tokens that drive the neuron's
gate and up projections, and the next-tokens its output column promotes or suppresses (logit lens). The model was never
run on text for this step.

Six Claude subagents wrote predictions from the dossiers. We computed the answer key only after that: the model ran over
372k tokens (about 75% Wikipedia, 25% math problems and reasoning traces) and recorded each neuron's top positive and
negative contexts. The file timestamps show all predictions were written by 13:19:28 and `truth.json` at 13:25:26.

**Scoring.** Nine scorer subagents saw one description plus 5 sets of real contexts. One set was the true neuron's;
the other 4 came from other neurons in the same layer band. Each scorer ranked the 5 sets. A random pick is right
20% of the time. The scorers were not told which condition a trial came from:
- **blind:** the prediction written from weights only.
- **ceiling:** a description written from a disjoint half of the neuron's real contexts.
- **shuffled:** another neuron's blind prediction. This is the empirical floor.

| Condition | Early | Middle | Late | All (n=60) | 95% CI |
|---|---|---|---|---|---|
| **Blind (weights only)** | 45% | 70% | 70% | **62%** | 49–73% |
| Ceiling (saw real examples) | 100% | 100% | 95% | 98% | 91–100% |
| Shuffled (floor) | 20% | 20% | 20% | 20% | 12–32% |

All blind results beat 20% (exact binomial p < 0.01 for every band).

Reader self-rated confidence predicted success: confidence 4–5 was **10/10** correct, 3 was 11/16, 2 was 12/26,
and 1 was 4/8.

## What kind of reading this is

- **Topic or genre reading works.** N40 (L19) promotes "fundamental"/"basic", and its input favours "research" and
  "theoretical". The reader predicted "science/research contexts". The neuron actually fires on ` in` / ` of` inside
  chemistry prose. The topic was right and the token was wrong. Distractors from Wikipedia biographies or math traces
  were easy to rule out.
- **Sometimes it is precise.** N44 (L19) promotes `\frac`. The reader predicted "fires just before LaTeX fractions,
  negative on fraction words like 'one-third'". The truth: positive on `$B[[ =]] \frac{`, negative on
  `is[[ one]]-fourth` and `blue,[[ one]]-half`. Both signs were right, from weights alone.
- **The output side carries most of the signal.** Late and middle layers beat early ones (70% vs 45%). Early
  neurons fire on word fragments that depend on the previous token (`Cort[[és]]`, `Kur[[zt]]man`), and a
  single-token lens of the input weights can't see that. The output lens is meaningless that early.
- **The gap to the ceiling is large.** Example-based descriptions are nearly perfect (98%), and weights-only ones get
  62%. The weights give the gist of what a neuron does but not the specific trigger.

## Caveats

- 5-way multiple choice with distractors across genres is a coarse test. A neuron's topic alone often wins. A harder
  version would use distractors from the same genre, or score token-level activation prediction (simulation scoring).
- n=20 per band. The confidence intervals are wide.
- Readers and scorers are the same model family (Claude). Their shared biases are not controlled.
- The dossier tools (logit lens, input token lens) are themselves interpretability methods. "Weights only" here means
  "weights plus the embedding matrix", which is also a weight.

Files: `dossiers.json` (what the readers saw), `pred_blind_*.json`, `truth.json`, `pred_ceiling_*.json`,
`score_batch_*.json`, `scores_*.json`, `trials_key.json`, `results_rows.json`. Scripts: `build_dossiers.py` →
`scan_truth.py` → `make_trials.py` → `score.py`.
