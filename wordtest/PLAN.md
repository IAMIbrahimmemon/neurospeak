# Word test: do SAE features work as words a human can check?

Written on 2026-11-03, before any feature was labeled or scored. The results go in RESULTS.md. This file does not change.

## Setup

- Units: 100 random eligible SAE features from each of L15 and L19, and 30 random MLP neurons from each layer as the comparison.
- Text: held-out WikiText validation and test, held-out GSM8K test, and reasoning traces. The traces were also in SAE training.
- Every article, problem, or trace prompt lands wholly on one side of a split: a labeling half and a test half.
- Active tokens: a feature's nonzero tokens, a neuron's top 1%. Inactive tokens: a feature's zeros, a neuron's below-median values.
- Labelers see only labeling-half contexts and don't know whether a unit is a feature or a neuron.
- Scorers see one label and 8 test-half snippets: 4 active (2 strongest, 2 weaker), 2 inactive on the same token as an active
  snippet (a hard test: the token alone can't decide it), and 2 inactive at random from the same source.
- A causal test removes the unit's direction at its token and records the next-token probabilities that move.
  Scorers pick which of two token lists fits the label.
- Floor: the same trials with another unit's label. Scorers can't tell real trials from floor trials.

## Pass bars for SAE features

1. **Read.** Mean balanced accuracy of at least 75% with real labels, near 50% with shuffled labels.
2. **Hard read.** At least 65% of same-token negatives correctly rejected with real labels.
3. **Better than neurons.** The 95% bootstrap interval of (feature minus neuron) balanced accuracy excludes zero.
4. **Causal strength.** Removing the word moves the next-token distribution more than removing a random direction of the same
   size for at least 80% of units, and more than another word's direction for at least 50%.
5. **Causal reading.** At least 65% correct on the two-list test with real labels, near 50% with shuffled labels.

## Limits known in advance

- Labelers and scorers are the same model family, so shared biases are not controlled.
- Removal is measured only at the next token.
- A neuron's "active" threshold is a choice that a feature's zero doesn't need.
