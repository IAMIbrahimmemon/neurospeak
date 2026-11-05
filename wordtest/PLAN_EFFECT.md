# Effect test: does a word predict what its unit does?

Written on 2026-11-03, before any label was scored on this test. It replaces the void causal reading test in RESULTS.md, which
used lists from different contexts and passed even with random-direction lists.

A pilot of 15 features and 15 neurons, looking only at list stability and never at labels, set the method: strength 3 times
a unit's typical strong firing, and lists from mean log-probability change on tokens seen at least 20 times in the corpus.

## Test

Same units and same labels as before. The labelers never saw these lists. Scorers see only the label's `effect` sentence, because
that is its claim about what the unit does, and they pick which of two token lists fits it.

| Condition | Effect sentence from | List A vs list B | What it tests |
|---|---|---|---|
| steer real | this unit | added this unit vs added another unit, same neutral contexts | **the main test** |
| steer null | this unit | random direction vs another random direction, same contexts | validity: must be chance |
| steer shuffled | a third unit | as steer real | floor |
| remove real | this unit | removed this unit vs removed a random direction, same contexts, shared tokens dropped | effect where it fires |
| remove shuffled | a third unit | as remove real | floor, and shows any bias toward real-unit lists |

Scorers can't tell the conditions apart.

## Bars for SAE features

- **Validity gate.** Steer null, steer shuffled and remove shuffled all between 45% and 55%. If any is outside, the matching main
  result is void.
- **E1.** Steer real at least 65%.
- **E2.** Remove real at least 60%.
- **E3.** A feature's split-half list overlap beats a random direction's, with a 95% paired interval above 0.

Reported without a bar: features vs neurons, and how often a list repeats the tokens a unit fires on. A unit that only "writes
its own token" is a weaker kind of effect than one that changes what comes next.

## Amendment, before the replication

Added after the first scoring, without changing the bars. Steer shuffled scored 57.8% (n=199), outside the 45–55% gate, so under the
rule above the steer real result of 71.9% is void. At this n, chance alone falls outside that range fairly often, but the
gate stays as written. The steering conditions are run again: new distractors, new third labels, new scorers, two trials per unit with
A/B exactly balanced. E1 is judged on this replication alone, using the same gate and the same 65% bar.
