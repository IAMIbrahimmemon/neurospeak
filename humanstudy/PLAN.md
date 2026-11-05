# Human study plan

Written on 2026-11-04. Not run yet. Every reader so far has been a Claude model, and the goal is opening the box to people, so this is the test that decides whether any of it reaches people.

**Question.** Can a person predict a language model's next word better with `why`'s word notes than without them? On Claude readers the answer has been no (whytest/RESULTS.md). People may differ.

## Design

- Open `humanstudy/study.html` in any browser. It needs no server and sends nothing anywhere. At the end it downloads one JSON file, which the participant sends back.
- 40 trials, about 15 minutes. 30 **predict** trials: pick the model's top choice of two words. Three conditions with 10 trials each: no notes; the real notes; another decision's notes. Which trial gets which condition rotates by participant number, so across 3 people every trial appears in every condition. 10 **cause** trials: pick which of two units really caused the model's choice.
- Trials are from the test half of the corpus and the same decisions Claude readers saw (`runs/whytest/reading.json`), so the two kinds of reader can be compared.
- Recruit at least 12 people. No expertise needed. Record whether each person has worked with neural networks, which the page asks.

## Bars, set before any person takes it

| Bar | Needed |
|---|---|
| H1. Real notes beat no notes | at least 5 points, with a by-person bootstrap CI above 0 |
| H2. Real notes beat another decision's notes | at least 5 points, CI above 0 |
| H3. Another decision's notes stay near no notes | within 5 points (otherwise H1 is void) |
| H4. Cause trials | at least 65% |

H4 carries the same floor problem as R1: drivers may look more driver-like whatever the context. A pass on H4 alone shouldn't be read as people understanding the model.

## How to score

Put returned files in `humanstudy/responses/`, then run `python3 humanstudy/score.py`. The key is `runs/humanstudy/key.json`, kept out of the page.

## Limits

Ask people not to look up answers or discuss trials, and take the time data with a grain of salt. It's self-selected volunteers on one small model. A pass means notes help people predict this model's next word, not that they understand it in general.
