# Does a report with evidence tell a reader what a direction does?

Written on 2026-11-04 before any result. **Readers are Claude models, not people.** This can't show that people understand a report.

**Question.** A report can name a behavior, or show what removal did. Which one lets a reader predict the effect on a new prompt?

**Trials.** Behaviors where removal changed something: repetition, Chinese, self-doubt, using the pairs method's saved outputs. 14 trials each, picked at random from prompts not shown in any report, so 42 in all. Each shows a new prompt and two outputs, A and B, one unedited and one with the direction removed. The reader picks which had it removed. Outputs are shown as their last 500 characters. A and B are balanced exactly within each condition.

**Conditions**, each trial appears in all four, across different readers:

| Condition | The reader gets |
|---|---|
| none | nothing about the direction |
| wrong | the before-and-after evidence for a different behavior |
| name | the behavior's name only, for example "this direction was built to capture repetition" |
| evidence | two before-and-after examples from that behavior's report, with their scores and no name |

8 readers, 21 trials each, Claude Sonnet, no reader sees the same trial twice.

| Bar | Needed |
|---|---|
| E1. Evidence is enough | evidence at least 75% right |
| E2. Evidence beats nothing | evidence minus none at least 10 points, trial bootstrap interval above 0 |
| E3. Evidence beats wrong evidence | evidence minus wrong at least 10 points, interval above 0 |
| E4. Showing against naming | evidence minus name, reported with no bar |

Limit: these behaviors are the ones that visibly change, and Chinese should be easy, so results are also shown per behavior. A pass doesn't show that people benefit, or that reports help with hidden behaviors.
