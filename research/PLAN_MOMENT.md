# Study 29: does the self-doubt signal fire when the model finds its answer, and does it predict whether it concludes?

Written on 2026-11-09 after study 28's review, before any activation for this study was measured.

**Why.** Study 28's review raised a rival reading. d may be a "keep thinking, don't conclude" direction rather than a doubt direction. Its lens pushes down "Consequently", Chinese "therefore" and end-of-text, and removing it lets 19 runs close their thinking against 0 unedited. Study 26 found h·d rises just before doubt words, but d was built from those words, so that is circular. This study asks two questions that don't use doubt words at all. Does h·d rise at the moment the model first writes the right answer, more than when it writes an intermediate result? And at that moment, does h·d predict whether the model goes on to conclude?

**Data.** The 192 unedited traces at the 3,000-token cap: `runs/whyfix/gsm64_none_traces.jsonl` (rows 0 to 63), `runs/whyfix/gsm64b_none_traces.jsonl` (rows 144 to 207) and `runs/research/layers_none_traces.jsonl` (rows 800 to 863). Each trace is fed back through the unedited model (prompt plus completion, teacher forced, on the GPU, no generation), and block 19's output is projected on d.

**Events.** Found in the completion text with character offsets mapped to tokens.

- Answer event: the first match of `=\s*\$?<answer>\b`. The event token is the token holding the answer's last character.
- Intermediate events: every other match of `=\s*\$?<number>\b` whose number isn't the answer, in the same trace.

**Score.** h·d at the event token minus the mean h·d over the 200 tokens before it. The local baseline removes the known rise of h·d through a run (0.23 early to 0.60 late), since answer events come later than most intermediate ones. Events with fewer than 50 tokens before them are dropped.

| Bar | Needed | Why |
|---|---|---|
| M1. Signal at the answer | per trace, answer score minus mean intermediate score, paired over traces with both, CI above 0 | d fires at the moment the answer is found, more than at other results |
| M2. Predicts concluding | at answer events, AUC of minus the score for "concludes within the next 150 tokens", at least 0.60 with CI above 0.5 | a low signal at the answer predicts the model wraps up |
| M3. Not just the next word | M1 recomputed with events dropped when a doubt word starts within 3 tokens after the event, signed | how much of M1 is the next word already being a doubt word |

"Concludes" means the next 150 tokens contain `\boxed{` followed by a non-space character, `</think>`, "final answer" or "the answer is", with no `DOUBT` word before that match. CIs are bootstrap over traces.

Secondary, no bars: the same scores for the raw repetition direction (`dir_neurospeak`) and for a random unit direction with seed 0, as controls for "any direction moves at answers".

**Predictions, written before data.** M1 passes with the answer score about 0.3 to 0.8 higher. M2 passes with AUC about 0.65. M3 keeps more than half of M1. The random direction shows no difference.

**How to read it.**

- M1 and M2 pass and M3 keeps most: d fires at the decision point where the model could conclude, and its size predicts whether it does. That fits "don't conclude" and doubt alike, but it is the first non-circular sign that d's activity is tied to the moment of commitment.
- M1 passes but M3 drops most of it: the signal at the answer is mostly the next doubt word being prepared, back to the circular reading.
- M1 fails: d doesn't single out the answer moment, and the "don't conclude at the answer" story weakens.

**Limits known in advance.** Matching "= answer" misses answers written other ways and can fire on a coincidental intermediate equal to the answer. Teacher forcing reads the signal on text the model wrote itself, so it is observational, not causal. Unedited traces rarely conclude (0 of 64 finished on rows 800 to 863), so M2 may have few positives, and if fewer than 15 answer events conclude, M2 is reported as underpowered rather than judged.
