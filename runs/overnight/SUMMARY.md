# Overnight run, 2026-11-03 to 2026-11-04

Goal: open the black box to humans. Every bar was written before its test ran. Some follow-up bars (F6, F7, F12, C1, C2) were added after seeing earlier results, and their plans say so. Details are in each folder's RESULTS.md, and the decision trail is `decisions.tsv`. Nothing is committed.

| Work | Result | Where |
|---|---|---|
| `why` | Mechanics pass on 3 dictionaries, 83% flips against 9% random at L19. Names don't help Claude readers predict. | whytest/RESULTS.md |
| `trace` | Mechanics pass. No valid test of whether names reveal links. | tracetest/RESULTS.md |
| Evidence records | 984 L19 and 279 L15 words have names, and 100 at each layer have test scores. | runs/words/ |
| Second model | distilgpt2: read 70.9% fails 75%, steering 86.5% passes, features +8.0 over neurons. | wordtest/distilgpt2/RESULTS.md |
| Human study | One person took it, and the bars need 12, so they can't be judged. No more responses are coming. | humanstudy/RESULTS.md |
| `diff` | Passes: 0.61 between real edits, 0.03 and 0.17 against random. | difftest/RESULTS.md |
| Reasoning dictionary | Fails both bars: traces 22.4% (needed 22%), Wikipedia 23.9% (needed 21%). Old one stays. | reasoning/RESULTS.md |
| Word lists against loops | Fail: F1–F3 and F5–F7 on a flawed detector, F9–F12 after the fix. | whyfix/RESULTS.md |
| Direction removal | Pass: F13–F16, A1, A2 on 64 new problems (3/64 to 10/64 correct, repeats 39.9% to 9.2%), C1, N1, N2 (cost about 0.02–0.03 nats). C2 fails by the letter because the metric can't see gibberish; removing the mean destroys the model. | whyfix/RESULTS.md |

## Second loop: debugging a behavior

| Work | Result | Where |
|---|---|---|
| Five behaviors, from example pairs | Chinese works. Hedging, restating, bullets fail, including after a confound fix. Self-doubt fell but missed my floor. B1 fails, 1 of 5. | investigate/RESULTS.md |
| Second model | distilgpt2 repetition fails, B2 and M2. | investigate/RESULTS.md |
| Repetition from examples alone | 34.2% to 8.4%. Failed as run on a detector bug, passes after the fix. | investigate/RESULTS.md |
| Specificity | B4 and B5 pass. | investigate/RESULTS.md |
| Learned directions | 0 of 5, M1 fails. | investigate/RESULTS.md |
| Reports, Claude readers | E1 passes, E2 and E3 fail. | investigate/RESULTS.md |
