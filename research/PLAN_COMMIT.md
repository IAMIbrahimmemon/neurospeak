# Study 10: does removing self-doubt reason better, or commit sooner?

Written on 2026-11-06 before computing anything below. No new generation: this reads existing traces on the CPU. Removing self-doubt raised correct answers from 11 to 49 of 128. Right-when-answered rates looked similar across arms, so I inferred it commits sooner and doesn't reason better. That reading was never tested. If it commits sooner, the unedited model should already reach the right value in its reasoning about as often, and then fail to put it in a box.

**Data.** `runs/whyfix/gsm64_*` and `gsm64b_*`, 128 problems: unedited, self-doubt removed, matched random removed. Same traces as the replication.

**Measures.** A run *reaches* the answer if the correct value appears as a standalone number anywhere in its text, with commas removed. Problems whose answer value appears in the problem statement are left out, since it would count without any reasoning. Reached is a noisy measure: a value can appear by coincidence or as a step toward something else. CIs are paired bootstraps over problems.

| Bar | Needed | Why |
|---|---|---|
| C1. Unedited already reaches it | unedited reaches the value in at least twice as many runs as it answers correctly | the raw material for "commits sooner" |
| C2. Reaching changes little | self-doubt removed minus unedited on reached is at most +10 points | it isn't finding more right values |
| C3. Committing changes a lot | among problems reached in both arms, self-doubt removed minus unedited on correct answers is at least +20 points, CI above 0 | same value found, more often stated |
| C4. Control | the same difference as C3 for matched random minus unedited has CI that includes 0 | a random edit doesn't do this |

**How to read it.**

- C1 to C4 pass: commits sooner is supported on this data.
- C2 fails and C3 passes: both more right values found and more stated, so it isn't only about committing.
- C3 fails: the "commits sooner" reading is wrong as stated.

**Limits known in advance.** Under a 3,000-token cap, a value reached late has less room to be stated. Reached is a word-level match and not a judgment that the reasoning was sound. Conditioning on reached in both arms selects problems. It's observational reading of traces, with no new intervention. One model.
