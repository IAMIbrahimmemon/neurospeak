# Study 26: which parts of the model write the self-doubt signal?

Written on 2026-11-09 at about 06:00, before any decomposition or generation for this study.

**Why.** Every study so far treats the self-doubt signal at block 19 as one number per token, h·d. Studies 23 and 24 found that flattening how it rises and falls carries most of the gain. The next step inside the black box is to ask where that variation comes from. Each block adds its token mixer output (attention or linear attention) and its MLP output to the residual stream. So block 19's output is exactly the embedding plus 40 component outputs, and h·d splits into 41 terms.

**Part A, read only (CPU).** On the first 12 unedited traces of study 20 (`layers_none_traces.jsonl`), over completion tokens only, record each component's projection c·d for blocks 0 to 19. For each component report:

- its mean projection;
- its variance share, cov(c·d, h·d) / var(h·d), which sums to 1 over the 41 terms and says how much of the token-to-token variation it carries;
- its mean projection on tokens where the next 3 tokens contain a doubt word (`DOUBT` in `investigate/behaviors.py`), minus its mean elsewhere.

Script `research/writers.py`, output `runs/research/writers.json`. A sum check is required: the 41 terms must reproduce h·d to within 1% on every token checked.

**Selection rule, fixed now.** The top 3 components by variance share are the "writers". If the top 3 together carry under 30% of the variance, the signal is spread out, Part B is skipped, and that's reported as the finding.

**Part B, causal (GPU).** Same generation as studies 20 to 25 (GSM8K test rows 800 to 863, seed 20, 3,000-token cap, thinking on). Arms:

| Arm | Edit |
|---|---|
| writers | remove d from the output of each of the 3 writers, at its own block, every token: c − (c·d)d |
| top writer | the same for the top writer alone |

Study 20's unedited (4) and full L19 removal (32, replications 23 and 26) are the comparisons.

| Bar | Needed | Why |
|---|---|---|
| W1. Writers matter | writers minus unedited, CI above 0 | does blocking the top writers reproduce part of the gain? |
| W2. How much | writers minus full L19, signed | do 3 components carry it all? |
| W3. One writer | top writer minus unedited, signed | |
| W4. Signal actually cut | on 8 runs of the writers arm, h·d at block 19 has variance at most half of unedited's | blocking the writers really flattens the signal downstream, otherwise later parts rebuild it |
| W5. Intact | at most 2 more runs collapsed or with 200 spaces in a row than unedited | |

**Predictions, written before data.** The top 3 carry 30 to 50% of the variance and sit in blocks 12 to 19. The writers arm gets about 12 to 18, below full removal. The top writer alone gets under 10. I'm unsure whether later blocks rebuild the signal (W4).

**How to read it.**

- Writers arm near full and W4 passes: a few components produce the doubt signal, and blocking them is enough.
- W4 passes but the writers arm is near unedited: the variation those writers carry isn't what matters, or block 19 isn't where it's read.
- W4 fails: the model rebuilds the signal after the writers are blocked, so the signal is redundant, and accuracy results for this arm can't be read as testing the writers.
- Under 30% for the top 3: no small set of parts writes it. The signal is distributed, and that is the finding.

**Limits known in advance.** Variance share isn't causal. A component can carry variance that is just passed along from earlier ones. Component outputs at early blocks live in the same residual space, but d was built at block 19, so their projections onto d may mean something different early on. 12 traces, one seed, one model, 64 problems.
