# Study 16: is one tilted axis enough to free a stuck run?

Written on 2026-11-06 before any data from these arms. Study 11 first read "both removed beats repetition-only at the same push" as an interaction. A review showed one single axis, tilted about 40 to 45 degrees from repetition-only toward self-doubt, orders every arm by push just as well. Those arms can't separate the two, and the angle was fitted after seeing them. This study freezes the angle and tests it on stuck runs no arm has been edited on.

**Directions**, all from existing vectors, in `research/build_tilt.py`, frozen before any arm. `only` is repetition-only, and `s` is self-doubt, which is orthogonal to it. Then `tilt` = cos 42 only + sin 42 s, and `perp` = -sin 42 only + cos 42 s, which lies in the same plane at a right angle. `rand` is a random unit vector orthogonal to that plane, seed 2026.

**Prefixes.** Unedited Qwen runs on rows no rescue has used: `pairs_none`, `add_none`, `both_none` and `long_none`, cut at loop onset by the same rule as study 7, keeping prefixes of at most 1,500 words, and dropping the 8 `long_none` prefixes used to calibrate study 8. MobileLLM runs are excluded. Rescue edits start at the prefix's last token, seed 16, 1,500 new tokens, as in study 7.

**Gains.** `tilt` and `only` at gain 1. `both` removes `only` and `s` at gain 1, the same plane removal as study 11. `perp` and `rand` use a gain set so that mean absolute push equals `tilt`'s, measured on inactive arm activations in the prefixes' last 64 tokens before any edited arm. That is the only use of those activations. So `perp_eq` and `rand_eq` displace activations as far as `tilt`, in other directions.

| Arm | Edit |
|---|---|
| `none` | continue unedited |
| `only` | remove repetition-only, gain 1 |
| `tilt` | remove the tilted axis, gain 1 |
| `both` | remove repetition-only and self-doubt, gain 1 |
| `perp_eq` | remove the perpendicular axis, gain for equal push |
| `rand_eq` | remove the random axis, gain for equal push |

| Bar | Needed | Why |
|---|---|---|
| T1. Tilted beats repetition-only | exits, `tilt` minus `only` at least +10 points, CI above 0 | the fitted axis on new runs, and it has less push than `only` |
| T2. Same plane, right angle, does little | `perp_eq` minus `none` at most +5 points, CI top at most +10 | one axis, or a whole plane? |
| T3. Plane adds nothing to the axis | `both` minus `tilt` at most +5 points, CI top below +15, which is under study 11's +19.6 gap | is a second direction needed? |
| T4. Study 11 replicates | `both` minus `only` at least +10 points, CI above 0 | is there a gap to explain? |
| T5. Any big push | `rand_eq` minus `none` at most +5 points, CI top at most +10 | is the effect just displacement? |
| T6. Intact | each arm has at most 3 more broken runs than `none`, else bars using that arm are void | exits from wrecked text don't count |

Exits use the `escaped` rule from `score_rescue.py`, unchanged.

**How to read it.**

- T1 to T6 pass: one axis in this plane accounts for the exits, and study 11's gap needs no interaction. The axis was chosen by fit, then confirmed on new runs.
- T4 fails: study 11's gap didn't replicate. Then T1 and T3 have little to explain and I report that first.
- T4 passes and T3 fails: removing the plane does more than removing the axis. A second direction matters, whether as interaction or a second route.
- T2 or T5 fails: moving activations far enough frees runs in general, and "this axis is special" loses ground. If T5 fails, every rescue result so far needs rereading.

**Limits known in advance.** 42 degrees is the middle of a fitted range and not an optimum. Equal mean absolute push isn't an equal push on every token. The plane is two directions out of 1,024, so this says nothing about axes outside it. One seed, one model, no correction across bars, and T2, T3 and T5 are bars for "little or nothing" that only exclude effects above their CI tops. Exits aren't correct answers; correct counts will be reported with no bar.

## Amendment, 2026-11-06, before any edited arm

The 1,500-word limit left 63 prefixes, which is too few for T3's CI, and study 7 had no such limit. It's removed, and every onset prefix from these files is used. Gains are recalibrated on the larger set. Only the prefix count and the activations had been looked at.

## Second amendment, 2026-11-06, after a Fable review and before any edited arm

The review showed the first set of bars couldn't support the first reading, and it found a bad control. Recomputed by me from its fit and its dose log. These replace T1 to T6 and the readings above. They override anything above that disagrees.

**What the theory predicts.** A logistic fit of per-run push to all 11 in-plane arms of studies 7, 8 and 11 gives a best angle of 39 degrees, a band of 26 to 49, and 42 fits as well. On these 102 prefixes it expects `only` 37 exits, `tilt` 48, `both` 48, `perp_eq` 1. So the expected gaps are about 10 points for T1 and T3's arms, and these prefixes have less self-doubt activity than study 7's, so `both` minus `only` is expected at only about 10. Simulated pass rates with the theory exactly true were 0.36 for the old T1 and 0.31 for the old T4. The size requirement on those two is dropped, and a failure of either says little by itself.

**The fit doesn't explain everything.** At 42 degrees it predicts 26 exits of 56 for study 11's `both`, and 33 were observed. It reproduces the dose curve and the repetition-only arms but not the size of that gap. So T3 is the live test.

**New arm.** `only_eq`, repetition-only at gain 1.11, set from the frozen activations so its push along the tilted axis equals `tilt`'s. One tilted axis predicts `tilt` and `only_eq` tie. An axis at repetition's own angle predicts `only_eq` wins by about 12 points, and repetition-only's own axis by about 30.

**Changed arm.** `rand_eq` was a fault. Gain 21 on a noisy near-zero projection flips its sign and amplifies its noise, with 13 times `tilt`'s KL dose and 21% of tokens pushed the opposite way. It is now a constant shift, `h - 3.57 rand`, every token the same, which matches `tilt`'s mean displacement. KL dose is reported for every arm by `research/tilt_dose.py`.

**What `perp_eq` is.** It pushes repetition-only about as far as `tilt` does (-2.38 against -2.63) and moves self-doubt the opposite way: `tilt` lowers it by 2.37 and `perp_eq` raises it by 2.64. So it's a test of the sign of the self-doubt change, and study 9 found that adding self-doubt induces loops. A low `perp_eq` count may partly reflect that.

| Bar | Needed | Why |
|---|---|---|
| T1. Signed | `tilt` minus `only`, CI above 0. CI below 0 reads as an axis nearer repetition | the fitted axis on new runs |
| T2. Tie at equal push | `tilt` minus `only_eq` within 5 points, CI top below +15 | the sharpest single-axis prediction |
| T3. Plane adds nothing | `both` minus `tilt` at most +5, CI top below +15 | is a second direction needed? |
| T4. Study 11 replicates | `both` minus `only`, CI above 0 | is there a gap to explain? |
| T5. Direction in the plane | `tilt` minus `perp_eq` at least +20, CI above 0, and `perp_eq` at most 8 of 102 | one axis, or a whole plane? |
| T6. Not displacement | `tilt` minus `rand_eq` at least +20, CI above 0 | is any equal move enough? |
| T7. Intact | each arm has at most 3 more broken runs than `none`, else bars using that arm are void | |

A Claude reader also audits exits in `perp_eq` and `rand_eq` for garbled text, since the broken rule misses it. Exits per source are reported as a check. `none` should be near 0: 0 of 79 of the `pairs`, `add` and `both` runs left a loop in their first 3,000 tokens.

**How to read it.**

- T1 to T7 pass: one axis in this plane accounts for the exits and no interaction is needed. The axis was chosen by fit, then confirmed on new runs.
- T2 and T3 pass, T1 or T4 fails: consistent with one axis, but the gaps were too small to see at this size. Not confirmed.
- T2 fails with `only_eq` ahead: the axis is nearer repetition-only than 42 degrees.
- T3 fails: removing the plane does more than removing the axis. A second direction matters, whether as interaction or a second route.
- T5 fails: pushing repetition-only frees runs whichever way self-doubt moves, so self-doubt's part in the axis is in doubt.
- T6 fails: any equal move frees runs, and the rescue results need rereading.

## Erratum, 2026-11-06, after the data

A Fable review found a mislabel in the second amendment. "An axis at repetition's own angle predicts `only_eq` wins by about 12 points, and repetition-only's own axis by about 30" gives the numbers for `only` minus `tilt`. For `only_eq` minus `tilt` a fit on studies 7, 8 and 11 gives +23.6 and +41. No bar used these numbers, so the verdicts are unchanged.
