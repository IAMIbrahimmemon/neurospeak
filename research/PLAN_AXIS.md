# Study 11: what about repetition's edit frees a stuck run?

Written on 2026-11-06 before any data from these arms. Study 8 showed repetition frees loops and self-doubt doesn't at a matched dose. The review pointed out that equal KL isn't equal push along repetition's axis, and it left the fall from 33 to 19 exits unexplained. Removing repetition pushes along that axis and also overshoots self-doubt's activity past zero, from +0.86 to about -0.93. The projected version keeps self-doubt at +0.86 and pushes a little less along repetition's axis, 3.91 against 4.27. Two candidate causes for the fall, then: slightly less push, or the lost change to self-doubt. These arms separate them, and they also measure the response to push size.

**Runs.** The same 56 prefixes, seed 7, 1,500 new tokens, edits from the prefix's last token on, as in studies 7 and 8. Unedited, repetition at gain 1, and repetition-only at gain 1 are reused. The escape rule is unchanged.

| Arm | Edit | Push on repetition's axis, from loop activity | Self-doubt activity after |
|---|---|---:|---:|
| `rep_25` | repetition, gain 0.25 | 1.07 | +0.41 |
| `rep_50` | repetition, gain 0.5 | 2.14 | -0.03 |
| `rep_75` | repetition, gain 0.75 | 3.20 | -0.48 |
| `only_109` | repetition-only, gain 1.09 | 4.27 | +0.86 |
| `both` | repetition-only and self-doubt both removed | 4.27 | 0 |

Activity numbers come from the mean loop activity in `loop_confidence.json` with cosine 0.419, and they're averages only.

| Bar | Needed | Why |
|---|---|---|
| G1. Response to size | `rep_75` minus `rep_25` at least +20 points, CI above 0 | does exit rate grow with push? |
| G2. Small push does something | `rep_25` minus unedited at least +10 points, CI above 0 | is there a threshold above this size? |
| G3. Same push, self-doubt kept | repetition at gain 1 minus `only_109` at least +15 points, CI above 0 | is the fall from 33 to 19 about more than push size? |
| G4. Self-doubt adds at zero | `both` minus `only_109` at least +15 points, CI above 0 | does taking self-doubt to zero add exits when repetition is removed? |
| G5. Intact | each new arm has at most 2 more collapsed runs than unedited, else bars using that arm are void | |

**How to read it.**

- G3 and G4 pass: self-doubt doesn't free loops alone, but changing it adds to removing repetition. That's an interaction, and neither one axis nor two separate routes.
- G3 fails: push along repetition's axis accounts for the fall, and the one-axis story gains ground.
- G3 passes and G4 fails: only overshooting self-doubt adds exits, and taking it to zero doesn't.
- G1 passes and G2 fails: the response has a threshold above a push of 1.07.

**Limits known in advance.** Push and activity figures are means, and individual tokens vary. Gain 1.09 overshoots a little. 56 runs, one seed, one model, no correction for 4 tests. These are stuck-run edits and say nothing about runs edited from token one.
