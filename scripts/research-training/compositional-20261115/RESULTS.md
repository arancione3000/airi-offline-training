# Native compositional result, 2026-11-15

The isolated candidate is rejected. It failed both predeclared novel-transfer
thresholds and lost 21 of the approved source's 23 correct old probe answers.
No production weights, runtime, activation, or deployment changed. These are
native neural metrics, separate from any explicit software memory results.

| Endpoint | Native exact literal + EOS | Both answers in counterfactual pair exact |
|---|---:|---:|
| TRAIN | 391/512 (76.37%) | 162/256 |
| Final: unseen values, familiar wording | 4/16 (25%) | 2/8 |
| Final: familiar values, new wording | 3/16 (18.75%) | 1/8 |
| Final: unseen values and new wording | 0/32 | 0/16 |

Development was 13/32: 4/8 unseen values, 6/8 new wording, and 3/16 both-new.
The frozen approved source was 0/32 development and 0/64 final, first measured
at this same endpoint. All TRAIN/development/final whole-gold teacher-forced
certificates agreed with native greedy raw-ID/EOS correctness. Final
counterfactual outputs were never identical within a pair, but only 3/32 pairs
were wholly correct; changing output with context is insufficient evidence of
reliable copying.

Training stopped at the committed wall cap after 493 of 512 planned updates,
600.543 seconds including the last indivisible update. It completed 1,972 row
exposures: 436 rows received four exposures and 76 received three. No learning
rate, data, dose, checkpoint selection, or follow-on trial was changed using
these results. The raw TRAIN CE was 0.2133. TRAIN was not fully learned, so this
bounded result does not establish an architectural or asymptotic limit.

Old native probes fell from 23/92 to 3/92. Two original successes survived and
one previously wrong probe became correct. Full 897-row certification was
skipped under the fixed potential-qualification trigger because both expert
transfer thresholds failed; the old exact-answer gate also already failed.
Old NLL gates were not evaluated, and there is no production acceptance claim.

The immutable approved tensor digest remains
`57bdb933b82bd8807af5c8b6b545c27ee9114ee4f4c59022434d39b6bf771c79`.
All pinned source/checkpoint/code hashes remained unchanged. The isolated
candidate export reload reproduced bit-exact native logits and raw greedy
IDs/EOS on six fixed TRAIN examples. Full raw answers, gold/generated IDs,
source provenance, fixed trajectory, and hashes are in the evidence artifacts.

See REPRODUCTION.md for exact design, costs, the transparent zero-update
collation repair, and the scheduled-bucket annotation limitation. Primary
condition metrics do not depend on that annotation.
