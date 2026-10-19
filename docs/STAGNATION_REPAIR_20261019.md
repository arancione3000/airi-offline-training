# Stagnation diagnosis and repair — 19 October 2026

The served offline champion remained at revision 1 because each failed development check restored its weights and created fresh Adam state. The previous completed cycle attempted 1,280 updates but accepted none of its 12 candidates. These were short independent attempts, not 1,280 cumulative learning updates. The subsequent interrupted cycle likewise rejected its first two candidates (one insufficient gain, one regression).

The native model does update. A separate diagnostic on the real revision-1 checkpoint reproduced all 23 previously passing development answers before training. With 64 human-only acquisition steps, human validation NLL fell from 3.353372 to 3.206238, while only 18/23 previously correct answers still passed. At 128 uninterrupted steps NLL fell further to 3.096297, still retaining 18/23. This is acquisition with interference, not evidence that conversation is solved. Some failed exact matches are paraphrases; others are incoherent. The existing promotion gates remain unchanged.

This diagnostic isolates ordinary human acquisition (learning rate 5e-7, gold retention/KL weights 16). It uses the champion for native anchors, omits archive steps, and evaluates the first 16 human development examples. It is not the complete production sampler or its exact development subset. Its outputs must not be compared numerically to the production human NLL as if they used the same evaluation. No final-test answers are trained or used to select weights.

The reference tokenizer and champion tokenizer have the same SHA256, `1099cf0cd718706212492967cb607034ef5fa8321a3e81125a106761b2d247f3`. A tokenizer mismatch does not explain this failure.

## Correction

- Failed intermediate promotion checks retain both candidate weights and Adam.
- A candidate receives 512 acquisition updates, with development checks every 64 updates.
- Only a complete failed trajectory advances the rejection count and learning-rate profile.
- Candidate state and trial counters survive checkpoint/job boundaries.
- Rejected final or globally exhausted candidates restore the champion, including when the last step was already evaluated.
- Consolidation remains an independent guarded transaction; its updates do not shorten the next exploration trial.
- The stable model pointer moves only after the original improvement and retention gates pass. Exploration never replaces the served champion directly.
- Limits remain six optimizer hours, 48 development decisions, 12 complete rejected trials, and 32 jobs. These are maximums, not promised wall-clock durations or guaranteed improvement.

Data, architecture, learning-rate profiles and loss weights are unchanged in the new experiment so the effect of preserving the trajectory can be assessed. Longer uninterrupted trials alone do not guarantee recovery of forgotten skills. Revision 1 is still a weak conversational model.

## Verification and provenance

55 transfer tests and 7 controller tests pass. Native tensor tests verify exact Adam continuation through save/reload, failed-interpolation restoration, non-finite update rejection, and final rollback. Remote preflight [39206281767](https://github.com/arancione3000/airi-offline-training/actions/runs/39206281767) passed on implementation commit `e66c7e49f8335df83366f1b686edfb87a27c6d8f`. Its ephemeral 32-step training test is an optimizer/integrity check, not conversational qualification.

Superseded run [39205173885](https://github.com/arancione3000/airi-offline-training/actions/runs/39205173885) was cancelled after durable checkpoints were preserved. The new seed is `airi-offline-410108-39205173885-1-2`, fingerprint `83d593500a25541237b5608bf296c1ac7e6356ffccce71038e5b682094bb0042`: revision 1, 25 archived lessons, 256 attempted updates and 245.6749 optimizer seconds in that superseded experiment. Its weights, configuration and tokenizer hashes match the verified champion; checkpoint metadata differs as expected. The OPTIMIZER archive SHA256 is `8a3fde450d9971096fa3ec12981be12a74b5e4e44c25d3eb00ff0afec079218c`. Using this latest seed preserves the additional archived lesson instead of reverting to the older 24-lesson seed.

The updated-seed preflight [39206739668](https://github.com/arancione3000/airi-offline-training/actions/runs/39206739668) passed on exact training commit `36938f17bded11370de48f4204655ab4fcc058eb`; the same commit passed branch and main CI. It was fast-forwarded to main after verification. The new experiment fingerprint is `ca314a40de4a37ae9c3f6b667bc3b589dbecfc5dd2ca93d43ff39d174434b22b`.

Training [39244699157](https://github.com/arancione3000/airi-offline-training/actions/runs/39244699157) was started after the user requested continuation at 13:26 Europe/Rome. Its native checks passed and its training step is running. The scheduled/completion-triggered controller was re-enabled; manual controller run [39244748628](https://github.com/arancione3000/airi-offline-training/actions/runs/39244748628) passed and retained the active job. The frozen training branch is `dialogue-persistent-exploration-20261019`.

No new champion or conversational improvement has been verified in this new cycle. The local isolated diagnostic was observed through 128 updates; its planned 256-update result was not recovered after the session interruption, so completion of that diagnostic is not claimed.

## Reproduce the isolated diagnostic

Set `PYTHONPATH` to the pinned native source's `computer` directory, install the pinned CPU torch runtime, then run:

```bash
python scripts/transfer-training/diagnose_stagnation.py \
  --checkpoint /path/to/verified/checkpoint/model \
  --baseline-result /path/to/verified/FINAL_RESULT.json \
  --output /path/to/diagnostic.json --steps 256
```

Use the Oct18 terminal MODEL archive (SHA256 `ba3ef42287182c8c6e399fc8515d07cb5f89cf9346d241a99bb530ef6dcf0050`) and its FINAL_RESULT for the reported baseline. The diagnostic never publishes weights or updates the stable model channel.

## First verified new-cycle progress

The initial durable checkpoint is `airi-offline-410108-39244699157-1-1`. Subsequent telemetry `AIRI_PROGRESS_39244699157.json`, attached to the migration seed release, was downloaded and SHA256-verified (`a84a3504748680936c7827e4878d54249ab6a7fc7ad8dc4a5eb80f0eb1d72118`). It reports step410124,16 new optimizer updates,18.942468 optimizer seconds,25 retained lessons,33 archived token positions,zero development decisions and zero adoptions. This confirms actual optimizer progress and successful lesson-bank migration, not a new conversational achievement. The controller's observed action was `active_job_retained`.
