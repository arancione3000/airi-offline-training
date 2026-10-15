# Selective success transfer for AIRI

A rejected native candidate can improve some outputs and regress on others. Before rollback, this runner mines a rotating, stratified sample from the **existing authored training set only**, and stores better gold-token distributions in `success_bank.json`. The checkpoint manifest covers this file along with the native weights and Adam state. Rollback restores weights and Adam but retains the bank in memory; later checkpoints and jobs restore the accumulated lessons.

## What is transferred

Each lesson records its exact training row, donor step, runner SHA, donor model digest, fingerprint, token positions, targets and probability distributions. A position is eligible only when the donor's top prediction equals the known target, its probability is at least 0.10 and its log probability exceeds the champion by 0.05 nats. Up to eight positions per row and 64 rows are kept, spanning languages and task families. This is **teacher-forced token evidence**, not a claim that the donor can answer the whole question. Up to four promising responses per harvest are also generated freely; only an exact correct donor response that the champion gets wrong is counted as a whole-answer gain.

Up to two out of sixteen training slots replay these lessons using supervised target loss, donor KL distillation, supervised anchor replay and KL to an independent frozen champion. Human dialogue slots and native replay slots remain present. The native model is the only model used at inference: no response routing, external teacher, pretrained weights or external API. A lesson can come from a different preceding AIRI candidate at every round.

## Independent adoption

Neither development nor final examples are harvested or replayed. Human references and self-generated answers are excluded from the verified bank. Development selection still requires all original metric gates and, additionally, retention of every previously correct development response by prompt identity. A new checkpoint is only called an improved champion after that check passes. Final tests remain outside adaptation. Partial output-level distillation cannot guarantee arbitrary combinations of incompatible skills, or improvements outside the measured cases.

## Bounds and audit

The proposed first deployment is a single one-hour optimizer pilot after the authorized fifty-hour chain finishes, with a fresh immutable branch and exact terminal seed. The fifty-hour chain is not modified. The controller is idempotent, respects cancellation, retries only classified transient errors at most twice and disables itself after completion. The pilot does not request more cycles. A preflight runs a two-task native transformer experiment and an ephemeral gradient/restart check on a real AIRI checkpoint. Those integration checks do not publish changed weights or prove general conversation.

Old rejected weights that were never saved cannot be reconstructed. The bank starts empty when the feature is first enabled. Older archived candidates cannot donate evaluation answers: they need eligible training observations and compatible native provenance.

## Conceptual sources

- Li and Hoiem, Learning without Forgetting: https://arxiv.org/abs/1606.09282
- Yu et al., Select and Distill: Selective Dual-Teacher Knowledge Transfer for Continual Learning on Vision-Language Models: https://arxiv.org/abs/2403.09296

These papers motivate output distillation and retention. This implementation adapts the principles to AIRI's byte-token native training and makes no claim to reproduce their experimental results.
