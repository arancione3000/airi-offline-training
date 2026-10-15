# One evolving AIRI model

The native model keeps the stable identity `airi-native`. Experimental candidates are temporary donors. The system stores verified training lessons before discarding a regressing candidate and tries to consolidate **one language/task family at a time into the current champion**. It does not copy an entire failed donor over the champion.

Each transaction is bounded to 32 optimizer updates drawn from the same one-hour pilot ledger. Every attempted update counts, including rejected transactions. After a transaction, all original development gates and preservation of each previously correct development answer must pass. A failed transaction restores both champion weights and Adam, preserving previously accepted skills, the lesson bank, exploration RNG and spent compute. Repeated unsuccessful identical lessons receive a bounded backoff. A successful transaction advances the same model identity by one internal revision.

## Stable access

After an immutable selected checkpoint is uploaded completely, `models/AIRI_CURRENT.json` on branch `airi-model-current` is atomically updated with a GitHub contents SHA compare-and-swap. It names the same model identity, its revision, verified weight lineage, immutable checkpoint and hashes of all native model files. A candidate, partial upload, skipped revision or stale lineage cannot advance this reference. A rollback with unchanged champion weights does not move it.

The channel is initialized by the first actual pilot save, not by the read-only preflight. The consumer command, with the project's native dependencies installed, is:

```bash
python scripts/transfer-training/chat_current.py --cache airi-current-cache
```

The consumer resolves and verifies the current native model automatically and checks for accepted revisions between messages, keeping conversation history. No manual release tag selection is required. Existing separate desktop/web applications are not silently rewired by adding this consumer. The model remains experimental and its general conversation quality is not established.

The old authorized fifty-hour chain remains frozen. This version replaces the pending **same one-hour pilot**, rather than adding another training budget. The automatic controller requires a successful preflight at the exact new template commit, waits for the verified fifty-hour terminal, creates the successor once, and disables itself at the end of that pilot. It does not authorize infinite compute or promise indefinite improvement.

---

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
