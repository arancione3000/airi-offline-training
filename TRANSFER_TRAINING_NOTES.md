# AIRI compositional training trial

Previous six-hour student mastered 16 repeated training probes but scored 0/16 on new copy/recall and regressed on old skills. It was not selected. This establishes memorization; it does not establish conversational ability or an architecture defect.

## Changes

- Add 12,288 deterministic solver examples, balanced across Italian/English, copy/recall and short strings/longer strings/multiple words. Vary instructions, acknowledgements, system messages and intervening turns. Reserve all previous development values and 16 new final-test values.
- Remove fixed probe oversampling. Sample six copy/recall length groups uniformly. Increase old authored replay from 12.5% to 25%.
- Use three predeclared learning rates, changing after 4h and 8h actual optimizer compute. Training-probe accuracy cannot reduce the learning rate.
- Keep the previous 92 development probes intact. Label them development because they have been inspected repeatedly. Add 64 final-only transfer probes, excluded from sampling, phase changes and selection.
- Preserve the previous champion separately; reject candidate regressions under existing offline selection. Original native final gates remain unchanged and no live deployment occurs.
- Preserve model, optimizer and RNG across verified checkpoints. Bound this new cycle to 12h update compute and 32 job continuations. GitHub-only continuation and transient recovery require no assistant calls.

## Verification

Local tests check deterministic generation, split separation, blocked examples, coverage, compute limits and selection exclusions. Remote preflight verifies actual Torch optimizer recovery, prompt/target masks for every row, cached decoding equivalence and a supervised-gradient smoke test.

## Limits

These changes address demonstrated curriculum and schedule weaknesses. They do not certify that the model can chat or solve arbitrary new tasks. Semantic conversational quality and generalization must still be measured after training. No pretrained model, external model API or student-generated labels are introduced.
