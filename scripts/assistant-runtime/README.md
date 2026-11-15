# Airi: experimental composed assistant

```sh
python scripts/assistant-runtime/chat.py
```

```text
Tu: Mi chiamo Marco e vivo a Torino. Come mi chiamo?
Airi [memoria]: Marco
Tu: In quale città vivo?
Airi [memoria]: Torino
Tu: Correzione: mi chiamo Giulia. Come mi chiamo?
Airi [memoria]: Giulia
```

The memory path uses Python's standard library only. It stores no conversation
files and needs no model download. `/reset` clears the process conversation;
`/exit` leaves. Its input limit is128 messages,8192 characters per message and
65536 characters overall, with values limited to200 characters/20 words. Over-limit
history abstains in full; it is not silently cropped. This is contextual recall
within the supplied history, not permanent profile storage or neural training.

When memory cannot answer, the CLI lazily uses the original approved native loader.
Use the same Python environment with Torch and the native `generalist_lm` package
on `PYTHONPATH`, and authenticated `gh` access, as the existing `chat_current.py`.
For example, with the existing project layout:

```sh
PYTHONPATH=../airi-pc-runtime/computer python scripts/assistant-runtime/chat.py
```

For an already downloaded approved checkpoint, supply both `--checkpoint` (its
parent containing `checkpoint.json`/`model/`) and `--model-record` (its matching
`AIRI_CURRENT.json`). Files are hash-checked by the original loader. Do not invent a
record for an unqualified candidate. The chosen model stays fixed for the process;
restarting selects the then-current approved model. The model remains experimental,
and no production qualification/provider gate is bypassed.

`--messages-file path.json` consumes one JSON array and emits one JSON answer with
`source` and provenance. `--native-only` bypasses memory for a clearly labeled
measurement of the neural model alone. `AiriAssistant(native_chat).respond(messages)`
provides the same composition API for callers; fallback receives the original
conversation unchanged. No candidate expert is loaded or selected by this API.

Only explicit supported user declarations and recall questions are recognized.
Names, locations and topics retain arbitrary unseen Unicode values. Corrections,
contradictions, quoted/reported/hypothetical text, role/speaker/session boundaries
and unsafe controls are checked conservatively. System/developer/tool messages
disable the memory route so it cannot override their instructions. Unsupported
syntax abstains; this limited grammar is not a general discourse understanding
system or a factual-world validator. Native fallback can still answer poorly.

## Verification

```sh
python -m unittest discover -s tests/assistant-runtime -v
```

The dedicated CI also checks functional coverage and preservation against the
existing frozen-native records. The inherited897-route check does not claim to
rerun every neural token: memory routes are compared to certified gold, and
abstaining routes forward the exact native call. Actual composed92 inference and
a separate32-case native/composed comparison are archived in the result evidence.

See [plan](../../docs/ARCHITECTURE_IMPROVEMENT_PLAN_20261115.md),
[results](../../docs/ARCHITECTURE_IMPROVEMENT_RESULTS_20261115.md), and
[neural curriculum experiment](../research-training/compositional-20261115/RESULTS.md).
