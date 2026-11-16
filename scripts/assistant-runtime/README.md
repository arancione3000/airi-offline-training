# Airi: experimental composed assistant with persistent learned experts

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
`source` and provenance. `--native-only` bypasses memory and learned experts for a clearly labeled
measurement of the neural model alone. `AiriAssistant(native_chat).respond(messages)`
provides the same composition API for callers; fallback receives the original
conversation unchanged. Pass `skill_bank=SkillBank(path)` to enable learned symbolic
experts in this API. The chat CLI enables the qualified bundled bank by default.
No neural candidate checkpoint is selected by this component.

Only explicit supported user declarations and recall questions are recognized.
Names, locations and topics retain arbitrary unseen Unicode values. Corrections,
contradictions, quoted/reported/hypothetical text, role/speaker/session boundaries
and unsafe controls are checked conservatively. System/developer/tool messages
disable the memory route so it cannot override their instructions. Unsupported
syntax abstains; this limited grammar is not a general discourse understanding
system or a factual-world validator. Native fallback can still answer poorly.

## Learning new procedural skills

The routing order is existing memory, ordered learned experts, then the native
model. Six synthetic experts are bundled: parcel-code copy, signal reversal,
card-text uppercase, plate-text lowercase, and two named integer operations.
For example, these messages are answered without loading Torch:

```json
[{"role":"user","content":"Il segnale acquisito è Abé42."},
 {"role":"user","content":"Come restituisci il segnale?"}]
```

The answer is `24ébA`, with `source: learned_skill`, expert identity and exact
source spans. The learner infers literal extraction anchors, an exact question,
and one uniquely consistent operation from copy/reverse/uppercase/lowercase/
add/subtract/multiply. It does not generate arbitrary algorithms or new neural
layers. New values generalize inside learned templates; paraphrases may abstain.

```sh
python scripts/assistant-runtime/learn.py --status
python scripts/assistant-runtime/chat.py --learn /path/to/trusted-lessons.json
python scripts/assistant-runtime/learn.py --watch /path/to/trusted-queue --interval 30
```

Interactive chat also accepts `/learn /path/to/trusted-lessons.json`. A lesson has
`lesson_id`, `trusted: true`, `examples` and `qualification`; each row contains
`messages` and `answer` (null for required abstention). Supply4–64 TRAIN examples,
then at least8 positive and4 negative distinct qualification examples with new
extracted values. The synthetic examples in the [reproduction bundle](../research-training/modular-growth-20261116/REPRODUCTION.md)
show the full format. Only TRAIN selects a program. Admission replays all old
expert examples and989 stable historical routing contracts; failure preserves
the current bank. Old memory and earlier experts retain priority.

The default private bank is `~/.cache/airi/skill-bank.json`, initially seeded
from the six bundled experts when learning first runs. `--skills PATH` in chat
or `--bank PATH` in the learner uses an explicit independent bank; an absent
explicit path starts empty. `--no-skills` retains just memory/native routing.
The watcher polls trusted local JSON files, admits new valid lessons and treats
repeats as unchanged. It needs a running Linux process; use a service supervisor
for machine restarts. Ordinary chat is not silently labeled as training, and the
watcher cannot improve without new correct lessons. No external model/API is used.

Validated snapshots are committed atomically before the convenience main copy.
An interrupted/corrupted main copy can recover from a valid committed snapshot;
if no valid bank remains, the assistant falls back to the native model with a component error. Checksums
detect accidental corruption, not a malicious filesystem owner. Banks are tied
to the compiler source hash: a compiler change requires requalification. Capacity
is32 experts, with no automatic knowledge eviction. This experiment does not
change the production provider, approved model pointer or frozen training service.

## Verification

Automatic curriculum cycles can run independently of incoming lesson files:

```sh
python scripts/assistant-runtime/auto_update.py --watch --interval 3600 --limit 2
```

This watcher generates exact-reference lessons from a finite12-family authored
catalog and stages at most two missing experts per cycle. It audits new values,
old examples and989 historical routes before atomic activation. A complete catalog
is audited without adding duplicates. The hourly GitHub workflow also validates
and publishes qualified bundled updates independently of this chat. Existing
private banks now qualify/import missing installed bundled experts while retaining
private additions; this does not fetch remote source. See [operation and limits](../../docs/AUTOMATIC_MODULAR_UPDATES_20261116.md).

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
