# Modular architecture review, 16 November 2026

The public bank reached12 experts at18:41UTC. The scheduled workflow had only
three runs in total, all successful; at22:25UTC no later run existed. This was
missing scheduler delivery, not failed learning or a full queue. An hourly cron
expression therefore did not provide the promised hourly execution.

## Scheduling repair

`airi-modular-supervisor.yml` now runs a serialized bounded observer. It checks
the latest successful update run, waits for the hourly interval, skips an update
already queued/running, and explicitly dispatches a due update. Every40minutes
it explicitly requests a replacement observer before exiting. A calendar and
native-training completion events provide independent recovery opportunities.
Supervision does not depend on this workspace or an assistant conversation.

The observer backs off after uncertain requests, retries its renewal, and pauses
automatic recovery after a learning/test failure so failed qualification is not
silently ignored. Due time uses successful Actions runs, including audits after
the finite catalog is complete; the last learned-bank timestamp is not a
heartbeat. Existing update admission, full candidate tests and fast-forward-only
publication remain mandatory.

This improves recovery from missed cron events; it cannot guarantee exact wall
clock execution during GitHub outages or runner shortages. The observer keeps an
Actions runner allocated while waiting and uses bounded API polls. It runs no
model or paid inference API. Disable both modular workflows to pause the service;
cancelling a single observer does not disable its recovery triggers.

## Registry bugs repaired

Deeply nested damaged JSON below the file-size limit could raise RecursionError
outside the normal error boundary and abort chat. It now becomes RegistryError,
allowing recovery from a valid snapshot or stable native fallback.

A legitimate revision0 seed plus32 experts requires33 snapshots. The old count
limit rejected that valid registry at its advertised capacity. The limit now
includes revision0. Both failures have reproducing tests. Compiler source and
existing program identities remain unchanged; no neural weights or frozen native
training modules were modified.

## Effectiveness evidence

The earlier24-case actual comparison found24/24 learned-program answers versus
0/24 for the same native model, both plain and with eight TRAIN demonstrations.
That narrow result concerns exact templates and known operations; it does not
establish general Transformer superiority. A fresh fixed-bank12 review separately
tests known templates, paraphrases and expert-abstention boundaries. Its complete
inputs, source snapshots and results are recorded with the review evidence.
