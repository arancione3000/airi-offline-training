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

The fresh fixed-bank12 review used the same verified revision4 checkpoint,
greedy decoding and actual native generation, plain and with the expert's eight
stored TRAIN demonstrations. No prompts were truncated.

| Fresh condition | Composed assistant | Native plain | Native8TRAIN |
| --- | --- | --- | --- |
| Original12 new-value cases, known wording | 9/12 | 0/12 | 0/12 |
| Six semantic question paraphrases | 0/6 | 0/6 | 0/6 |
| Separate post-review arithmetic diagnostic, inside bounds | 3/3 | 0/3 | 0/3 |

Three original arithmetic inputs exceeded the declared integer bound of1,000,000.
The experts correctly abstained, but native fallback was wrong. Those failures
remain in the original9/12 result. All nine original text cases were correct.
The separately preregistered in-range diagnostic is explicitly post-review and
is not pooled into a new blind12/12 claim. No program or weight changed.

Experts also abstained correctly on6/6 quoted sources. All240 stored contracts
and989 old routing checks passed. Those989 checks inherit native behavior; they
are not fresh neural generation on989 questions. Checkpoint hashes and RNG
remained unchanged. The comparator is the small Airi native model, not current
strong pretrained Transformers or an equal-compute architectural benchmark.

The modules provide useful bounded procedural behavior, while new wording and
out-of-range fallback remain substantial limitations. General conversation and
Transformer-class superiority are unproven. See [complete results](../scripts/research-training/modular-review-20261116/RESULTS.md)
and [frozen inputs, code and reproduction](../scripts/research-training/modular-review-20261116/REPRODUCTION.md).

## Live activation

PR15 merged as4116dbbd0a19302b3e411c71211fa541d5d111c6 after both GitHub checks
passed. The60-second observer test43971589118 succeeded, dispatched update
43971615534 and explicitly renewed observer43971779345. The recovered update
published12→14 experts after160/160 audit cases and the full candidate test suite.
The replacement observer was confirmed running. This verifies live handoff,
not merely a cron configuration or a simulated timer.

The persistent local private bank had independently completed the catalog at18
experts. A fresh operational audit passed240/240 catalog probes,360 stored expert
contracts and989 historical routes; the actual chat CLI returned90−30=60 through
its learned expert. Private and public banks have different training histories;
their revisions must not be conflated. This local audit is monitoring, not another
blind study. The public bank still has four catalog families pending.

Validation passed111 assistant tests,7 frozen-training tests and30 native-controller
tests. The frozen31 training modules and compiler hash remain unchanged.
