# Autonomous operation without assistant calls

The active conversation-curriculum-v3 chain trains, validates, checkpoints and
self-dispatches on GitHub Actions. No ChatGPT automation or external model API is
used. The user's hourly assistant controller was disabled on 2026-10-07.

The watchdog runs on GitHub after completion events and hourly as a fallback.
It retries only the latest configured frozen experiment, only after an observed
transient network/platform failure, at most two retries. It refuses overlapping
training, older ancestors, changed refs, integrity failures and intentional
cancellation. Retried jobs retain their original verified input checkpoints;
unsaved work is recomputed and never counted twice. CPU GitHub jobs have their
own platform limits, independently of ChatGPT credits.

The existing bounded curriculum search and 512-update learning assessments remain
unchanged. Three stagnant validation rounds stop learning and retain the best
offline release. The watchdog does not bypass quality gates, create new untested
recipes, promote weights live, or promise indefinite conversational progress.
Code bugs or persistent service failures outside the retry budget require a later
inspection. These limits are deliberate; successful training is not a guarantee
of a generally capable conversational model.
