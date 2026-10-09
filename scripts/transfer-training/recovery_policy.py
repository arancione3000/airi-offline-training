"""Deterministic, resumable training-only recovery sampling."""
START_STEP = 13107
EXTRA_UPDATES = 4096
LEARNING_RATE = 2e-6


def migrate_optimizer(optimizer, counters):
    if counters['completed_steps'] != START_STEP:
        raise ValueError('migration requires the exact completed first pass')
    for group in optimizer.param_groups:
        group['lr'] = LEARNING_RATE


def choose_batch(step, counters, rng, by_language, dialogue, replay):
    phase_step = step - START_STEP
    if not 0 <= phase_step < EXTRA_UPDATES:
        raise ValueError('recovery update outside finite budget')
    if phase_step % 2 == 0:
        lang = ['it', 'en'][(phase_step // 2) % 2]
        index = rng.choice(by_language[lang])
        counters['recovery_dialogue_updates'] = counters.get('recovery_dialogue_updates', 0) + 1
        return 'dialogue-' + lang, dialogue[index]
    cursor = counters.get('recovery_replay_updates', 0)
    groups = list(replay)
    group = groups[cursor % len(groups)]
    batch = replay[group][rng.randrange(len(replay[group]))]
    counters['recovery_replay_updates'] = cursor + 1
    counters['replay_updates'] += 1
    return group, batch
