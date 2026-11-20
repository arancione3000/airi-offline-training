"""Route approved literal TRAIN requests to installed immutable program IDs."""
import json
import os
from pathlib import Path
import re
import stat

import skill_programs
from skill_bank import RegistryError, digest

DEFAULT = Path(__file__).resolve().parent / 'queries/approved-aliases.json'
MAX_ALIASES = 512
MAX_BYTES = 256 * 1024


def read_aliases(path):
    """Read one atomic snapshot. A missing initial snapshot has no aliases."""
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW)
    except FileNotFoundError:
        return {}
    try:
        with os.fdopen(fd, 'rb') as stream:
            if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                raise RegistryError('alias state is not a regular file')
            raw = stream.read(MAX_BYTES + 1)
        if len(raw) > MAX_BYTES:
            raise RegistryError('alias state too large')
        state = json.loads(raw)
        aliases = state['aliases']
        if (state['schema'] != 1 or not isinstance(aliases, dict)
                or len(aliases) > MAX_ALIASES or digest(aliases) != state['sha256']):
            raise RegistryError('invalid alias state')
        for query, row in aliases.items():
            if (not 0 < len(query) <= skill_programs.MAX_QUERY_CHARS or not query.isprintable()
                    or set(row) != {'program_id', 'lesson_id', 'train_id'}
                    or not re.fullmatch(r'[0-9a-f]{64}', row['program_id'])
                    or not all(isinstance(row[key], str) and 0 < len(row[key]) <= 128
                               for key in ('lesson_id', 'train_id'))):
                raise RegistryError('invalid alias binding')
        return aliases
    except (KeyError, ValueError, TypeError, RecursionError) as error:
        raise RegistryError('invalid alias snapshot') from error


def resolve_alias(state, messages, binding):
    """Keep source eligibility, conflict checks, original spans and speaker IDs."""
    sources = skill_programs._eligible(messages)
    if not sources:
        return None
    chosen = next((e for e in state['entries'] if e['id'] == binding['program_id']
                   and e['lesson_id'] == binding['lesson_id']), None)
    if chosen is None:
        return None
    program = chosen['program']
    answer = skill_programs._predict_valid(program, sources)
    if answer is None:
        return None
    for entry in state['entries']:
        other = entry['program']
        if other['source'] == program['source'] and other['operation'] == program['operation']:
            continue  # The same procedural expert can already have another exact question.
        if skill_programs._predict_valid(other, sources) is not None:
            return None
    return {**answer, 'skill_id': chosen['id'], 'skill_revision': state['revision'],
            'routing': 'verified_exact_alias', 'query_provenance': {
                'original_query': messages[-1]['content'], 'canonical_query': program['query'],
                **binding}}


class AliasRouter:
    def __init__(self, bank, path=DEFAULT):
        self.bank = bank
        self.path = Path(path)

    def match(self, messages):
        """Return ownership even when an exact alias must abstain on its context."""
        if (type(messages) not in (list, tuple) or not messages or type(messages[-1]) is not dict
                or type(messages[-1].get('content')) is not str or self.bank is None):
            return False, None
        binding = read_aliases(self.path).get(messages[-1]['content'])
        if binding is None:
            return False, None
        return True, resolve_alias(self.bank._read()[0], messages, binding)

    def predict(self, messages):
        return self.match(messages)[1]
