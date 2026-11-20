"""Ground one object in the installed inventory before applying the frozen grammar.

The lexicon is rebuilt from public object names, with no lookup of the source
turn or the requested operation. A single exclusive content word, or a jointly
exclusive combination, must distinguish a family from all other installed ones.
The operation grammar still has to consume everything outside the noun phrase.
"""
import math
import re
import unicodedata
from collections import Counter
from functools import lru_cache

import parser


ARTICLES = frozenset("il lo la i gli le l' un uno una un' questo questa questi queste quel quello quella quei quelle".split())
LINKS = ARTICLES | frozenset("di a da in su per con e".split())
NOUN_LINKS = ARTICLES | frozenset(('di', 'in', 'su'))
# Generic heads may introduce a more specific referent. They cannot identify
# one on their own. Unknown modifiers remain visible to the operation grammar.
GENERIC = frozenset("dato dati valore valori numero numeri quantità testo informazione informazioni elemento elementi campo campi voce voci coppia coppie due".split())


def _stem(token):
    token = ''.join(c for c in unicodedata.normalize('NFD', token) if not unicodedata.combining(c))
    if len(token) > 3 and token[-1] in 'aeio':
        token = token[:-1]
        if token.endswith(('ch', 'gh')):
            token = token[:-1]
    return token


NEUTRAL = frozenset(_stem(token) for token in GENERIC)


@lru_cache(maxsize=8)
def _lexicon(inventory):
    words = {}
    frequency = Counter()
    for lesson, name in inventory:
        content = {_stem(w) for w in parser._canonical(name).split() if w not in LINKS}
        words[lesson] = content
        frequency.update(content)
    weights = {key: math.log((1 + len(words)) / (1 + n)) for key, n in frequency.items()}
    return words, weights


def _resolve(query, families):
    if not isinstance(query, str) or not 0 < len(query) <= 512 or not query.isprintable():
        return None
    raw = unicodedata.normalize('NFC', query).strip().casefold().replace('’', "'")
    raw = re.sub(r'[?!.]$', '', raw)
    text = parser._canonical(raw)
    tokens = text.split()
    words, weights = _lexicon(tuple((f['id'], f['object_phrase']) for f in families))
    stems = [_stem(t) for t in tokens]
    evidence = {s for s in stems if s in weights and s not in NEUTRAL}
    if not evidence:
        return None
    # At least one bit of separation from a runner-up, with an explicit veto if
    # any of the discriminating words describes a competing object exclusively.
    scores = sorted(((sum(weights[s] for s in sorted(evidence & terms)), lesson)
                     for lesson, terms in words.items()), reverse=True)
    score, lesson = scores[0]
    if score < math.log(2) or (len(scores) > 1 and score - scores[1][0] < math.log(2) - 1e-12):
        return None
    if not evidence <= words[lesson]:
        return None
    exact = parser._bind(text, families)
    if exact is not None and exact[0]['id'] == lesson:
        return lesson, exact[1]
    indices = [i for i, s in enumerate(stems) if s in evidence]
    start, end = indices[0], indices[-1] + 1
    # Multiple copies of a name can identify an object but cannot bind one
    # well-defined slot. Leave quantification/comparisons to a different grammar.
    if len(indices) != len(evidence):
        return None
    joinable = [t in NOUN_LINKS or s in words[lesson] or s in NEUTRAL for t, s in zip(tokens, stems)]
    if not all(joinable[start:end]):
        return None
    while start and joinable[start - 1]:
        start -= 1
    # Leave outer "di/per/con" intact for the existing verb and topic rules.
    while tokens[start] in NOUN_LINKS - ARTICLES:
        start += 1
    while end < len(tokens) and joinable[end]:
        end += 1
    while end > indices[-1] + 1 and tokens[end - 1] in NOUN_LINKS:
        end -= 1
    return lesson, ' '.join([*tokens[:start], '@', *tokens[end:]])


def bind(query, families):
    """Identify the referred-to catalog object independently of the speech act."""
    result = _resolve(query, families)
    return result[0] if result else None


def predict(query, families):
    """Reuse the exact parser, then test its operation grammar with one inferred slot."""
    exact = parser.predict(query, families)
    if exact is not None:
        return exact
    result = _resolve(query, families)
    if result is None:
        return None
    raw = unicodedata.normalize('NFC', query).strip().casefold().replace('’', "'")
    if not re.fullmatch(r"[a-zàèéìòù0-9' ,:]+[.?!]?", raw):
        return None
    lesson, frame = result
    family = next(f for f in families if f['id'] == lesson)
    rule = parser._RULES.get(family['operation'])
    frame = ' '.join(frame.replace(',', ' ').split())
    if rule and any(rule.fullmatch(request) for request in parser._requests(frame)):
        return lesson
    return None
