"""Conservative Italian request grammar for operations with a named object.

Each production must consume the entire utterance. A unique object occurrence
binds one installed family; a separately parsed operation must agree with it.
The grammar carries no information about source values or program execution.
"""
import re
import unicodedata


def _alternatives(*forms):
    return '(?:' + '|'.join(forms) + ')'


# Imperatives, infinitives and dative clitics used in ordinary requests.
_SHOW = _alternatives(
    'dimmi', 'dirmi', 'dire', 'mostra(?:mi|re|rmi)?', 'indica(?:mi|re|rmi)?',
    'riporta(?:mi|re|rmi)?', 'forni(?:sci(?:mi)?|re|rmi)',
    'restitui(?:sci(?:mi)?|re|rmi)', 'dammi', 'darmi', 'dare',
    'comunica(?:mi|re|rmi)?', 'fammi vedere', 'farmi vedere',
    'fammi sapere', 'farmi sapere', 'conoscere', 'sapere', 'vedere',
)
_READ = _alternatives('leggi(?:mi)?', 'leggere', 'leggermi', 'trascrivi', 'trascrivere',
                     'copia', 'copiare', 'ripeti', 'ripetere', 'recupera', 'recuperare')
_WRITE = _alternatives('scrivi', 'scrivere', 'riscrivi', 'riscrivere',
                      'converti', 'convertire', 'trasforma', 'trasformare',
                      'metti', 'mettere', 'porta', 'portare')
_ASK = _alternatives('qual è', 'quale sarebbe', 'quanto fa', 'quanto vale',
                    'a quanto ammonta', 'quanto viene', 'mi dici', 'mi dai')
_GET = _alternatives(_SHOW, _ASK, 'calcola(?:mi|re|rmi)?', 'trova(?:mi|re|rmi)?',
                    'ottieni', 'ottenere', 'esegui', 'eseguire', 'effettua', 'effettuare')
_REL = _alternatives(':', 'di', 'per', 'tra', 'fra', 'su', 'con', 'usando', 'relativo a',
                    'ottenuto da', 'che si ottiene con')
_JOIN = rf'(?: {_REL})? @'
_FROM = r'(?: (?:dal contesto|dal messaggio(?: precedente)?|come indicato|così com\x27è))?'
# A request may use a polite outer clause around a direct clause, such as
# "potresti dirmi quanto vale il prodotto ...".
_POLITE = re.compile(r'^(?:per favore|per cortesia|per piacere|gentilmente|cortesemente) ')
_END_POLITE = re.compile(r' (?:per favore|per cortesia|per piacere|grazie)$')
_MODAL = re.compile(
    r'^(?:(?:mi )?(?:puoi|potresti|potrebbe|potete|potreste)|'
    r'(?:ti|le) (?:chiedo|domando) di|(?:vorrei|voglio|desidero)|'
    r'(?:mi (?:occorre|serve|interessa)|ho bisogno di|avrei bisogno di)) '
)
_EMBED = re.compile(r'^(?:' + _SHOW + r') (?=(?:qual è|quanto |a quanto |quale ))')


def _noun(noun):
    return rf'(?:{_GET} )?{noun}{_JOIN}'


def _patterns():
    rules = {
        'copy': [
            rf'(?:{_SHOW}|{_READ}|{_ASK}) @{_FROM}',
            rf'@{_FROM}',
            rf'(?:{_SHOW}|{_READ}) @ (?:senza modificarl[oa]|tale e quale|alla lettera)',
        ],
        'reverse': [
            r'(?:inverti|invertire|capovolgi|capovolgere|rovescia|rovesciare) @'
            r'(?: carattere per carattere)?',
            rf'(?:{_SHOW} |{_WRITE} |{_READ} |riordina |riordinare )?@ '
            r'(?:al contrario|a ritroso|alla rovescia|al rovescio|in ordine (?:inverso|rovesciato)'
            r'|da destra (?:a|verso) sinistra|con i caratteri invertiti|invertit[oa])',
        ],
        'add': [
            r'(?:somma|sommare|addiziona|addizionare|aggiungi|aggiungere) @(?: (?:tra (?:di )?loro|insieme))?',
            _noun(r'(?:la somma|il totale(?: complessivo)?|il risultato (?:della somma|dell\x27addizione))'),
            r'quanto (?:fa(?:nno)?|vengono) (?:in tutto )?@ (?:in tutto|insieme|sommat[ie])',
        ],
        'multiply': [
            r'(?:moltiplica|moltiplicare) @(?: tra (?:di )?loro)?',
            _noun(r'(?:il prodotto|la moltiplicazione|il risultato (?:del prodotto|della moltiplicazione))'),
            r'quanto fanno @ moltiplicati(?: tra (?:di )?loro)?',
        ],
        'subtract': [
            rf'(?:{_GET} )?(?:il )?primo(?: valore)? meno (?:il )?secondo(?: valore)?{_JOIN}',
            rf'(?:sottrai|sottrarre|togli|togliere) il secondo(?: valore)? dal primo(?: valore)?{_JOIN}',
            rf'(?:{_GET} )?(?:la )?differenza{_JOIN} (?:primo meno secondo|in ordine di presentazione)',
        ],
    }
    for op, case in [('uppercase', 'maiuscol'), ('lowercase', 'minuscol')]:
        form = rf'(?:in (?:lettere )?{case}[oe]|(?:tutt[oa] )?{case}[oa]|'
        form += rf'(?:usando|con) (?:solo |tutte le )?(?:lettere )?{case}e|nella forma {case}a)'
        rules[op] = [
            rf'(?:{_SHOW} |{_WRITE} )?@ (?:scritt[oa] |convertit[oa] )?{form}',
            rf'(?:{_SHOW}|{_WRITE}|rendi|rendere) {form} @',
            rf'(?:usa|usare) (?:solo )?lettere {case}e per @',
            rf'(?:{_SHOW} )?la versione {case}a{_JOIN}',
        ]
    return {op: re.compile(_alternatives(*ruleset)) for op, ruleset in rules.items()}


_RULES = _patterns()
_OBJECT_ARTICLES = {
    'il ': ('del ', 'di '), 'i ': ('dei ', 'di '),
    'le ': ('delle ', 'di '), 'la ': ('della ', 'di '), "l'": ("dell'", 'di '),
}


def _bind(text, families):
    matches = []
    for family in families:
        obj = family['object_phrase'].lower().replace('’', "'")
        forms = [(obj, '@')]
        for article, (prefix, relation) in _OBJECT_ARTICLES.items():
            if obj.startswith(article):
                forms.append((prefix + obj[len(article):], relation + '@'))
                break
        for literal, slot in forms:
            pattern = r"(?<![\w'])" + re.escape(literal) + r"(?![\w'])"
            for match in re.finditer(pattern, text):
                matches.append((family, text[:match.start()] + slot + text[match.end():]))
                if len(matches) > 1:
                    return None
    return matches[0] if matches else None


def predict(query, families):
    if not isinstance(query, str) or not 0 < len(query) <= 512 or not query.isprintable():
        return None
    text = unicodedata.normalize('NFC', query).strip().lower().replace('’', "'")
    text = text.removesuffix('?') if text.endswith('?') else re.sub(r'[.!]$', '', text)
    if not re.fullmatch(r"[a-zàèéìòù0-9',: ]+", text):
        return None
    bound = _bind(text, families)
    if bound is None:
        return None
    family, frame = bound
    # Retain a colon as an explicit grammatical relation. An unrecognized word
    # or clause survives this normalization and makes fullmatch fail.
    frame = ' '.join(frame.replace(',', ' ').replace(':', ' : ').split())
    frame = _END_POLITE.sub('', _POLITE.sub('', frame))
    request = _MODAL.sub('', frame)
    request = _EMBED.sub('', request)
    # A bare object identifies a copy request only after an explicit request
    # prefix such as "vorrei" or "mi serve".
    if request == frame and request.startswith('@'):
        return None
    rule = _RULES.get(family['operation'])
    if rule and rule.fullmatch(request):
        return family['id']
    return None
