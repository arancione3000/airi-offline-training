"""Offline Italian intent recognizer for one installed object and one operation.

The caller supplies the instruction turn and the public family descriptions.
Each successful derivation consumes the whole instruction. Request, operand and
operation are parsed independently; the installed program handles its own data.
"""
import re
import unicodedata
from functools import lru_cache


def _forms(*variants):
    return '(?:' + '|'.join(variants) + ')'


# Expand only grammatical contractions at word boundaries, on both the catalog
# and the request. Retain the preposition as its own token, including direction.
_CONTRACTIONS = {}
for _prep, _forms_by_article in {
    'di': ('del', 'dello', 'della', 'dei', 'degli', 'delle', "dell'"),
    'a': ('al', 'allo', 'alla', 'ai', 'agli', 'alle', "all'"),
    'da': ('dal', 'dallo', 'dalla', 'dai', 'dagli', 'dalle', "dall'"),
    'in': ('nel', 'nello', 'nella', 'nei', 'negli', 'nelle', "nell'"),
    'su': ('sul', 'sullo', 'sulla', 'sui', 'sugli', 'sulle', "sull'"),
}.items():
    for _form, _article in zip(_forms_by_article, ('il', 'lo', 'la', 'i', 'gli', 'le', "l'")):
        _CONTRACTIONS[_form] = _prep + ' ' + _article

_WORD = re.compile(r"[a-zàèéìòù0-9]+'?|[':,]")


def _canonical(text):
    text = unicodedata.normalize('NFC', text.casefold()).replace('’', "'")
    tokens = _WORD.findall(text)
    return ' '.join(_CONTRACTIONS.get(token, token) for token in tokens)


@lru_cache(maxsize=256)
def _object_pattern(phrase):
    return re.compile(r'(?<!\S)' + re.escape(_canonical(phrase)) + r'(?!\S)')


def _bind(text, families):
    bound = None
    for family in families:
        for occurrence in _object_pattern(family['object_phrase']).finditer(text):
            if bound is not None:
                return None
            frame = text[:occurrence.start()] + '@' + text[occurrence.end():]
            bound = family, frame
    return bound


_POLITE = _forms('per (?:favore|cortesia|piacere|gentilezza)', 'gentilmente', 'cortesemente')
_REQUEST = _forms(
    r'(?:tu )?(?:mi )?(?:puoi|potresti|potrebbe|potete|potreste|riesci a|riusciresti a)',
    r'(?:ti|le|vi) (?:chiedo|domando) di', r'(?:avrei|ho) bisogno di',
    r'(?:vorrei|voglio|desidero)(?: che tu)?', r'mi (?:serve|servirebbe|occorre|occorrerebbe)',
    r'(?:mi )?faresti (?:il favore|la cortesia) di',
    r'(?:saresti (?:così |tanto )?gentile da|ti dispiacerebbe|mi piacerebbe|è possibile)',
)
_DELIVER = _forms(
    r'dimmi|dirmi|dammi|darmi|mi dai|mi dici|mi diresti|mi scrivi',
    r'(?:mostra|riporta|indica|comunica|presenta|manda|passa|elenca|fornisci|restituisci)'
    r'(?:mi)?',
    r'(?:mostra|riporta|indica|comunica|presenta|manda|passa|elenca|forni|restitui)r(?:e|mi)',
    r'(?:fammi|farmi) (?:vedere|avere|sapere|leggere)', r'(?:vedere|avere|conoscere|sapere)',
)
_WRITE = _forms(r'(?:ri)?scriv(?:i|imi|ere|ermi)', r'(?:trasforma|porta|stampa|visualizza|rappresenta)(?:mi|re|rmi)?',
                r'convert(?:i|imi|ire|irmi)', r'rend(?:i|imi|ere|ermi)', r'mett(?:i|imi|ere|ermi)')
_CALCULATE = _forms(r'(?:calcola|calcola(?:re|mi|rmi)|determina(?:re|mi|rmi)?|trova(?:re|mi|rmi)?)',
                    r'(?:esegui|eseguire|ottieni|ottenere|ricava|ricavare|effettua|effettuare)')
_QUESTION = _forms(r'(?:(?:qual è|quale (?:è|sarebbe)|quanto (?:è|fa|vale|viene|risulta))'
                   r'(?: il (?:valore|risultato) di)?|a quanto (?:ammonta|corrisponde))')
_OUTPUT = _forms(_DELIVER, _WRITE, _CALCULATE, _QUESTION)
_COPY = _forms(r'(?:ri)?copi(?:a|ami|are|armi)', r'(?:ri)?trascriv(?:i|imi|ere|ermi)',
               r'ripet(?:i|imi|ere|ermi)', r'(?:ri)?legg(?:i|imi|ere|ermi)', r'recupera(?:mi|re|rmi)?')
_EXACT = _forms(
    r'(?:così |esattamente |proprio )?com\x27 è', r'(?:così |esattamente )?come (?:appare|compare|riportat[oa])',
    r'(?:esattamente |proprio )?(?:tale e quale|alla lettera|pari pari|fedelmente|integralmente|testualmente)',
    r'(?:senza (?:alcuna |nessuna )?modific(?:a|he)|senza (?:cambiar|modificar|alterar)l[oa])',
    r'(?:in forma originale|in la forma originale|inalterat[oa]|invariat[oa]|intatt[oa])',
)
_SOURCE = _forms(r'da il (?:messaggio|testo|contesto)(?: (?:precedente|originale|fornito))?',
                 r'(?:già )?(?:indicat|fornit|riportat|specificat)[oa] (?:prima|sopra|in precedenza)')
_REL = _forms('di', 'per', 'su', 'con', 'tra', 'fra', 'usando', 'relativ[oa] a', 'ottenut[oa] da', ':')
_OPERAND = rf'(?:{_REL} )?@'
_CHAR = r'(?:i caratteri|le lettere|i simboli)'
_BACK = _forms(
    r'a ritroso|a l\x27 inverso|a l\x27 indietro', r'a il contrario|a la rovescia|a rovescio',
    r'(?:in|in il) (?:senso|ordine) (?:inverso|contrario|opposto|rovesciato|invertito)',
    r'da (?:destra a sinistra|l\x27 (?:ultimo|ultima) (?:a il primo|a la prima))',
    r'(?:a partire |partendo )?da (?:la fine a il principio|la fine a l\x27 inizio|l\x27 ultimo a il primo)',
)
_REVERSE = _forms(r'capovolg(?:i|imi|ere|ermi)', r'(?:rovescia|ribalta|rigira)(?:mi|re|rmi)?',
                  r'invert(?:i|imi|ire|irmi)')
_N1 = r'(?:(?:il |la )?(?:primo|prima)(?: (?:numero|valore|termine|quantità|cifra|operando))?)'
_N2 = _N1.replace('primo|prima', 'secondo|seconda')
_DIFFERENCE = _forms(rf'{_N1} meno {_N2}', rf'(?:sottrai|sottrarre|sottraendo|togli|togliere|detrai|detrarre) {_N2} da {_N1}',
                    rf'da {_N1} (?:sottrai|togli|detrai) {_N2}')
_PAIRED = r'(?:(?:tra|fra) (?:di )?loro|insieme|l\x27 uno (?:a l\x27 altro|per l\x27 altro))'
# Nominal operations can take an output or evaluation verb, as in a request
# to display a result, independently of the noun's underlying operation.
_NOMINAL_HEAD = rf'(?:{_OUTPUT} )?(?:(?:il (?:risultato|valore) di) )?'


def _compile(*patterns):
    return re.compile(_forms(*patterns))


_RULES = {
    'copy': _compile(
        rf'(?:{_COPY}|{_DELIVER}|{_QUESTION}) (?:{_EXACT} )?@(?: {_EXACT})?(?: {_SOURCE})?',
        rf'(?:{_WRITE} )?@ {_EXACT}', rf'{_COPY} il (?:testo|contenuto|valore) di @',
        rf'(?:{_DELIVER}|{_WRITE}) (?:la copia|il contenuto|il valore|il testo) di @(?: {_EXACT})?',
    ),
    'reverse': _compile(
        rf'{_REVERSE} (?:l\x27 ordine di )?(?:{_CHAR} di )?@(?: carattere per carattere)?',
        rf'(?:(?:{_OUTPUT}|{_COPY}) )?(?:{_CHAR} di )?@ (?:lett[oa] |scritt[oa] )?{_BACK}',
        rf'(?:{_OUTPUT}|{_COPY}) {_BACK} (?:{_CHAR} di )?@',
        rf'{_NOMINAL_HEAD}(?:la (?:versione|sequenza) (?:inversa|rovesciata|invertita) di )@',
    ),
    'add': _compile(
        rf'(?:somma(?:re|mi|rmi)?|addiziona(?:re|mi|rmi)?|metti insieme) @(?: {_PAIRED})?',
        rf'{_NOMINAL_HEAD}(?:la somma|l\x27 addizione|il totale(?: (?:complessivo|generale|combinato))?) {_OPERAND}',
        rf'(?:{_QUESTION}|quanto (?:fanno|danno)) @ (?:in (?:tutto|totale)|complessivamente|sommat[ie](?: {_PAIRED})?)',
        rf'(?:{_OUTPUT} )?@ (?:sommat[ie]|addizionat[ie])(?: {_PAIRED})?',
    ),
    'multiply': _compile(
        rf'moltiplica(?:re|mi|rmi)? @(?: {_PAIRED})?',
        rf'{_NOMINAL_HEAD}(?:il prodotto(?: aritmetico)?|la moltiplicazione) {_OPERAND}',
        rf'(?:(?:{_OUTPUT}|quanto fanno) )?@ moltiplicat[ie](?: {_PAIRED})?',
    ),
    'subtract': _compile(
        rf'(?:{_OUTPUT} )?{_DIFFERENCE} {_OPERAND}',
        rf'(?:{_OUTPUT} )?@ (?:facendo |calcolando )?{_DIFFERENCE}',
        rf'{_NOMINAL_HEAD}(?:la differenza|la sottrazione) {_OPERAND} (?:{_DIFFERENCE}|'
        r'(?:in|in il) ordine (?:dato|indicato|di presentazione|originale))',
    ),
}
for _operation, _stem in (('uppercase', 'maiuscol'), ('lowercase', 'minuscol')):
    _case = _forms(
        rf'(?:in |con |a )?(?:sol[oe] |tutte (?:le )?)?(?:lettere |caratteri )?{_stem}[aeio]',
        rf'(?:in formato|in forma|in la forma|in la versione|con (?:le|i)) (?:lettere |caratteri )?{_stem}[aeio]',
    )
    _RULES[_operation] = _compile(
        rf'(?:(?:{_OUTPUT}|{_COPY}) )?@ (?:tutt[oa] |interamente |completamente )?(?:scritt[oa] |convertit[oa] )?{_case}',
        rf'(?:{_OUTPUT}|{_COPY}) (?:tutt[oa] |interamente )?{_case} @',
        rf'(?:{_OUTPUT} )?(?:tutt[ei] )?{_CHAR} di @ (?:tutt[ei] )?{_case}',
        rf'{_NOMINAL_HEAD}la (?:versione|forma|scrittura|trascrizione) (?:{_case} di @|di @ {_case})',
        rf'(?:usa|usare|impiega|impiegare) {_case} (?:per|nello scrivere|per scrivere) @',
    )

# A topic establishes one operand. A subsequent direct-object clitic refers
# to that operand only when its host verb is a recognized command.
_TOPIC = re.compile(r'^(?:(?:per|con|di|su|usando|considerando|dato|data|'
                    r'partendo da) @ (?:[: ]+)?|(?:prendi|considera) @ (?:e |: ))(.+)$')
_CLITIC = re.compile(r'^(?P<verb>(?:ri)?scrivi|(?:ri)?copia|trascrivi|leggi|'
                     r'riporta|indica|mostra|inverti|rovescia|capovolgi|'
                     r'converti|trasforma|moltiplica|somma|addiziona)'
                     r'(?:lo|la|li|le)(?= |$)')


def _requests(frame):
    frame = re.sub(rf'^{_POLITE} ', '', frame)
    frame = re.sub(rf' (?:{_POLITE}|grazie)$', '', frame)
    frame = re.sub(rf'^{_REQUEST} ', '', frame)
    frame = re.sub(rf'^{_POLITE} ', '', frame)
    # An output verb can introduce an indirect question.
    frame = re.sub(rf'^{_DELIVER} (?={_QUESTION} )', '', frame)
    frame = re.sub(rf'^({_DELIVER}) (?:soltanto|esclusivamente|solo) ', r'\1 ', frame)
    yield frame
    topic = _TOPIC.fullmatch(frame)
    if topic:
        # An overtly fronted operand leaves an implicit operand in the clause.
        # Try the two grammatical positions, preserving every remaining word.
        predicate = topic.group(1)
        explicit, count = _CLITIC.subn(r'\g<verb> @', predicate)
        if count:
            yield explicit
            return
        yield predicate + ' @'
        yield '@ ' + predicate


def predict(query, skills):
    """Return the unique compatible installed family ID, or abstain."""
    if not isinstance(query, str) or not 0 < len(query) <= 512 or not query.isprintable():
        return None
    query = unicodedata.normalize('NFC', query).strip().casefold().replace('’', "'")
    query = re.sub(r'[?!.]$', '', query)
    if not re.fullmatch(r"[a-zàèéìòù0-9' ,:]+", query):
        return None
    bound = _bind(_canonical(query), skills)
    if not bound:
        return None
    family, frame = bound
    rule = _RULES.get(family['operation'])
    frame = ' '.join(frame.replace(',', ' ').split())
    if rule and any(rule.fullmatch(request) for request in _requests(frame)):
        return family['id']
    return None
