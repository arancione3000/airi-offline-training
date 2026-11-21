"""Require a request for a value, a supported subject and an executable operation.

The lexical router underneath is frozen. This layer distinguishes requests to
carry out a procedure from mentions of it inside a request for other information.
It consumes the user's current question only.
"""
import re

import joint


def words(pattern):
    return re.compile(r'\b(?:' + pattern + r')\b')


# A reference to an action governed by a question about its representation,
# purpose or future use does not authorize running it.
_META = words(
    r'formula|equazione|espressione (?:matematica|aritmetica)|sintassi|regola|passaggi|'
    r'(?:quale|che|il|un) (?:segno|simbolo|metodo|operatore) (?:user\w*|serv\w*|occor\w*|si usa|si mette)|'
    r'(?:nome|tipo|scelta|definizione) dell\x27operazione|(?:quale|che) operazion[ei]|'
    r'a (?:che|cosa) serv\w*|a (?:che|quale) scopo|come (?:funzion\w*|si fa|si dice)|'
    r'(?:sai|sapresti) (?:come|se)|sapere come|conosci (?:il modo|la regola)|'
    r'sapere se|dire se|stabilire se|verificare se|valutare se|'
    r'(?:prima|invece) di|più avanti|piu avanti|per il futuro|in seguito|'
    r'(?:mi|ci) servirà|a titolo (?:di|informativo)|in teoria|in un (?:racconto|manuale|dialogo)|'
    r'(?:sto|stiamo) (?:parlando|discutendo|valutando)|intendev\w*|'
    r'(?:come|cosa) (?:risponder\w*|proceder\w*|farest\w*)|'
    r'(?:chiede|chiedeva|chiedesse|sostiene|racconta) di|(?:leggo|cit\w*) (?:che|in)|'
    r'conserva\w* per|memorizz\w*|annot\w* per|senza (?:eseguir\w*|risponder\w*|calcolar\w*)'
)

# Typical fields belonging to an entity, instead of the entity's stored value.
# For a customer's reference, for example, the reference of a related record is
# a separate field even if both the word "reference" and the customer occur.
_ATTRIBUTE = words(
    r'emittent[ei]|ente|emess[oaie]|fabbri\w*|costruttor[ei]|rilasciat\w*|'
    r'fattur[ae]|ricevut[ae] fiscal[ei]|documento fiscale|scadenz[ae]|'
    r'proprietari[oaie]|titolare|intestatari[oaie]|indirizzo (?:postale|fisico)|'
    r'color[ei]|peso|pesante|dimension[ei]|materiale|data di (?:nascita|emissione|consegna)|'
    r'prefisso|suffisso|second[oa] (?:lettera|carattere)|penultim\w*|'
    r'(?:ultim\w*|prim\w*) (?:due|tre|quattro|cinque|[0-9]+) (?:simbol\w*|caratter\w*)'
)

# Only a supported role or display-field relationship may attach a reference
# identifier to a nested possessor. Generic "X di Y del cliente" is ambiguous.
_NESTED = words(
    r'(?:riferiment[oi]|codic[ei]|identificativ[oi]|id)\s+'
    r'(?:dell\x27|(?:di|del|della|dello|dei|degli|delle)\s+)'
    r'(?:[a-zà-ù]+\s+){1,3}(?:del(?:la|lo|le)?|dei|degli|dell\x27)\s*'
    r'(?:client[ei]|ordin[ei]|pacc\w*|coll[oi]|tesser\w*|account|spedizion\w*)'
)
_EXTRA = words(
    r'(?:solo|soltanto) (?:la|il|le|i) (?:parol\w*|segn\w*|codic\w*) (?:che|con)|'
    r'suddivid\w*|raggrupp\w*|ordina (?:per|in ordine|dalla|dal)|'
    r'(?:spazi|punteggiatura) (?:compres\w*|esclus\w*)|'
    r'(?:metà|meta|radice|potenza|modulo aritmetico|resto della divisione)|'
    r'(?:con|usando|in) (?:due|tre|[0-9]+) decimal\w*|'
    r'(?:alla|al) (?:seconda|quadrato|cubo)|senza (?:gli spazi|ripetizioni)|'
    r'al posto (?:di|del|della|dei)|rispettivamente|scel[gt]\w* tra'
)

_OBJECT_FORMS = (
    (words(r'tesserin[oi]'), 'tessera'), (words(r'targhett[ae]'), 'targa'),
    (words(r'ordinazion[ei]'), 'ordine'), (words(r'cestin[oi] (?:degli|per gli) acquisti'), 'carrello'),
)
# Common descriptions of an operation using endpoints or paired operands.
_REVERSE = words(r'(?:dall\x27ultim[oa]|dalla fine)\b.{0,32}?(?:fino )?(?:al prim[oa]|alla prima|all\x27inizio)')
_PRODUCT = words(r'l\x27un[oa] per l\x27altr[oa]|un[oa] per l\x27altr[oa]')
_VALUE_QUERY = words(
    r'(?:me|ce) (?:lo|la|li|le) (?:passi|dai|invia\w*|scrivi|mostri)|'
    r'(?:con quale|quale) (?:riferimento|identificativo|nominativo|sigla|nome|codice)|'
    r'(?:mi|me lo|me la) (?:scrivi|mandi|giri|trasmetti)|'
    r'(?:nel modo|nella forma|nella grafia|così|esattamente) (?:in cui|come)'
)


def bind(query, families):
    text = joint._text(query)
    if not text or _ATTRIBUTE.search(text) or _NESTED.search(text):
        return None
    # "ordine dei caratteri" is an ordering, not an order in the sales ledger.
    text = re.sub(r"\b(?:l'|in )?ordine (?:di|dei|delle) (?:ogni |tutti i )?(?:caratteri|lettere|simboli)\b",
                  'sequenza dei caratteri', text)
    for pattern, canonical in _OBJECT_FORMS:
        text = pattern.sub(canonical, text)
    return joint.bind(text, families)


def action(query):
    text = joint._text(query)
    if not text or _META.search(text) or _EXTRA.search(text):
        return None
    # Restrict broad verb stems inherited from the base model: the nouns
    # "inventario" and "nome alternativo" do not request invention or alternation.
    text = re.sub(r'\binventari[oi]\b', 'magazzino', text)
    text = re.sub(r'\bnome alternativo\b', 'alias', text)
    text = re.sub(r'\bdue conteggi\b', 'due quantità', text)
    # Expand these phrasings into the same decision path (including every veto).
    text = _REVERSE.sub('in ordine inverso', text)
    text = _PRODUCT.sub('moltiplicati', text)
    text = _VALUE_QUERY.sub('dimmi', text)
    return joint.action(text)


def predict(query, families):
    requested = action(query)
    if requested is None:
        return None
    source = bind(query, families)
    if source and any(f['id'] == source and f['operation'] == requested for f in families):
        return source
    return None
