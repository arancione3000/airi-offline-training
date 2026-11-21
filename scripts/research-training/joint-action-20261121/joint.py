"""Factor a bounded Italian request into an explicit referent and a requested action.

This research adapter has an authored domain vocabulary. Neither detector sees the
source turn or uses the predicted family to fill in a missing action. A new family
outside the vocabulary requires separate evaluation.
"""
import re
import unicodedata
from functools import lru_cache


def _text(query):
    if not isinstance(query, str) or not 0 < len(query) <= 512 or not query.isprintable():
        return ''
    return unicodedata.normalize('NFC', query.casefold()).strip().replace('’', "'")


def _rx(pattern):
    return re.compile(r'\b(?:' + pattern + r')\b')


# Domain concepts shared between related objects. Unqualified terms such as
# "codice", "alias" and "riferimento" cannot identify one of them.
_PARCEL = _rx(r'pac(?:co|chi)|coll[oi]|pacchett[oi]')
_IDENTIFIER = _rx(r'codic[ei]|identificativ[oi]|identificator[ei]|riferiment[oi]|sigl[ae]|id')
_SHIPPING = _rx(r'spedizion[ei]|spedire|spedito|invio|consegna')
_LABEL = _rx(r'etichett[ae]|cartellin[oi]|talloncino|adesivo')
_ORDER = _rx(r'ordin[ei]|ordinativ[oi]|commess[ae]')
_PERSON = _rx(r'referent[ei]|contatt[oi]|incaricat[oa]|responsabil[ei]|nominativo|persona|interlocutor[ei]')
_CUSTOMER = _rx(r'client[ei]|clientela|acquirente')
_TOKEN = _rx(r'token|gettone|codic[ei]|chiave|stringa|sequenza')
_VERIFY = _rx(r'verifica|verificare|convalida|validazione|autenticazione|controllo')
_ACCOUNT = _rx(r'account|utenz[ae]|profil[oi]|login')
_ALIAS = _rx(r'alias|pseudonim[oi]|soprannom[ei]|nome (?:alternativo|utente)|nomignolo|nickname|identificativ[oi]')
_EMAIL = _rx(r'e[ -]?mail|mail|posta elettronica|casell[ae] di posta|indirizzo di posta')
_PRICE = _rx(r'prezz[oi]|cost[oi]|tariff[ae]')
_UNIT = _rx(r'unitari[oa]|unità|unita|pezz[oi]|articol[oi]|cadauno|singol[oa]|esemplare')
_SIMPLE = {
    'signal_reverse': _rx(r'segnal[ei]'),
    'card_upper': _rx(r'tesser[ae]|badge'),
    'plate_lower': _rx(r'targ(?:a|he)'),
    'delta_sum': _rx(r'delta'),
    'sigma_product': _rx(r'sigma'),
    'auto_v1_remaining': _rx(r'sald[oi]'),
    'auto_v1_stock_remaining': _rx(r'scort[ae]|giacenz[ae]'),
    'auto_v1_cart_total': _rx(r'carrell[oi]'),
    'auto_v1_inventory_total': _rx(r'inventari[oi]|inventarial[ei]'),
}


@lru_cache(maxsize=1024)
def _referents(text):
    found = {label for label, pattern in _SIMPLE.items() if pattern.search(text)}
    person = _PERSON.search(text)
    identifier = _IDENTIFIER.search(text)
    label = _LABEL.search(text)
    parcel = _PARCEL.search(text)
    order = _ORDER.search(text)
    if identifier and (parcel or _SHIPPING.search(text)) and not label:
        found.add('parcel_copy')
    if label and (parcel or _SHIPPING.search(text)):
        found.add('auto_v1_label_upper')
    if person and order:
        found.add('auto_v1_contact')
    if identifier and order and not person:
        found.add('auto_v1_order_reference')
    if identifier and _CUSTOMER.search(text):
        found.add('auto_v1_customer_reference')
    if _TOKEN.search(text) and _VERIFY.search(text):
        found.add('auto_v1_tracking_reverse')
    if _ALIAS.search(text) and _EMAIL.search(text):
        found.add('auto_v1_email_alias')
    if _ALIAS.search(text) and _ACCOUNT.search(text):
        found.add('auto_v1_account_lower')
    if _PRICE.search(text) and _UNIT.search(text):
        found.add('auto_v1_unit_total')
    return frozenset(found)


def bind(query, families):
    """Resolve the subject of the question without inferring it from a source or action."""
    text = _text(query)
    found = _referents(text)
    if len(found) == 1:
        lesson, = found
        if any(f['id'] == lesson for f in families):
            return lesson
    return None


# These requests ask about part of a value, another output, the wording of a
# question or a condition. A known action word inside them is insufficient.
_BLOCK = _rx(
    r'non|mai|evit\w*|nega\w*|anziché|anziche|invece|oppure|o|forse|'
    r'se|qualora|purché|purchè|suppon\w*|ipot\w*|immagina\w*|a patto|nel caso|'
    r'frase|citazion\w*|virgolette|titolo|esercizio|lingua|significa\w*|'
    r'spiega\w*|descriv\w*|descrizion\w*|metodo|procedimento|algoritmo|istruzion\w*|'
    r'possibil[ei]|capace|capacità|direbbe|ha (?:detto|scritto)|mi ha (?:chiesto|detto)|'
    r'poi|successivamente|anche|infine|inoltre|separat\w*|separa\w*|'
    r'prima lettera|primo carattere|ultim[oaie] (?:\w+ )?(?:letter[ae]|cifr[ae])|'
    r'soltanto (?:il|la|i|le) (?:prim\w*|ultim\w*)|vocal[ei]|consonant\w*|'
    r'cifra (?:delle|dei) (?:unità|decine|centinaia)|una (?:lettera|cifra) sì|alternat\w*|'
    r'inizial[ei]|lung\w*|contien\w*|conta(?:re|mi|ndo|li|le)?|contegg\w*|quanti|quante|'
    r'preval\w*|più spesso|alfabetic\w*|trattin\w*|asterisc\w*|'
    r'sostitu\w*|esclud\w*|tranne|eccetto|senza (?:il|la|i|le|gli)|'
    r'supera\w*|maggiore|minore|massim\w*|minim\w*|medi[ao]|'
    r'coincid\w*|uguali|confront\w*|pari|dispari|distanza|assolut\w*|'
    r'divid\w*|division\w*|quozient\w*|raddopp\w*|dimezz\w*|'
    r'arrotond\w*|a caso|casual\w*|percentual\w*|random|formatt\w*|'
    r'perché|perche|quando|eventualmente|ipoteticamente|defini\w*|citando|raccont\w*|'
    r'potrei|equivalen\w*|esempio|esempi|inventa\w*|ignora\w*|propon\w*|consigli\w*|traduc\w*|conferma\w*|'
    r'filtra\w*|scambia\w*|duplic\w*|salta\w*|lascia (?:fuori|ferma|fermo)|'
    r'parzial\w*|tenendo ferm\w*|due volte|rimuo\w*|elimin\w*|scart\w*'
)
_EXTRA_ADD = _rx(r'aggiung\w*|accod\w*|antepon\w*|in coda|in testa')
_REVERSE = _rx(r'invert\w*|invers[oaie]|rovesc\w*|ribalt\w*|capovol\w*|contrario|ritroso|'
               r'indietro|retrograd\w*|dalla fine|partendo dall[\x27 ]ultim\w*|'
               r'dall[\x27 ]ultim\w* (?:\w+ )?(?:a|al|alla) (?:prim\w*)|destra (?:verso|a) sinistra|ordine opposto')
_UPPER = _rx(r'maiuscol\w*|lettere (?:tutte )?grandi|capitali')
_LOWER = _rx(r'minuscol\w*|lettere (?:tutte )?piccole')
_ADD = _rx(r'somm\w*|addizion\w*|in tutto|complessivamente|mette(?:ndo|re) insieme|'
           r'quanto (?:fanno|danno|valgono|vengono) insieme|primo\b.{0,55}\bpiù (?:il |la )?second\w*')
_PRODUCT = _rx(r'moltipl\w*|prodott[oi]|prim[oa]\b.{0,45}\bper (?:il |la )?second\w*|'
               r'(?:prezzo|costo)(?: unitario)? per (?:la )?quantità|(?:costo|spesa|prezzo) (?:complessiv[oa]|total[ei])')
_TOTAL = _rx(r'totale|totalizz\w*')
_SUBTRACT = _rx(r'(?:prim[oa]\b.{0,55}\bmeno\b.{0,36}\b(?:second[oa]|quell[oa] che (?:lo|la) segue))|'
                r'(?:togl\w*|sottra\w*|detr\w*)\b.{0,28}\bsecond[oa]\b.{0,28}\bdal(?:la)? prim[oa]|'
                r'dal(?:la)? prim[oa]\b.{0,48}\b(?:togl\w*|sottra\w*|detr\w*)\b.{0,16}\bsecond[oa]|'
                r'prim[oa]\b.{0,50}\b(?:sottraigli|sottraile|togligli|toglile) (?:il |la )?second[oa]')
_COPY = _rx(r'(?:ri)?copi\w*|(?:ri)?trascriv\w*|ripeti\w*|riprodu\w*|fedelmente|identic\w*|'
            r'pari pari|tale e quale|invariat\w*|inalterat\w*|senza modificar\w*|'
            r'(?:qual|quale|chi) è|quale (?:alias|nome|codice)|che (?:nome|codice|identificativo)|'
            r'(?:ri)?dammi|darmi|dirmi|dimmi|riport\w*|indicami|mostr\w*|'
            r'(?:mi|me lo) (?:serve|passi|occorre|dai|ridai|detti)|'
            r'vorrei (?:conoscere|sapere|vedere)|leggi\w*|dare|recuper\w*|'
            r'fammi (?:leggere|vedere)|passam\w*|indic(?:hi|armi)|leggermi|ripeter\w*|riprendi|'
            r'come si chiama|ho bisogno (?:del nome|di sapere)|desidero (?:vedere|sapere|conoscere)')

# A second request can add a different output without naming a second operation.
_SUBREQUEST = _rx(r'(?:e|ma) (?:inoltre |quindi |successivamente )?(?:'
                  r'rispondi a|scrivi(?:mi)? (?:un|una)|calcola(?:mi)? (?:ancora|pure|a parte)|'
                  r'fornisci due|fornisci un altro|dimmi (?:se|quanti|quante|quanto è lungo)|'
                  r'scegli\w*|formatta\w*|evidenzia\w*|sottolinea\w*|segnala\w*)')


def action(query):
    """Detect a single requested transformation, independent of eligible skills."""
    text = _text(query)
    if not text or re.search(r'["«»“”‘`<>\[\]{}|]', text) or re.search(r"(?:^|\s)'|'(?:\s|$)", text):
        return None
    # 'pari pari' describes a verbatim copy. A lone 'pari' asks about parity.
    guarded = text.replace('pari pari', 'fedelmente')
    if _BLOCK.search(guarded) or _EXTRA_ADD.search(guarded) or _SUBREQUEST.search(guarded):
        return None
    ops = {name for name, pattern in (('reverse', _REVERSE), ('uppercase', _UPPER), ('lowercase', _LOWER),
                                      ('add', _ADD), ('multiply', _PRODUCT), ('subtract', _SUBTRACT))
           if pattern.search(text)}
    if _TOTAL.search(text) and 'multiply' not in ops:
        # A request for the total monetary cost of a number of units is a product.
        ops.add('multiply' if _PRICE.search(text) and _UNIT.search(text) else 'add')
    if len(ops) > 1:
        return None
    if ops:
        return next(iter(ops))
    return 'copy' if _COPY.search(text) else None


def predict(query, families):
    """Require independently identified procedure and referent to agree."""
    operation = action(query)
    if operation is None:
        return None
    target = bind(query, families)
    if target and any(f['id'] == target and f['operation'] == operation for f in families):
        return target
    return None
