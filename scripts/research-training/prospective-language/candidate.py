"""Derive full instruction frames from verified aliases and recombine with typed objects."""
import re
import unicodedata
from collections import defaultdict


def tokenize(text):
    if not isinstance(text, str) or not 0 < len(text) <= 512 or not text.isprintable():
        return None
    text = unicodedata.normalize('NFC', text).casefold().replace('’', "'").strip()
    if not re.fullmatch("[a-zà-öø-ÿ' ,:?!.]+", text):
        return None
    if re.search(r"[!?.]", text.rstrip('!?.')) or re.search(r"(?:^| )'|'(?: |$)", text):
        return None
    return ' '.join(re.findall(r"[a-zà-öø-ÿ]+", text))


# Independently specified whole noun phrases. Possessors, fields and modifiers
# outside the noun phrase stay visible to the instruction matcher.
NOUNS = {
    'parcel_copy': r'(?:codice|identificativo|numero) (?:del (?:pacco|collo)|della spedizione)',
    'signal_reverse': r'(?:segnale|sequenza del segnale)(?: acquisit[oa]| registrat[oa]| ricevut[oa])?',
    'card_upper': r'(?:testo|scritta|dicitura|parole) (?:sulla|della|riportat[aeo] sulla) (?:tessera|carta)',
    'plate_lower': r'(?:testo|scritta|dicitura|parole) (?:sulla|della|riportat[aeo] sulla) (?:targa|targhetta)',
    'delta_sum': r'(?:due )?(?:valori|numeri|interi|dati|addendi)(?: di)? delta|coppia delta',
    'sigma_product': r'(?:due )?(?:valori|numeri|interi|dati|fattori)(?: di)? sigma|coppia sigma',
    'auto_v1_remaining': r'(?:due )?(?:valori|numeri|importi|cifre) del saldo',
    'auto_v1_contact': r'(?:referente|contatto|responsabile|nominativo del referente) (?:dell ordine|della commessa)',
    'auto_v1_customer_reference': r'(?:riferimento|codice|identificativo) (?:del )?cliente',
    'auto_v1_order_reference': r'(?:riferimento|codice|identificativo) (?:dell ordine|ordine)',
    'auto_v1_email_alias': r'(?:alias|nome alternativo|pseudonimo) (?:email|e mail|di posta elettronica)',
    'auto_v1_tracking_reverse': r'(?:token|codice) (?:di verifica|di controllo)',
    'auto_v1_label_upper': r'(?:testo (?:dell |sull ))?etichetta (?:di spedizione|del pacco|della spedizione)',
    'auto_v1_account_lower': r'(?:alias|nome alternativo|pseudonimo) (?:dell |del profilo |del mio )?account',
    'auto_v1_stock_remaining': r'(?:due )?(?:valori|numeri|quantità|livelli) (?:della scorta|delle scorte)',
    'auto_v1_cart_total': r'(?:due )?(?:quantità|numeri|valori) (?:del|nel) carrello',
    'auto_v1_inventory_total': r'(?:due )?(?:quantità|numeri|valori) (?:dell inventario|in inventario)',
    'auto_v1_unit_total': r'(?:prezzo|costo) (?:unitario|per unità) e (?:la )?quantità',
}
ARTICLE = r"(?:(?:il|lo|la|gli|le|i|l) )?"
POLITE = r'(?:per (?:favore|cortesia|piacere)|gentilmente|cortesemente)'
AUX = r'(?:(?:mi )?(?:puoi|potresti|potrebbe|riesci a)|ti (?:chiedo|prego) di|vorrei che tu)'
DISPLAY = r'(?:dimmi|dammi|mostrami|indicami|scrivimi|riportami|restituiscimi|forniscimi|mostra|fammi (?:vedere|leggere)|mi (?:dai|dici|mostri))'
VERBS = [
    ('mostra', DISPLAY + r'|(?:dirmi|darmi|mostrarmi|indicarmi|fornirmi|riportarmi|scrivermi|restituirmi)'),
    ('somma', r'(?:somm|addizion)(?:a|are|ami|armi)'),
    ('moltiplica', r'moltiplic(?:a|are|ami|armi)'),
    ('inverti', r'(?:invert(?:i|ire|imi|irmi)|rovesci(?:a|are)|capovolg(?:i|ere))'),
    ('converti', r'(?:convert(?:i|ire|imi|irmi)|trasform(?:a|are)|rend(?:i|ere))'),
    ('calcola', r'calcol(?:a|are|ami|armi)'),
    ('trascrivi', r'(?:trascriv(?:i|ere|imi|ermi)|copi(?:a|are|ami|armi))'),
]


def frame(text):
    text = re.sub('^' + POLITE + ' ', '', text)
    text = re.sub(' ' + POLITE + '$', '', text)
    text = re.sub('^' + AUX + ' ', '', text)
    for lemma, pattern in VERBS:
        text = re.sub(r'^(?:' + pattern + r')\b', lemma, text)
    return text


class FrameRouter:
    def __init__(self, families, examples):
        self.operations = {f['id']: f['operation'] for f in families}
        self.entities = {f['id']: re.compile(r'\b(?:' + re.escape(tokenize(f['object_phrase']))
                         + '|' + ARTICLE + '(?:' + NOUNS[f['id']] + r'))\b') for f in families}
        self.labels = defaultdict(set)
        for query, family in examples:
            extracted = self.parse(query)
            if extracted and extracted[0] == family:
                self.labels[extracted[1]].add(self.operations[family])

    def parse(self, query):
        text = tokenize(query)
        if not text:
            return None
        matches = [(family, match) for family, pattern in self.entities.items()
                   for match in pattern.finditer(text)]
        if len(matches) != 1:
            return None
        family, match = matches[0]
        return family, frame(text[:match.start()] + '@' + text[match.end():])

    def predict(self, query):
        parsed = self.parse(query)
        if parsed:
            family, instruction = parsed
            if self.labels.get(instruction) == {self.operations[family]}:
                return family
        return None
