"""Independent Italian request holdout for evaluating one frozen candidate."""
import gzip
import hashlib
import json
from pathlib import Path
import sys
import unicodedata

ARCHIVE = Path(__file__).resolve().parents[1] / "semantic-routing-20261116/evidence/precommitted"
VALUE_OFFSET = 45000

# Each pair contains at least one literal object mention. The second may use
# inflection, clitic attachment, contraction or a different clause structure.
POSITIVES = {
    "parcel_copy": ("Mi puoi dare il codice del pacco?", "Fammi leggere il codice del pacco."),
    "signal_reverse": ("Rovescia il segnale acquisito.", "Per favore leggi al contrario il segnale acquisito."),
    "card_upper": ("Riscrivimi in maiuscolo il testo della tessera.", "Voglio il testo della tessera in stampatello maiuscolo."),
    "plate_lower": ("Riscrivimi in minuscolo il testo della targa.", "Puoi usare tutte minuscole per il testo della targa?"),
    "delta_sum": ("Per favore somma i due valori delta.", "Mi dai il totale dei due valori delta?"),
    "sigma_product": ("Quanto vale il prodotto dei due interi sigma?", "Per favore moltiplica i due interi sigma."),
    "auto_v1_remaining": ("Dei due valori del saldo, quanto fa il primo meno il secondo?", "Vorrei i due valori del saldo sottratti nell'ordine in cui te li ho dati, primo meno secondo."),
    "auto_v1_contact": ("Mi dici chi è il referente dell'ordine?", "Ho bisogno del nome del referente dell'ordine."),
    "auto_v1_customer_reference": ("Dammi il riferimento cliente per favore.", "Mi puoi ripetere il riferimento cliente?"),
    "auto_v1_order_reference": ("Mi dai il riferimento dell'ordine?", "Trascrivimi il riferimento dell'ordine."),
    "auto_v1_email_alias": ("Fammi vedere l'alias email.", "Ti ricordi l'alias email? Me lo puoi ripetere?"),
    "auto_v1_tracking_reverse": ("Scrivimi al rovescio il token di verifica.", "Inverti i caratteri del token di verifica."),
    "auto_v1_label_upper": ("Puoi mettere in maiuscolo l'etichetta di spedizione?", "Riscrivimi in lettere maiuscole l'etichetta di spedizione."),
    "auto_v1_account_lower": ("Puoi mettere in minuscolo l'alias account?", "Riscrivimi l'alias account usando tutte lettere minuscole."),
    "auto_v1_stock_remaining": ("Prendi i due valori della scorta e sottrai il secondo dal primo.", "Quanto fa la differenza fra i due valori della scorta, prendendo il primo meno il secondo?"),
    "auto_v1_cart_total": ("Calcolami il totale delle quantità del carrello.", "Per favore somma le quantità del carrello."),
    "auto_v1_inventory_total": ("Quanto fa la somma delle quantità dell'inventario?", "Per favore somma le quantità dell'inventario."),
    "auto_v1_unit_total": ("Quanto fa il prodotto tra il prezzo unitario e la quantità?", "Per favore moltiplica il prezzo unitario e la quantità."),
}

# The expected output for these tasks is outside the family's contract.
OTHER = {
    "copy": "Indicami solo la lunghezza in caratteri di {obj}.",
    "reverse": "Ordina alfabeticamente i caratteri di {obj}.",
    "uppercase": "Cambia in maiuscolo soltanto la prima lettera: {obj}.",
    "lowercase": "Metti in minuscolo una lettera sì e una no: {obj}.",
    "add": "Invece di sommare, dimmi qual è il maggiore: {obj}.",
    "subtract": "Usa {obj} e dammi il secondo meno il primo.",
    "multiply": "Arrotonda al centinaio più vicino il prodotto che ottieni usando {obj}.",
}


def load(name):
    return json.loads(gzip.decompress((ARCHIVE / (name + ".json.gz")).read_bytes()))


def key(query):
    return " ".join(unicodedata.normalize("NFC", query).casefold().strip(" .!?").split())


def dataset():
    families = load("FAMILIES")["families"]
    if {f["id"] for f in families} != set(POSITIVES):
        raise ValueError("original family identities required")
    rows = []
    for family in families:
        target, obj = family["id"], family["object_phrase"]
        first, second = POSITIVES[target]
        statements = (
            ("positive_1", first, target),
            ("positive_2", second, target),
            ("embedded", f"Che cosa intende dire una persona che domanda «{first}»?", None),
            ("negated", f"Non eseguire la richiesta seguente: {first}", None),
            ("extra_condition", first.rstrip(".?") + " e sostituisci ogni cifra con un asterisco.", None),
            ("operation", OTHER[family["operation"]].format(obj=obj), None),
        )
        for category, query, label in statements:
            rows.append(dict(id=f"verified-frames-{target}-{category}", query=query, lesson_id=label,
                             source_lesson_id=target, condition=category))
    questions = {key(row["query"]) for row in rows}
    if len(questions) != len(rows) or questions & {key(row["query"]) for row in load("TRAIN")["rows"]}:
        raise ValueError("duplicate within holdout or with TRAIN")
    return dict(rows=rows, value_offset=VALUE_OFFSET)


if __name__ == "__main__":
    raw = (json.dumps(dataset(), sort_keys=True, indent=2, ensure_ascii=False) + "\n").encode()
    if sys.argv[1:] == ["--sha256"]:
        print(hashlib.sha256(raw).hexdigest())
    elif not sys.argv[1:]:
        sys.stdout.buffer.write(raw)
    else:
        raise SystemExit("usage: holdout.py [--sha256]")
