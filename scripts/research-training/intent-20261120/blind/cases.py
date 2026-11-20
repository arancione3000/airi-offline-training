"""Independent, fixed Italian routing test. Emit rows for query_learning.evaluate."""
import gzip
import hashlib
import json
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
ARCHIVE = ROOT / "scripts/research-training/semantic-routing-20261116/evidence/precommitted"
VALUE_OFFSET = 36000

# One request per family explicitly uses the original entity's name. A second
# uses everyday wording for the same task. Labels are authored independently.
REQUESTS = {
    "parcel_copy": ("Potresti copiarmi il codice del pacco?", "Mi ridai il codice che identifica il pacco?"),
    "signal_reverse": ("Scrivi il segnale acquisito al contrario.", "Puoi leggermi il segnale partendo dalla fine?"),
    "card_upper": ("Metti il testo della tessera in maiuscolo.", "Sulla tessera c'è una scritta. Me la riscrivi tutta in lettere maiuscole?"),
    "plate_lower": ("Metti il testo della targa in minuscolo.", "Come si legge la scritta sulla targa usando solo lettere minuscole?"),
    "delta_sum": ("Quanto vale la somma dei due valori delta?", "Se aggiungo un delta all'altro, a quanto arrivo?"),
    "sigma_product": ("Moltiplica tra loro i due interi sigma.", "Quanto viene il primo sigma per il secondo?"),
    "auto_v1_remaining": ("Calcola la differenza tra i due valori del saldo, il primo meno il secondo.", "Del saldo iniziale, quanto avanza togliendo il secondo importo?"),
    "auto_v1_contact": ("Mi indichi il referente dell'ordine?", "Come si chiama la persona di riferimento per questo ordine?"),
    "auto_v1_customer_reference": ("Puoi leggermi il riferimento cliente?", "Che riferimento è stato indicato per il cliente?"),
    "auto_v1_order_reference": ("Puoi leggermi il riferimento dell'ordine?", "Mi ripeti il riferimento che accompagna questo ordine?"),
    "auto_v1_email_alias": ("Puoi copiarmi l'alias email?", "Quale alias è stato scritto per la posta elettronica?"),
    "auto_v1_tracking_reverse": ("Puoi riscrivere il token di verifica al contrario?", "Dammi i caratteri del token di verifica dall'ultimo al primo."),
    "auto_v1_label_upper": ("Scrivi l'etichetta di spedizione in maiuscolo.", "Mi serve la dicitura di spedizione tutta in lettere maiuscole."),
    "auto_v1_account_lower": ("Scrivi l'alias account in minuscolo.", "Puoi riscrivermi il nome alternativo dell'account usando solo minuscole?"),
    "auto_v1_stock_remaining": ("Per i due valori della scorta, togli il secondo dal primo.", "Prendi la scorta iniziale e dimmi quanto avanza dopo il prelievo indicato dal secondo numero."),
    "auto_v1_cart_total": ("Mi calcoli la somma delle quantità del carrello?", "Quanti articoli ho in tutto mettendo insieme le due quantità nel carrello?"),
    "auto_v1_inventory_total": ("Mi calcoli la somma delle quantità dell'inventario?", "A quanto ammonta il conteggio complessivo dei due lotti in inventario?"),
    "auto_v1_unit_total": ("Moltiplica fra loro il prezzo unitario e la quantità.", "Quanto spendo se ogni pezzo costa la cifra indicata e ne compro il numero indicato?"),
}

WRONG = {
    "copy": "Conta i caratteri che compongono {obj}.",
    "reverse": "Copiami {obj} esattamente com'è.",
    "uppercase": "Trasforma in minuscolo {obj}.",
    "lowercase": "Trasforma in maiuscolo {obj}.",
    "add": "Moltiplica tra loro {obj}.",
    "multiply": "Somma tra loro {obj}.",
    "subtract": "Calcola la media tra {obj}.",
}


def load(name):
    return json.loads(gzip.decompress((ARCHIVE / (name + ".json.gz")).read_bytes()))


def normalize(text):
    return " ".join(unicodedata.normalize("NFC", text).casefold().strip(" .!?").split())


def dataset():
    families = load("FAMILIES")["families"]
    if {f["id"] for f in families} != set(REQUESTS):
        raise ValueError("blind test requires its original set of approved families")
    rows = []
    for family in families:
        target = family["id"]
        obj = family["object_phrase"]
        first, second = REQUESTS[target]
        # These ask for different actions despite containing the relevant entity
        # and, in most cases, a complete supported request.
        phrases = [
            ("explicit", first, target),
            ("conversational", second, target),
            ("reported_request", f"Il testo dell'esercizio dice «{first}». In che lingua è scritto?", None),
            ("conflicting_outputs", first.rstrip(".?") + ", poi dammi anche un secondo risultato diverso dal primo.", None),
            ("negated", f"Evita di rispondere alla domanda «{second}».", None),
            ("wrong_operation", WRONG[family["operation"]].format(obj=obj), None),
        ]
        for condition, text, label in phrases:
            rows.append({"id": f"blind-{target}-{condition}", "source_lesson_id": target,
                         "condition": condition, "lesson_id": label, "query": text})
    questions = {normalize(r["query"]) for r in rows}
    train = {normalize(r["query"]) for r in load("TRAIN")["rows"]}
    if len(questions) != len(rows) or questions & train:
        raise ValueError("duplicate phrase within blind test or overlap with TRAIN")
    return {"schema": 1, "value_offset": VALUE_OFFSET, "rows": rows}


def encoded():
    return (json.dumps(dataset(), ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode()


if __name__ == "__main__":
    data = encoded()
    if sys.argv[1:] == ["--sha256"]:
        print(hashlib.sha256(data).hexdigest())
    elif not sys.argv[1:]:
        sys.stdout.buffer.write(data)
    else:
        raise SystemExit("usage: cases.py [--sha256]")
