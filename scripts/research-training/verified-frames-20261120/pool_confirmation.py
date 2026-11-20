"""Independent confirmation proposals authored without seeing ranking results.

Use the same schema as pool.POOL. Each literal object occurs once. A supported
request has the operation of its object family as its gold label. Unsupported
or underspecified requests have a null label. Text was written from the object
inventory and ordinary Italian usage. TRAIN and the first pool were checked
only for overlaps after this corpus was written.
"""

import hashlib


_REQUESTS = (
    ('parcel_copy', 'copy', 'Mi lasci qui il codice del pacco così come l’ho dato?'),
    ('auto_v1_email_alias', 'copy', "Ridammi per iscritto l'alias email così com'è."),
    ('auto_v1_order_reference', 'copy', "Mi occorre leggere esattamente il riferimento dell'ordine."),
    ('auto_v1_customer_reference', 'copy', 'Voglio che tu trascriva il riferimento cliente senza cambiarne la grafia.'),
    ('auto_v1_contact', None, "Non ho chiesto chi sia il referente dell'ordine, ma a che ora risponda."),
    ('auto_v1_email_alias', None, "Come correggeresti l'alias email se fosse incompleto?"),
    ('auto_v1_order_reference', None, "Mostrami le prime due lettere che compongono il riferimento dell'ordine."),
    ('parcel_copy', None, 'Riporta il codice del pacco soltanto se è diverso da quello della ricevuta.'),
    ('signal_reverse', 'reverse', 'Ribalta la sequenza che vedi per il segnale acquisito, dall’ultimo carattere al primo.'),
    ('auto_v1_tracking_reverse', 'reverse', 'Riscrivi al contrario il token di verifica partendo dall’ultimo simbolo.'),
    ('signal_reverse', 'reverse', 'Mi daresti il segnale acquisito con tutti i caratteri letti dalla fine?'),
    ('auto_v1_tracking_reverse', 'reverse', 'Scrivi il token di verifica cominciando dall’ultimo carattere e finendo con il primo.'),
    ('signal_reverse', None, 'Ordina in ordine alfabetico i caratteri che formano il segnale acquisito.'),
    ('auto_v1_tracking_reverse', None, 'Inverti il token di verifica solo se il risultato resta uguale.'),
    ('signal_reverse', None, 'Che differenza c’è tra invertire il segnale acquisito e invertire soltanto le sue cifre?'),
    ('auto_v1_tracking_reverse', None, 'Ripeti il token di verifica senza leggere i caratteri all’indietro.'),
    ('card_upper', 'uppercase', 'Mi restituisci in maiuscolo il testo della tessera, lettera per lettera?'),
    ('auto_v1_label_upper', 'uppercase', "Passa al maiuscolo ogni lettera che trovi nell'input per l'etichetta di spedizione."),
    ('card_upper', 'uppercase', 'Mi serve il testo della tessera con tutte le lettere trasformate in maiuscolo.'),
    ('auto_v1_label_upper', 'uppercase', "È possibile avere l'etichetta di spedizione nella versione con ogni lettera in maiuscolo?"),
    ('card_upper', None, 'Cambia in maiuscolo solo le vocali nel valore indicato come il testo della tessera.'),
    ('auto_v1_label_upper', None, "Fammi vedere l'etichetta di spedizione e dopo confronta la versione maiuscola con l'originale."),
    ('card_upper', None, 'Mi confermi se il testo della tessera si trova già tutto in maiuscolo?'),
    ('auto_v1_label_upper', None, "Per esercitarti, inventa l'etichetta di spedizione e poi scrivila in maiuscolo."),
    ('plate_lower', 'lowercase', 'Dammi il testo della targa nella versione con tutte le lettere abbassate in minuscolo.'),
    ('auto_v1_account_lower', 'lowercase', "Per l'alias account, converti ogni carattere alfabetico nella rispettiva minuscola."),
    ('plate_lower', 'lowercase', 'Mi riscrivi il testo della targa facendo diventare minuscole anche le eventuali maiuscole?'),
    ('auto_v1_account_lower', 'lowercase', "Metti in minuscolo tutte le lettere che trovi quando leggi l'alias account."),
    ('plate_lower', None, 'Prendi il testo della targa e cancella le maiuscole invece di convertirle.'),
    ('auto_v1_account_lower', None, "Mi servono sia l'alias account in minuscolo sia il conteggio dei caratteri."),
    ('plate_lower', None, 'Non so se il testo della targa vada scritto in minuscolo. Cosa suggerisci?'),
    ('auto_v1_account_lower', None, "Leggi l'alias account e porta in minuscolo soltanto l’ultima lettera."),
    ('delta_sum', 'add', 'Usa l’addizione per combinare i due valori delta in un unico risultato.'),
    ('auto_v1_cart_total', 'add', 'Vorrei sapere che totale ottieni se sommi le quantità del carrello.'),
    ('auto_v1_inventory_total', 'add', "Per le quantità dell'inventario, dimmi il numero ottenuto facendo l’addizione."),
    ('delta_sum', 'add', 'Mi restituisci il totale dopo aver aggiunto tra loro i due valori delta?'),
    ('auto_v1_inventory_total', None, "Prima somma le quantità dell'inventario, poi dividi il totale per due."),
    ('auto_v1_cart_total', None, 'Aggiungi cinque a ciascun numero che trovi per le quantità del carrello.'),
    ('delta_sum', None, 'Se una persona dicesse «somma i due valori delta», sarebbe un ordine completo?'),
    ('auto_v1_cart_total', None, 'Mostrami come potresti controllare se le quantità del carrello sono numeri interi.'),
    ('sigma_product', 'multiply', 'Usa i due interi sigma come fattori della stessa moltiplicazione.'),
    ('auto_v1_unit_total', 'multiply', 'Fai venire fuori il prodotto moltiplicando tra loro il prezzo unitario e la quantità.'),
    ('sigma_product', 'multiply', 'Mi dici che valore assume il prodotto quando moltiplichi i due interi sigma?'),
    ('auto_v1_unit_total', 'multiply', 'Calcola il prodotto usando come fattori il prezzo unitario e la quantità.'),
    ('sigma_product', None, 'Scarta il più basso tra i due interi sigma, poi raddoppia l’altro.'),
    ('auto_v1_unit_total', None, 'Moltiplica il prezzo unitario e la quantità per cento, separatamente.'),
    ('sigma_product', None, 'Se i due interi sigma fossero entrambi negativi, quale sarebbe il segno del prodotto?'),
    ('auto_v1_unit_total', None, 'Questa volta sottrai tra loro il prezzo unitario e la quantità, nell’ordine dato.'),
    ('auto_v1_remaining', 'subtract', 'Mi serve il primo numero meno il secondo quando usi i due valori del saldo.'),
    ('auto_v1_stock_remaining', 'subtract', 'Considera i due valori della scorta e dal primo numero rimuovi il secondo.'),
    ('auto_v1_remaining', 'subtract', 'Quanto ottieni se sottrai al primo il secondo usando i due valori del saldo?'),
    ('auto_v1_stock_remaining', 'subtract', 'Per i due valori della scorta riporta la differenza, con il secondo sottratto al primo.'),
    ('auto_v1_remaining', None, 'Sottrai l’uno dall’altro i due valori del saldo nell’ordine che preferisci.'),
    ('auto_v1_stock_remaining', None, 'Vorrei che tu elencassi i due valori della scorta in ordine decrescente.'),
    ('auto_v1_remaining', None, 'Se i due valori del saldo mancassero, potresti stimare la differenza?'),
    ('auto_v1_stock_remaining', None, 'Per i due valori della scorta, calcola quanto rimane del secondo dopo avere tolto il primo.'),
)


def _row(family, intent, query):
    key = hashlib.sha256(f'{family}\0{query}'.encode()).hexdigest()[:16]
    return {'id': key, 'query': query, 'family': family, 'intent': intent}


POOL = tuple(sorted((_row(*request) for request in _REQUESTS), key=lambda row: row['id']))
