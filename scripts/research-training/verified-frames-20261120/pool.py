"""Manually authored Italian proposals for a budgeted verification experiment.

Each proposal names an object from the precommitted FAMILIES inventory and has
one gold label. A string expresses the supported operation on that object.
None indicates a request the single-operation expert cannot fulfil faithfully.
Rows contain wording only. Source values are supplied independently by the
evaluator. Reviewers and donors receive the same wording and family, with the
gold label withheld until a proposal is checked.
"""

import hashlib


# One example per construction. The author used the family inventory and
# ordinary Italian usage without consulting qualification, final or challenge
# data. Position in _REQUESTS is editorial. Candidate ordering is by opaque ID.
_REQUESTS = (
    ('parcel_copy', 'copy', 'Mi detti il codice del pacco così come compare?'),
    ('auto_v1_contact', 'copy', "Ti va di indicarmi il referente dell'ordine?"),
    ('auto_v1_customer_reference', 'copy', 'Quale valore è riportato come il riferimento cliente?'),
    ('auto_v1_order_reference', 'copy', "Posso avere il riferimento dell'ordine riportato sopra?"),
    ('auto_v1_email_alias', 'copy', "Vorrei ritrovare l'alias email che avevo indicato."),
    ('parcel_copy', 'copy', 'Trascrivi fedelmente il codice del pacco.'),
    ('parcel_copy', None, 'Se inventassi il codice del pacco, come lo costruiresti?'),
    ('auto_v1_contact', None, "Chi potrebbe sostituire il referente dell'ordine?"),
    ('auto_v1_customer_reference', None, 'Scrivi il riferimento cliente e aggiungi un numero a caso.'),
    ('auto_v1_order_reference', None, "Spiega la domanda «come ritrovo il riferimento dell'ordine?»."),
    ('auto_v1_email_alias', None, "Scrivi l'alias email con tutte le lettere maiuscole."),
    ('parcel_copy', None, 'Dimmi se il codice del pacco e il codice della ricevuta coincidono.'),
    ('signal_reverse', 'reverse', 'Riscrivi il segnale acquisito carattere per carattere partendo dalla fine.'),
    ('auto_v1_tracking_reverse', 'reverse', 'Puoi invertire la sequenza di caratteri per il token di verifica?'),
    ('signal_reverse', 'reverse', 'Mostrami il segnale acquisito leggendolo a ritroso.'),
    ('auto_v1_tracking_reverse', 'reverse', 'Vorrei leggere il token di verifica cominciando dal suo ultimo carattere.'),
    ('signal_reverse', 'reverse', 'Capovolgi l’ordine dei caratteri che formano il segnale acquisito.'),
    ('auto_v1_tracking_reverse', 'reverse', 'Riporta il token di verifica in ordine inverso, dalla fine all’inizio.'),
    ('signal_reverse', None, 'Come si fa di solito a invertire il segnale acquisito? Spiega il metodo.'),
    ('auto_v1_tracking_reverse', None, 'Riporta il token di verifica nell’ordine originale.'),
    ('signal_reverse', None, 'Inverti il segnale acquisito e scarta il primo carattere del risultato.'),
    ('auto_v1_tracking_reverse', None, 'Se il token di verifica fosse un numero, sarebbe più grande del suo inverso?'),
    ('signal_reverse', None, 'Riesci a invertire soltanto le cifre che compongono il segnale acquisito?'),
    ('auto_v1_tracking_reverse', None, 'Non invertire il token di verifica.'),
    ('card_upper', 'uppercase', 'Scrivi il testo della tessera tutto con le lettere grandi.'),
    ('auto_v1_label_upper', 'uppercase', "Puoi rimandarmi in maiuscolo l'etichetta di spedizione?"),
    ('card_upper', 'uppercase', 'Rendi maiuscole tutte le lettere che formano il testo della tessera.'),
    ('auto_v1_label_upper', 'uppercase', "Vorrei l'etichetta di spedizione con la grafia tutta maiuscola."),
    ('card_upper', 'uppercase', 'Riporta in caratteri maiuscoli il testo della tessera.'),
    ('auto_v1_label_upper', 'uppercase', "Riscrivi l'etichetta di spedizione mettendo ogni lettera in maiuscolo."),
    ('card_upper', None, 'Quante lettere maiuscole contiene il testo della tessera?'),
    ('auto_v1_label_upper', None, "L'invito «metti in maiuscolo l'etichetta di spedizione» ti sembra cortese?"),
    ('card_upper', None, 'Lascia il testo della tessera esattamente com’è, anche nelle minuscole.'),
    ('auto_v1_label_upper', None, "Puoi mettere in maiuscolo solo la prima lettera di ogni parola per l'etichetta di spedizione?"),
    ('card_upper', None, 'Converti il testo della tessera in maiuscolo e poi inverti i caratteri.'),
    ('auto_v1_label_upper', None, "Questa volta usa le lettere minuscole per l'etichetta di spedizione."),
    ('plate_lower', 'lowercase', 'Fai vedere il testo della targa scritto per intero in minuscolo.'),
    ('auto_v1_account_lower', 'lowercase', "Vorrei trasformare tutte le lettere in minuscole per l'alias account."),
    ('plate_lower', 'lowercase', 'Mostrami in caratteri minuscoli il testo della targa.'),
    ('auto_v1_account_lower', 'lowercase', "Riporta l'alias account mettendo in minuscolo ogni singola lettera."),
    ('plate_lower', 'lowercase', 'Puoi far comparire il testo della targa senza alcuna maiuscola?'),
    ('auto_v1_account_lower', 'lowercase', "Riesci a rendere interamente minuscolo l'alias account?"),
    ('plate_lower', None, 'Controlla il testo della targa e indica le lettere già minuscole.'),
    ('auto_v1_account_lower', None, "Restituisci l'alias account in minuscolo soltanto se contiene una cifra."),
    ('plate_lower', None, 'Scrivi il testo della targa a lettere alterne maiuscole e minuscole.'),
    ('auto_v1_account_lower', None, "Per favore, non riscrivere in minuscolo l'alias account."),
    ('plate_lower', None, 'Metti il testo della targa tutto in maiuscolo.'),
    ('auto_v1_account_lower', None, "Puoi inventare l'alias account usando solo lettere minuscole?"),
    ('delta_sum', 'add', 'Somma insieme i due valori delta e dammi il totale.'),
    ('auto_v1_cart_total', 'add', 'Quanto fanno insieme le quantità del carrello, in totale?'),
    ('auto_v1_inventory_total', 'add', "Vorrei il totale che si ottiene sommando le quantità dell'inventario."),
    ('delta_sum', 'add', 'Per i due valori delta, voglio il risultato dell’addizione.'),
    ('auto_v1_cart_total', 'add', 'Metti insieme le quantità del carrello in una sola somma.'),
    ('auto_v1_inventory_total', 'add', "Puoi calcolare la somma per le quantità dell'inventario?"),
    ('delta_sum', None, 'Calcola la media aritmetica per i due valori delta.'),
    ('auto_v1_cart_total', None, 'Somma le quantità del carrello solo se sono uguali.'),
    ('auto_v1_inventory_total', None, "Moltiplica tra loro le quantità dell'inventario."),
    ('delta_sum', None, 'Puoi dirmi come si sommerebbero i due valori delta, senza farlo?'),
    ('auto_v1_cart_total', None, 'Dimmi qual è la più piccola tra le quantità del carrello.'),
    ('auto_v1_inventory_total', None, "Quando ti chiederò «somma le quantità dell'inventario», rispondi quanto fa."),
    ('sigma_product', 'multiply', 'Fai la moltiplicazione per i due interi sigma.'),
    ('auto_v1_unit_total', 'multiply', 'Quanto viene moltiplicando il prezzo unitario e la quantità?'),
    ('sigma_product', 'multiply', 'Vorrei conoscere il prodotto che si ricava usando i due interi sigma.'),
    ('auto_v1_unit_total', 'multiply', 'Tratta il prezzo unitario e la quantità come fattori e calcola il prodotto.'),
    ('sigma_product', 'multiply', 'Moltiplica tra loro i due interi sigma e riporta il risultato.'),
    ('auto_v1_unit_total', 'multiply', 'Dammi il risultato ottenuto facendo la moltiplicazione tra il prezzo unitario e la quantità.'),
    ('sigma_product', None, 'C’è un motivo per cui dovrei moltiplicare i due interi sigma?'),
    ('auto_v1_unit_total', None, 'Addiziona il prezzo unitario e la quantità per favore.'),
    ('sigma_product', None, 'Non fare il prodotto per i due interi sigma, dammi un esempio generale.'),
    ('auto_v1_unit_total', None, 'Prendi il prezzo unitario e la quantità e stimami anche l’imposta.'),
    ('sigma_product', None, 'Moltiplica i due interi sigma per un terzo fattore a tua scelta.'),
    ('auto_v1_unit_total', None, 'Se conoscessi il prezzo unitario e la quantità, quale operazione useresti?'),
    ('auto_v1_remaining', 'subtract', 'Per i due valori del saldo, calcola il primo diminuito del secondo.'),
    ('auto_v1_stock_remaining', 'subtract', 'Cosa rimane togliendo il secondo numero dal primo per i due valori della scorta?'),
    ('auto_v1_remaining', 'subtract', 'Vorrei la sottrazione tra i due valori del saldo nell’ordine dato, primo meno secondo.'),
    ('auto_v1_stock_remaining', 'subtract', 'Prendi i due valori della scorta e dimmi di quanto il primo supera il secondo, anche se il risultato è negativo.'),
    ('auto_v1_remaining', 'subtract', 'Mi dai la differenza con segno per i due valori del saldo, partendo dal primo?'),
    ('auto_v1_stock_remaining', 'subtract', 'Togli dal primo un ammontare pari al secondo e usa i due valori della scorta.'),
    ('auto_v1_remaining', None, 'Togli il primo numero dal secondo per i due valori del saldo.'),
    ('auto_v1_stock_remaining', None, 'Qual è lo scarto senza segno per i due valori della scorta?'),
    ('auto_v1_remaining', None, 'Puoi scegliere tu in quale ordine sottrarre i due valori del saldo?'),
    ('auto_v1_stock_remaining', None, 'Per i due valori della scorta, ti chiedo il primo meno il secondo solo se il primo è maggiore.'),
    ('auto_v1_remaining', None, 'Fammi sapere sia la somma sia la differenza per i due valori del saldo.'),
    ('auto_v1_stock_remaining', None, 'Spiegami perché nel registro sono scritti i due valori della scorta.'),
)


def _row(family, intent, query):
    key = hashlib.sha256(f'{family}\0{query}'.encode()).hexdigest()[:16]
    return {'id': key, 'query': query, 'family': family, 'intent': intent}


POOL = tuple(sorted((_row(*request) for request in _REQUESTS), key=lambda row: row['id']))
