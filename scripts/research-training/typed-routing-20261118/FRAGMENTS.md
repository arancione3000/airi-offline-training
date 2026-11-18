# Trasferimento selettivo dal candidato V5

Il compilatore conserva una formulazione esatta per ognuna delle 12 procedure con un generatore di valori indipendente. Scorre il solo TRAIN in ordine, esclude la domanda canonica e le etichette ambigue e interroga il donatore congelato. Accetta la prima formulazione per cui il donatore sceglie la procedura indicata dall'etichetta indipendente. Il manifest conserva domanda, identificatore TRAIN e impronte del candidato e dei dati. Per riprodurre il manifest dalla radice con l'ambiente encoder del protocollo V5:

```sh
python scripts/research-training/typed-routing-20261118/extract_fragments.py /tmp/fragments.json
cmp /tmp/fragments.json scripts/assistant-runtime/skills/v5-fragments.json
```

Il normale ciclo modulare sintetizza il programma dagli esempi etichettati dall'oracolo, verifica nuovi valori e astensioni e ricontrolla gli esperti precedenti. Un rifiuto in ammissione lascia spazio al prossimo candidato indipendente. Un errore nel controllo complessivo finale impedisce la pubblicazione del gruppo. Le esecuzioni pianificate continuano a verificare il catalogo e possono acquisire aggiunte approvate. L'estrazione dal donatore V5 è una fase offline riproducibile.

Partendo dal banco a 18 esperti del commit 0b93a90, tre cicli con limite 4 e seed 2026111801, 2026111802 e 2026111803 portano a 30 esperti su 32 posti. Il seed del terzo ciclo dà 504/504 casi di audit, inclusi 144 valori nuovi e 120 richieste di astensione per le 12 nuove formulazioni. Superati anche 600 esempi memorizzati e i controlli su 989 richieste storiche.

Con il checkpoint nativo rev4, decodifica greedy fino a 192 token e la prima prova positiva per ciascuna nuova formulazione al seed 9958117, la chat composta passa da 0/12 a 12/12. Le risposte integrali sono in [fragment-chat.json](evidence/fragment-chat.json). La domanda è nota, il valore da elaborare cambia. Il risultato misura copertura procedurale. La comprensione di formulazioni mai viste, i pesi neurali e il confronto con altri Transformer richiedono prove distinte.
