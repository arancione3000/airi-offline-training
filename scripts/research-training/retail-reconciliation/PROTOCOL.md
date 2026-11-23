# Prova di riconciliazione di quantità

Il compito consiste nel calcolare le unità spedite al netto degli storni leggendole da un breve resoconto. Una risposta automatica è corretta solo se dà il saldo intero per una richiesta di saldo con una fonte utente univoca. Dati mancanti, due resoconti in disaccordo, fonti fornite solo dall’assistente e richieste di altri calcoli richiedono revisione umana.

I valori provengono da vendite e cancellazioni aggregate per codice prodotto nel dataset [Online Retail](https://doi.org/10.24432/C5BW33). I messaggi in italiano, le variazioni di formato e i casi problematici sono simulati. La prova misura un passaggio di un flusso plausibile di riconciliazione. I valori e i formati non provengono dai documenti di un cliente.

Due formati di fonte arrivano in sequenza. Per ciascuno, quattro esempi etichettati insegnano il compito e dodici casi diversi sono usati per l’ammissione. Il test finale usa 36 coppie disgiunte per formato, una per caso. Per formato include 14 richieste esatte, 6 parafrasi, 4 variazioni nel resoconto e 12 richieste da rimandare alla revisione. Le coppie di tutti gli insiemi sono disgiunte per codice e per valori. I casi conflittuali attingono da coppie riservate al proprio caso.

- Airi usa il banco esistente di procedure, con il verificatore storico. Un fallback vuoto segnala revisione e non simula il comportamento generativo dell’APK. Il banco viene ricaricato da disco a ogni passaggio.
- Lo script Python usa regole dichiarate dai due formati, sottrazione esatta e verifica di unicità delle fonti. Accetta spazi e ordine dei campi variabili. Le regole linguistiche di richiesta sono definite prima di eseguire il confronto.
- Qwen2.5-1.5B-Instruct Q4_K_M riceve la definizione del compito e i medesimi otto esempi TRAIN nel prompt. L’uscita è vincolata alla coppia di operandi o a REVISIONE. Per la coppia si usa la stessa sottrazione esatta. Il modello non riceve le etichette di ammissione o del test.

Si riportano separatamente risposte corrette, mancate automazioni su richieste valide, saldi sbagliati e numeri emessi dove serve revisione. Si misura il tempo di inferenza per metodo sulla stessa macchina. Si osserva la conservazione delle risposte del primo formato dopo il secondo, comprese le astensioni, e si segnala qualsiasi regressione. I tempi di processo non stimano le ore dell’operatore né il prezzo di un servizio.

Corpus, regole manuali e criteri sono progettati dallo stesso autore. Si registrano hash e si esegue il test una volta congelato il corpus. Nessuna modifica di un metodo dopo aver visto gli esiti entra nella stessa prova.

SHA-256 del corpus congelato
`0a0747e2e818e1cb7d9aff3b53032b33140f66851169deec404299fff170786d`.
