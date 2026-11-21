# Collegare un oggetto all'azione richiesta

Il collo di bottiglia emerso nella prova precedente era misurabile. Airi riconosceva 20 oggetti nelle 36 richieste eseguibili formulate con nomi alternativi, ma risolveva 0 comandi completi. Questa prova conserva i sette operatori e le diciotto famiglie approvate al commit `7137c60b07edc1561db35dee7fbebc452fb17977`.

## Ipotesi e criterio di ammissione

Il candidato identifica l'oggetto e interpreta l'azione dal testo della domanda. Può proporre una procedura soltanto quando i due elementi concordano. La chat verifica separatamente i dati nel contesto ed esegue il programma approvato. Il candidato può usare il TRAIN e le due prove pubblicate il 20 novembre come materiale di sviluppo. Il confronto principale è con l'ultima versione del riconoscitore di oggetti, eseguita sugli stessi messaggi.

Un autore separato prepara 108 richieste italiane. Per ognuna delle 18 famiglie scrive tre comandi validi con modi diversi di nominare l'oggetto e tre richieste incompatibili. La prova è congelata con SHA-256 `aa808942ff23461fa4c58feba2398a117866cfa97d7ba00497dbeda773e0e63a`. Il codice di ciascun candidato viene fissato prima della sua esecuzione sulla prova nuova. Ogni domanda usa due valori indipendenti dal routing. Il punteggio conta una sola volta la formulazione, richiedendo che entrambe le risposte abbiano valore, operazione, identità della famiglia e provenienza corretti. Il fallback nativo conta come astensione.

La soglia minima è 27 risposte corrette su 54 comandi, nessuna attivazione sui 54 casi incompatibili, nessun errore sulle risposte emesse e conservazione dei 600 esempi memorizzati e dei 989 contratti storici. Per attivare l'interpretazione libera si applica anche il criterio stabilito dal progetto sulle prove precedenti (almeno metà di ciascuna, senza errori) e la verifica finale ancora riservata. Le osservazioni precedenti utilizzate per lo sviluppo misurano l'adattamento. Riportiamo la corrispondenza con domande già osservate e l'unità della misura è la formulazione, non la singola variante di valore.

## Confronti

- Confrontare la chat approvata, il riconoscitore precedente e il nuovo sistema oggetto-azione. Riportare anche quante delle differenze sono risposte guadagnate o perse sugli stessi comandi. Il solo riconoscimento degli oggetti noti si misura anche sui comandi incompatibili che nominano un oggetto realmente supportato.
- Confrontare il nuovo sistema con un piccolo Transformer aperto istruito a proporre sia un oggetto sia un'azione, con possibilità di astensione. Il modello opera in locale e riceve la stessa domanda e le definizioni dei compiti. Usa i medesimi programmi per eseguire le risposte. Riportare revisione, quantizzazione, costo delle decisioni e risultati, senza estendere il confronto alla qualità generale della conversazione.
- Misurare separatamente il riconoscimento dell'oggetto e dell'azione sulle richieste valide. Le etichette forniscono il riferimento soltanto durante il calcolo delle metriche. Un candidato che indovina l'oggetto ma interpreta male l'azione deve astenersi.

## Trasferimento delle formulazioni verificate

Il percorso per domande letterali può importare una previsione corretta su TRAIN come nuova formulazione di una procedura già approvata. Per questa fase si utilizzano esclusivamente i due cataloghi di proposte già etichettate, con 140 righe complessive. La previsione legge il testo senza conoscere famiglia sorgente o etichetta. Soltanto dopo la previsione si verifica che oggetto e operazione concordino con le etichette e con un programma installato. L'ammissione richiede esecuzioni su nuovi valori, fonti incompatibili e controlli di conservazione. Restano in vigore il limite del ciclo automatico e l'ordine di precedenza delle abilità. L'effetto su una frase verificata con nuovi valori è un risultato operativo distinto dal punteggio sulle frasi nuove.

## Conferma dopo la prima misura

Il candidato iniziale supera la soglia di copertura sulla prima prova, ma produce sei attivazioni indebite. In una singola iterazione successiva si può usare quella prova per sviluppare un filtro sulle richieste. Per la verifica si congela il filtro e si usa un secondo autore che vede soltanto le definizioni delle famiglie. La seconda prova comprende 36 richieste valide e 54 richieste incompatibili in 18 famiglie. Si confrontano sullo stesso insieme il primo candidato congelato e quello con il filtro. Restano valide le soglie precedenti su ogni insieme, inclusi almeno 18 comandi corretti sul secondo e nessuna attivazione indebita o regressione. La prima prova diventa materiale di sviluppo per il filtro. Il confronto con il modello esterno usa la prima prova e il protocollo fissato prima della valutazione.
