# Piano di miglioramento nativo di Airi

## Punto di partenza

La revisione nativa approvata è la 4. Il checkpoint di lavoro al passo 597.564 registra 39 epoche consecutive senza guadagni. Le prove precedenti con tutti i pesi hanno acquisito esempi di addestramento e perso molte risposte già corrette. La ricetta recente ha respinto 63 selezioni su 64 per regressioni sulle risposte storiche.

## Esperimento

Addestrare una correzione a rango 16 della proiezione in uscita sul checkpoint approvato. Il Transformer produce le rappresentazioni congelate. La correzione apprende da esempi TRAIN bilanciati tra contesto, calcolo, copia e dialogo italiano e inglese. Ogni aggiornamento combina perdita supervisionata, replay di esempi certificati e distillazione dei punteggi del modello approvato. Dopo l'addestramento una proiezione analitica limita la correzione per conservare i margini dei token corretti dei certificati TRAIN, inclusa la fine della risposta. Le soglie e le dosi sono fisse per questo esperimento.

Confrontare la revisione 4 e il candidato con generazione greedy sulle 92 richieste storiche e sullo sviluppo per famiglia. Verificare tutti i certificati TRAIN e misurare la perdita su dialoghi umani IT/EN fuori dal TRAIN. Per superare la prova servono almeno due risposte nuove corrette sullo sviluppo, tutte le risposte prima corrette, tutti i certificati e nessuna crescita della perdita umana superiore al 2% per lingua. Aprire il finale originale solo dopo il superamento di questi controlli. Conservare il candidato separatamente e lasciare la promozione al protocollo del canale nativo.

## Cicli automatici

Rivedere la composizione dei controlli per le formulazioni esatte e per il componente semantico. Ogni ciclo deve verificare le risposte approvate anche quando cambia la precedenza dei componenti. I cicli pratici possono pubblicare capacità verificate su dati attendibili. Le famiglie rimaste senza guadagni richiedono nuovi esempi e nuovi esperimenti. Il confronto con Transformer moderni richiede misure comparabili su compiti indipendenti e resta l'obiettivo di ricerca.

## Risultato e decisione

Il candidato ha completato 240 aggiornamenti. Il vincolo ha ridotto la correzione al 13,9% e il controllo sui margini dei token, incluso EOS, ha confermato 897 certificati TRAIN su 897. Sullo sviluppo i risultati sono scesi da 34/195 a 33/195 con corrispondenza testuale dopo la normalizzazione di maiuscole e spazi. Sul benchmark storico sono scesi da 23/92 a 22/92. La stessa richiesta è presente in entrambe le serie. Contesto, calcolo, copia e richiamo restano al punteggio iniziale. La perdita media sui dialoghi umani è scesa da 3,614 a 3,549 in italiano e da 3,373 a 3,331 in inglese. Il sottoinsieme di sviluppo umano contiene due esempi italiani e 42 inglesi.

Il candidato è respinto per mancato guadagno nelle risposte complete e perdita di una risposta già corretta. Il finale resta chiuso. La revisione nativa approvata è la 4. Conservare tutti i certificati TRAIN ha limitato la dimenticanza misurata, mentre il controllo di generazione resta necessario per le risposte fuori dal TRAIN. [Rapporto e riproduzione](../scripts/research-training/native-projection-20261120/README.md)

Il ciclo modulare ha acquisito autonomamente altre otto formulazioni, arrivando a 24 sulle 150 disponibili. La verifica congiunta dei componenti è attiva e controlla anche la conservazione delle risposte già qualificate. [Ciclo con i controlli aggiornati](https://github.com/arancione3000/airi-offline-training/actions/runs/44743965432)
