# Diagnosi dell'apprendimento — protocollo del15 novembre2026

## Domanda e validità del confronto

L'utente autorizza verifica ed esecuzione dei cinque punti: percorso training/chat, apprendibilità, isolamento dei vincoli e dello spazio aggiornabile, intervento motivato dai risultati, verifica su esempi nuovi e conoscenza conservata. Il servizio remoto può continuare; tutte le prove sono isolate dal modello approvato e non cambiano i rami congelati.

Il piano è sensato come diagnosi limitata, non come garanzia di miglioramento. Confrontare configurazione protetta e non protetta isola il pacchetto di conservazione nel suo insieme: loss, normalizzazione, sentinelle, rollback e LR effettivo. Non distingue da solo quale componente interferisca. Il confronto fra spazio ristretto e modello completo vale per gli esempi, il LR e il budget scelti, non per ogni possibile ottimizzazione.

## Controlli e prova predefinita

1. Controllare gli stessi messaggi attraverso SFT e chat nativa: serializzazione, shift causale, maschere, fine risposta/EOS, causalità, gradienti, cache e generazione con/senza KV. Verificare export nativo e immutabilità del checkpoint sorgente. I due controlli nativi esistenti su prefix e continual-step sono già passati; l'audit specifico delle nuove righe seguirà.
2. Partire dalla revisione4 approvata, non da un candidato respinto. Fissare otto esempi TRAIN bilingui inizialmente sbagliati, con valori nuovi e coppie che differiscono nel contesto ma mantengono la stessa domanda. Registrare dati e ordine prima degli aggiornamenti. Non utilizzare etichette development/final per l'ottimizzatore.
3. Tre bracci indipendenti dallo stesso checkpoint e AdamW nuovo: A attenzione finale/norme/testa con conservazione; B medesimo spazio aggiornabile senza il pacchetto di conservazione; C tutti i parametri nativi aggiornabili senza conservazione. LR3e-5, clipping1 e identico obiettivo acquisizione nativo, inclusi EOS1.2 e repetition-unlikelihood0.05. Nessuna cache di prefisso congelato nel braccio completo.
4. Massimo64 unità logiche per braccio; limiti temporali del solo training A240s, B60s, C600s. Stessa traiettoria predefinita fino al limite raggiunto. Riportare unità confermate/rifiutate, proposte fisiche, esposizioni, token e costo: non è un confronto a FLOP o tempo uguali. Un braccio troncato non prova incapacità di apprendere.
5. Misure TRAIN iniziali/intermedie/finali: loss, token corretti, margini, gradienti, risposte complete e EOS. Generazione nativa effettiva all'inizio/fine; le certificazioni teacher-forced non sostituiscono la verifica numerica della generazione con cache.
6. Fissare varianti indipendenti per distinguere valori visti con template nuovo, valori nuovi con template visto e valori/template entrambi nuovi. Valutare a fine prova, includendo il checkpoint sorgente come baseline; riportare risposte guadagnate e perse. Non adattare i dati/LR dopo averle osservate.
7. Misurare le92 risposte originali e le23 precedentemente corrette. Eseguire l'intero ledger897 se un candidato raggiunge almeno6/8 TRAIN completi oppure acquisisce una risposta held-out nuova. Anche un incremento TRAIN inferiore a6/8 resta evidenza positiva limitata, non viene cancellato dalla soglia di controllo costoso.

## Decisioni condizionali

- A fallisce e B impara: analizzare il pacchetto di conservazione, con contatori di rifiuto/LR e gradienti; non indebolire automaticamente i controlli globali.
- Solo C impara: lo spazio congelato limita questo caso sotto il budget scelto. Un intervento successivo può ampliare lo spazio aggiornabile, ma deve conservare le conoscenze prima di essere adottato.
- TRAIN migliora ma varianti no: la prova dimostra memorizzazione, non una nuova abilità generale. L'intervento deve riguardare copertura/composizione dei dati e trasferimento, non soltanto più passi.
- Tutti falliscono: esito inconcludente fra budget, LR, difficoltà, rappresentazione e implementazione. Non autorizza a dichiarare con certezza un bug del modello.

Applicare al massimo un intervento motivato dalla diagnosi, poi valutarlo in isolamento. Non adottare candidati che perdono vecchie risposte o certificati, né chiamare progresso una sola diminuzione della NLL. Nessun checkpoint sperimentale diventa il modello approvato senza i controlli originali. I risultati e l'eventuale intervento saranno registrati separatamente dopo osservazione; questo documento non anticipa un esito.
