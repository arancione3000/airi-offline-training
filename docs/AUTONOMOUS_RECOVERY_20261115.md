# Recupero autonomo del training — 15 novembre 2026

## Guasto riprodotto

Il run43452114543/job266 è fallito il13 novembre scaricando il checkpoint490876: `stream error: stream ID 5; PROTOCOL_ERROR; received from peer`. Il classificatore non riconosceva questo errore HTTP/2 e impostava un arresto permanente. Inoltre, le chiamate `gh` prive di timeout potevano esaurire il tempo del supervisore; tre errori temporanei del controller erano trattati come guasto permanente.

## Correzioni

- Timeout di60 secondi per ogni chiamata API. Errori di rete, HTTP429/5xx, timeout e log temporaneamente indisponibili mantengono il controller in recupero, con attese fino a300 secondi fra le prove.
- I job falliti per rete ripartono dopo attese durature di60,120,240 secondi, fino a3600. Dopo due rilanci effettivi o ricevute incerte esaurite viene creato un nuovo run dal medesimo checkpoint verificato. Nessun limite globale ai tentativi; limiti locali e backoff evitano richieste continue.
- Una ricevuta di dispatch incerta viene ricercata prima di ripetere la richiesta, con attesa crescente fino a3600 secondi. I nuovi run mantengono chiave, sorgente e indice logico; un watermark esclude ricevute di run precedenti. Il claim CAS già presente nel runner congelato mantiene un solo scrittore anche se arriva in ritardo un vecchio worker autorizzato.
- Cancellazioni e timeout dei job possono ripartire finché il servizio è abilitato e i log non mostrano errori di integrità o del programma. Per fermare il servizio si usa `enabled:false` nel controllo duraturo.
- Il checkpoint finale viene ricavato dalla ricevuta obbligatoria `SAVED_RELEASE` del tentativo corrente. Verifiche di run, tentativo, passo, hash e provenienza precedono il passaggio successivo. Rimossa la ricerca limitata alle prime300 release.
- Il supervisore si rinnova ogni20 minuti, lasciando margine al limite del runner. Un watchdog separato esegue una transizione e riattiva un supervisore assente su completamento/cancellazione e ogni15 minuti, anche se il training è ancora attivo. Usa un proprio gruppo concorrente per non restare in coda al watcher; il CAS sullo stato continua a impedire dispatch e scrittori concorrenti. Controlla i supervisori attivi/in coda prima di richiederne uno. Non si attiva sul proprio completamento e non chiama l'assistente. La callback del training usa il nome statico esplicito del workflow, senza dipendere soltanto da tale evento.

Errori di digest, provenienza, checkpoint, programma, autorizzazione o incompatibilità CLI continuano a produrre un arresto visibile. Nessun sistema può proseguire mentre GitHub o le credenziali restano indisponibili; i guasti temporanei non richiedono invece uno sblocco manuale. La periodicità dei cron dipende dalla disponibilità di GitHub Actions.

## Verifica e attivazione

30 test del controller e7 della ricetta continua passati localmente, inclusi errore HTTP/2 reale, più di tre guasti API, timeout, log404, rinnovo, cancellazione, POST incerta, rollover, scrittore unico e rifiuto dei checkpoint di tentativi precedenti. I31 moduli della ricetta di training, il fingerprint e il ramo congelato18c9dd5 restano invariati. Nessuna modifica ai pesi, ai dati o ai criteri di adozione.

CI remota, ripartenza del job266 e checkpoint successivo vengono registrati dopo l'attivazione. Questa modifica riguarda la continuità del servizio; non certifica nuove capacità del modello, che resta alla revisione4 approvata.
