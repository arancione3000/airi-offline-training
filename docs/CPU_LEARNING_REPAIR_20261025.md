# AIRI: correzione del percorso di apprendimento su CPU, 25 ottobre 2026

Richiesta: ottenere miglioramenti concreti considerando l'hardware disponibile. Modello e insegnanti restano esclusivamente nativi AIRI, senza modelli preaddestrati esterni o API LLM.

## Stato verificato prima della modifica

Il ciclo `airi-offline-421180-39614174241-1-2` è terminato con 4.288 aggiornamenti tentati, 5.077,155933 secondi di ottimizzazione, 64 verifiche negative e zero adozioni. Il modello approvato è ancora `airi-native`, revisione 1. I 64 elementi della banca sono osservazioni di training, non 64 nuove capacità generali.

Sul checkpoint reale ho riprodotto la baseline originale: 15/28 risposte legacy, 4/16 dialoghi semplici, 4/16 fondamenti, totale 23/92 includendo i compiti contestuali. Comprensione del contesto ancora 0. La NLL dei dialoghi umani di sviluppo è 3,377343348 per l'inglese. Non esistono esempi italiani umani rimasti in quel sottoinsieme dopo i vecchi filtri: questa misura non dimostra miglioramento della conversazione italiana. Il test finale sigillato non è usato per selezionare le modifiche.

## Problemi corretti

- **Costo sproporzionato su CPU.** La rete ha 50.127.552 parametri, dimensione interna 96 e dodici enormi strati feed-forward da 14.336 unità. Ogni modifica completa richiedeva il calcolo dei gradienti della rete e di Adam. Ho separato la proiezione finale dall'embedding, inizialmente con pesi identici, congelando il corpo. Si ottengono 36.864 parametri addestrabili e 50.164.416 totali. Il formato nativo supporta già questa separazione.
- **Dati di acquisizione rumorosi.** Il vecchio insieme combinato conteneva 5.928 dialoghi OASST1 non revisionati contro 465 OASST2 revisionati rimasti. Nel primo esempio esaminato la risposta non corrispondeva alla domanda. Questo non significa che tutte le 5.928 righe siano errate. Il nuovo percorso acquisisce solo OASST2 revisionato: 866 esempi dopo i filtri, 848 inglesi e 18 italiani. Sono ammessi esempi revisionati già presenti nel vecchio TRAIN, mantenendo invariati i filtri di sviluppo/finale e la separazione dei testi. Il ripasso nativo rimane distinto.
- **Ripetizione eccessiva dei pochi dialoghi italiani.** La regola precedente assegnava due aggiornamenti su tre all'italiano anche per quei soli 18 dialoghi. L'acquisizione ora campiona le lingue proporzionalmente alla radice del numero di esempi: circa 12,7% italiano e 87,3% inglese nei soli slot umani. Il curriculum e il ripasso bilingue restano attivi.
- **Ripetizione di calcoli immutabili.** Una cache LRU limitata conserva soltanto le attivazioni del corpo congelato, mai risposte, etichette o predizioni. La proiezione corrente viene ricalcolata a ogni accesso. Limite studente 256 MiB e insegnanti 64 MiB ciascuno. Modificare il corpo invalida le attivazioni; tentare di addestrarlo con la cache attiva genera errore.
- **Perdita di competenze e controlli troppo ravvicinati.** Il ripasso supervisionato usa quattro ricordi certificati per lingua e aggiornamento. La proiezione opzionale del delta della testa protegge fino a 48 direzioni tratte esclusivamente da risposte di TRAIN già corrette. Non garantisce conservazione generale: restano obbligatori tutti i vecchi criteri e ogni precedente risposta corretta di sviluppo. I controlli avvengono dopo 1.024 nuovi aggiornamenti; il contatore relativo funziona anche dopo transazioni aggiuntive di consolidamento.

## Misure e verifiche

[Misura sul checkpoint reale](evidence/CPU_HEAD_BENCHMARK_20261025.json): stessa CPU, stesso piccolo lotto revisionato e stessa funzione obiettivo, due thread Torch. Otto aggiornamenti completi: 13,6631 s; otto aggiornamenti della testa includendo il riempimento iniziale delle cache: 0,5171 s, rapporto 26,42. La NLL di quel compito di TRAIN scende da 3,5832 a 3,3283. Esportazione nel normale motore nativo: logits e chat identici. Insegnante immutato. Il riferimento della misura locale è il campione corrente; il preflight remoto usa anche il riferimento ancestrale originale.

Questi tempi escludono caricamento, valutazioni e pubblicazione; la ripetizione di un singolo lotto non misura la velocità dell'intero training e non dimostra capacità di conversare. Nei confronti locali con 1.024 lotti distinti sono stati misurati circa 125–156 secondi di ottimizzazione iniziale e circa 28–47 secondi ripercorrendo lotti già in cache. Non sono stime garantite per GitHub Actions.

[Confronti locali](evidence/CPU_HEAD_LOCAL_TRIALS_20261025.json): le prime varianti migliorano la NLL umana ma perdono risposte precedenti e vengono respinte. Non sono state promosse. I confronti isolati usano il campione corrente anche come insegnante nativo; il ciclo reale conserva il suo riferimento e i suoi ancoraggi originali.

Verifiche locali completate: 60 test di trasferimento e 7 del controller; `check_resume.py`, `check_success_native.py`, `check_continual_native.py`, `check_fast_head.py`. Comprendono gradienti causali, nessuna adozione con regressioni, ripristino di pesi/Adam, continuazione esatta di Adam, memoria limitata, esportazione nativa, decodifica KV e protezione delle direzioni selezionate.

## Ciclo concreto

Ramo fissato: `native-cpu-head-repair-20261025`. Sorgente immutabile: `d078e718dfba0d68babd31de90fa55c0beac1951`. Ripartenza verificata: passo 421180, impronta sorgente `c3cfbecdd1b5c425446f95dd5a95755010907e42a1b386eed63820db49d008cb`.

Migrazione esplicita: stessi pesi iniziali e RNG, nuovi momenti Adam soltanto per la testa; tutte le riprese successive ripristinano quell'Adam. Banca e identità native conservate. Tentativi da 4.096 aggiornamenti, verifica ogni 1.024, profili 1e-4 / 3e-5 / 1e-5, pesi di conservazione 16/16. Limiti: 24 decisioni, 6 tentativi completi respinti, 21.600 secondi di ottimizzazione oppure 32 job. Il controller continua da checkpoint verificati e si arresta al limite; non richiede chiamate periodiche a ChatGPT.

Il nuovo ciclo è un esperimento di adattamento economico della testa, non una rete che inventa autonomamente un'architettura. Non può promettere miglioramento perpetuo. Un candidato entra nello stesso modello corrente solo superando i criteri originali; la qualificazione per uso generale resta distinta.

## Verifica remota

[Preflight 40139417255](https://github.com/arancione3000/airi-offline-training/actions/runs/40139417255): **successo**, commit `52b13ba6da8d70fd9aae3fb0bfe587bff5d01bf8`. Anche sul runner reale passano tutti i controlli nativi. Quattro CPU visibili, due thread Torch. Sullo stesso piccolo lotto: 8 aggiornamenti completi 10,2795 s, testa con cache inizialmente vuota 0,4733 s, rapporto **21,72**. La NLL del compito di TRAIN scende da 3,4835 a 3,3530. Esportazione nativa esatta e campione immutato. [JSON misurato](evidence/CPU_HEAD_REMOTE_PREFLIGHT_20261025.json). Restano i limiti di interpretazione descritti sopra: non è una misura end-to-end né una promozione di qualità.

[Training reale 40139709413](https://github.com/arancione3000/airi-offline-training/actions/runs/40139709413) avviato sul ramo congelato e sullo stesso commit. Controller riabilitato e verifica manuale [40139726630](https://github.com/arancione3000/airi-offline-training/actions/runs/40139726630): successo, 7 test e azione `active_job_retained`, senza duplicare il job di training. Prima telemetria reale: **1.024 aggiornamenti**, passo **422204**, **165,879 s di ottimizzazione**. Corpus effettivo 866 righe revisionate. Dei 512 aggiornamenti umani, 445 inglesi e 67 italiani; ripasso sempre 8 esempi e pesi 16/16 anche nei trasferimenti. Testa 36.864 parametri, rango protetto 48; cache 9.120 hit, 1.227 miss, 61.572.224 byte su 268.435.456, zero invalidazioni. Banca iniziale conservata: 64 lezioni, 92 posizioni e 7 guadagni su risposte intere di TRAIN. [Snapshot](evidence/CPU_HEAD_FIRST_1024_20261025.json).

La telemetria precede il primo controllo indipendente: revision1, zero adozioni a quel punto. Questi aggiornamenti dimostrano che il ciclo gira e modifica i pesi; non dimostrano ancora una conversazione migliore.

Il vecchio controller orario `airi-evolving-controller.yml`, fissato all’esperimento storico `evolving-context-v1`, è stato disabilitato per evitare ulteriori controlli inutili e mantenere un solo controller di apprendimento corrente. Il controller `airi-continual-controller.yml` rimane attivo.


### Esito del primo job completo e continuazione automatica

Il job 40139709413 è terminato con **successo**, salvando il checkpoint `airi-offline-422812-40139709413-1-2`: **1.632 aggiornamenti**, **217,743 s di ottimizzazione**. Indice, stato e archivio di ripresa sono stati scaricati e verificati con SHA256. [Dati del checkpoint](evidence/CPU_HEAD_FIRST_CHECKPOINT_20261025.json).

Al controllo dopo 1.024 aggiornamenti la NLL umana inglese passa da 3,377343 a 3,351440; le NLL native italiane/inglesi da 1,215603/1,050222 a 1,186867/1,034634. Tuttavia il candidato passa da 15 a 14 risposte legacy corrette; dialoghi 4/16 e fondamenti 4/16 invariati. Risultato complessivo candidato 22/92 contro 23/92 del campione. Anche le due interpolazioni vengono respinte. **Zero adozioni, campione approvato invariato alla revisione 1.** Il miglioramento delle loss non viene presentato come migliore conversazione.

Il tentativo da 4.096 aggiornamenti non è ancora completo: pesi candidati e Adam vengono conservati al confine del job. Il controller, attivato dall’evento `workflow_run` e senza una nuova chiamata a ChatGPT, ha avviato [la continuazione 40140414835](https://github.com/arancione3000/airi-offline-training/actions/runs/40140414835). [Controllo automatico 40140402847](https://github.com/arancione3000/airi-offline-training/actions/runs/40140402847): successo, ripresa dello stesso checkpoint verificato, continuazione 1. Al momento della verifica il nuovo job era nella fase di training dopo aver superato i controlli nativi.
