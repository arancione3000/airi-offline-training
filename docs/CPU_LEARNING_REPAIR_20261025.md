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

In attesa del preflight GitHub Actions sul codice pubblicato e dell'avvio verificato. I risultati remoti saranno aggiunti qui dopo l'esecuzione.
