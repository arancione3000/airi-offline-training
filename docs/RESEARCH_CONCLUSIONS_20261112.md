# Esito della ricerca — 12 novembre 2026

Il guasto del recupero è riprodotto e corretto. L'ipotesi che basti rendere addestrabile la proiezione FFN finale è invece respinta dalla prova controllata. Non abbiamo dimostrato un miglioramento della chat, né identificato una causa unica di tutti i limiti del modello.

## Che cosa non funzionava

Il job `42277407762` era fallito il 6 novembre per HTTP 500 scaricando l'indice del checkpoint. Il supervisore cercava i log attraverso `gh api`, che rifiutava gli escape ANSI prima ancora di arrivare al classificatore degli errori transitori. Il watcher intercettava l'errore generico, riprovava e si rinnovava: un supervisore attivo mascherava così un addestramento fermo.

La [correzione pubblicata nella PR6](https://github.com/arancione3000/airi-offline-training/pull/6) permette la cattura dei log, elimina escape e controlli prima della classificazione e limita gli errori consecutivi del controller. Conserva il massimo di due rilanci effettivi del job; limita anche le richieste con ricevuta incerta. I 17 test includono la forma reale ANSI/HTTP500, errori di autorizzazione e ricevute incerte. La CI remota del commit di integrazione è passata. Il 12 novembre l'indice dello stesso checkpoint risponde HTTP 200; le verifiche di hash, provenienza e ricetta restano obbligatorie al ripristino.

## Prova sull'architettura

Procedura fissata prima della prova in [RESEARCH_WORKFLOW_20261112.md](RESEARCH_WORKFLOW_20261112.md). Entrambi i bracci partono dai pesi approvati revisione4, con Adam nuovo e la medesima traiettoria TRAIN di 256 unità logiche, ricordi e sentinelle. A aggiorna attenzione finale/norme/testa; B aggiunge solo la proiezione di uscita dell'ultimo FFN. Nessuna etichetta di valutazione entra nell'ottimizzatore, nessuna scelta intermedia, nessun test finale sigillato eseguito.

| Misura | Modello approvato | A: configurazione attuale | B: aggiunta FFN finale |
| --- | ---: | ---: | ---: |
| Parametri addestrabili | 74.304 | 74.304 | 1.450.560 |
| Risposte corrette | 23/92 | 22/92 | 21/92 |
| Nuove risposte corrette | — | 0 | 0 |
| Risposte prima corrette perse | — | 1 | 2 |
| Contesto | 0/32 | 0/32 | 0/32 |
| Certificati TRAIN conservati | 897/897 | 897/897 | 896/897 |
| NLL italiana | 1,162535 | 1,140033 | 1,127350 |
| NLL inglese | 0,981683 | 0,962868 | 0,954508 |
| NLL conversazioni umane EN | 3,302020 | 3,289916 | 3,284963 |
| Aggiornamenti confermati / proposte fisiche | — | 252/262 | 243/262 |
| Token supervisionati, tentativi inclusi | — | 43.109 | 44.003 |
| Token risposta di riferimento | — | 29.459 | 29.666 |
| Secondi di training e guardie | — | 168,58 | 162,20 |

La riproduzione iniziale coincide con tutte le 92 generazioni e i 897 certificati del modello approvato. Entrambi gli export nativi coincidono con i rispettivi modelli addestrati; i file del checkpoint sorgente sono rimasti invariati. Entrambi i bracci raggiungono il limite inferiore del moltiplicatore LR, 1/64. Le norme medie dei gradienti prima del clipping sono 0,538 e 0,673, massime 1,988 e 2,956; il clipping interviene nel 16,79% e 26,34% delle proposte. Non sono misure separate dei gradienti di ogni termine dell'obiettivo. I conteggi dei token includono i tentativi e si sovrappongono; non vanno sommati come token unici del corpus. [Report, log, sorgenti originali e comandi di riproduzione](../scripts/research-training/ffn-output-20261112/README.md).

**Decisione: respinti entrambi i candidati.** La NLL scende, ma non compensa la perdita di risposte complete. Nessun peso sperimentale viene pubblicato come campione. La proiezione FFN resta un esperimento archiviato; i 31 moduli della ricetta in produzione mantengono i loro hash originali.

## Che cosa possiamo concludere, e cosa resta aperto

- L'interruzione di sei giorni ha una causa operativa dimostrata. Ripararla consente di riprendere gli aggiornamenti; non dimostra un aumento delle capacità.
- Il vecchio training modifica solo lo 0,148% di circa 50 milioni di parametri. Aumentarlo al 2,89% con questa specifica proiezione e questa ricetta non basta nella prova di 256 unità. Non segue che ogni FFN, durata, LR o corpus debba fallire.
- Le otto sentinelle campionate per aggiornamento non garantiscono tutti i 897 certificati: B ne perde uno al controllo completo. Il controllo globale prima dell'adozione serve proprio a respingere questi casi e resta attivo.
- La sola diminuzione della NLL ha già prodotto revisioni senza nuove risposte corrette. Va riportata separatamente dal numero di risposte acquisite e perse.
- Il corpus umano usato conta 866 conversazioni, appena 18 italiane. Questo limite e le sonde strette non consentono di certificare una chat generale; non dimostrano da soli un difetto del codice o la causa unica del problema.
- La prossima ricerca utile deve misurare l'apprendimento su esempi TRAIN inizialmente errati, contributi e conflitti dei gradienti e copertura dei dati, prima di ampliare ulteriormente lo spazio addestrabile. Aggiungere sottosistemi o ripetere più volte le stesse prove non è una soluzione dimostrata.

Il confronto usa ricordi/sentinelle fissati in anticipo anziché la priorità adattiva del servizio. Gli Adam partono nuovi per entrambi i bracci, perché B cambia lo spazio dei parametri. È un test dello spazio addestrabile, non una riproduzione esatta della continuazione live né un confronto a token o FLOP uguali. Una sola traiettoria non stabilisce significatività statistica. Il benchmark di 92 risposte e le NLL umane non costituiscono una valutazione conversazionale generale.

## Ripartenza

Dopo la conclusione negativa della prova, il controllo del servizio è stato riabilitato preservando il job133, il checkpoint sorgente, i contatori e la ricetta congelata. È stato richiesto un supervisore aggiornato sul ramo principale. Il servizio non ha una scadenza globale: procede per job limitati, salva i checkpoint e si arresta su errori d'integrità o recuperi esauriti. L'osservazione del vero recupero e del passaggio successivo viene registrata separatamente; la sola richiesta di avvio non certifica training attivo.
