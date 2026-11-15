# Risultati della diagnosi dell'apprendimento — 15 novembre2026

## Esito

La rete sa imparare gli otto esempi nuovi quando tutti i parametri nativi sono aggiornabili. Con lo spazio ristretto della strategia corrente, gli stessi64 passi non bastano a impararne nemmeno uno. Tuttavia l'aggiornamento completo senza protezioni dimentica molte risposte; reintroducendo le protezioni apprende solo1/8 e continua a perdere conoscenza. **Nessun candidato supera i requisiti di adozione.** Il modello approvato resta revisione4,23/92 e contesto0/32.

Questi risultati individuano limiti misurati della strategia, non una causa unica e definitiva di tutti i problemi della chat. Il [protocollo](LEARNING_DIAGNOSIS_WORKFLOW_20261115.md) è stato verificato e seguito: controlli del percorso, tre confronti diagnostici, una modifica condizionale motivata, verifica su dati nuovi e conoscenza precedente.

## 1. Percorso training → chat

Baseline e quattro endpoint superano gli audit indipendenti: identici prompt e token SFT/chat, maschere e shift corretti, EOS incluso, influenza dei token futuri nulla, raw token generati uguali con/senza KV cache, testo della chat uguale alla decodifica, RNG invariato dalla generazione e checkpoint immutati. Export/reload conservano esattamente logits e generazioni. I controlli nativi preesistenti su prefix e continual-step passano.

Non è stato trovato un difetto di tokenizzazione, cache, salvataggio o aggiornamento che spieghi questi risultati. È un'affermazione sui percorsi e casi verificati, non una certificazione dell'intera applicazione.

## 2–3. Apprendibilità e confronto controllato

Otto righe TRAIN bilingui, inizialmente tutte sbagliate; coppie con valori diversi ma domanda identica. Identico checkpoint approvato, ordine, AdamW nuovo, LR3e-5, clipping1 e obiettivo acquisizione nativo (EOS1.2, repetition-unlikelihood0.05). Ogni braccio completa64 unità logiche: nessun limite temporale tronca il confronto. Nel braccio protetto alcuni tentativi vengono annullati; riportiamo quindi anche gli aggiornamenti effettivi.

| Braccio | Parametri aggiornabili | TRAIN nuovi corretti/EOS | Vecchio benchmark | Dei23 prima corretti persi | Certificati TRAIN conservati | Adam confermati / proposte fisiche |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Sorgente approvata | — | 0/8 | 23/92 | — | 897/897 | — |
| A: spazio attuale + conservazione | 74.304 | 0/8 | 22/92 | 1 | 897/897 | 63/68 |
| B: spazio attuale, sola acquisizione | 74.304 | 0/8 | 21/92 | 2 | 896/897 | 64/64 |
| C: intero modello, sola acquisizione | 50.164.416 | 8/8 | 11/92 | 15 | 567/897 | 64/64 |
| D: intero modello + conservazione | 50.164.416 | 1/8 | 19/92 | 4 | 896/897 | 37/70 |

C acquisisce anche3 risposte prima sbagliate del vecchio benchmark, ma ne perde15: il saldo non può essere presentato come miglioramento. Tutti i punteggi del benchmark coincidono anche usando corrispondenza raw, quindi il risultato non dipende dalla normalizzazione del testo.

TRAIN CE iniziale6,2122; finale A5,5443/B5,3756/C0,02943/D3,47978. Tempo dei soli aggiornamenti/guardie: A63,17s/B1,35s/C23,91s/D251,98s, totale340,41s. Preparazione, valutazioni indipendenti, controlli dei897 certificati ed export sono aggiuntivi. Non è un confronto a FLOP uguali o una stima della velocità del servizio reale.

## 4. Modifica motivata e risultato

Il successo di C giustifica provare lo spazio completo conservando il pacchetto di A: è l'unica modifica condizionale D, non una nuova ricerca di LR o di checkpoint favorevoli. Stessa traiettoria TRAIN/replay/sentinelle/ancore di A, stessi64 passi e limite600s, rispettato. Le callback protettive e le transazioni restano originali; nessun nuovo modello esterno.

D conferma un compromesso concreto:33 proposte fisiche respinte,37 aggiornamenti confermati, LR effettivo finale1/64 del valore iniziale. Alcuni controlli osservati segnalano full-sequence KL; le statistiche non consentono di attribuire ogni rifiuto allo stesso vincolo. Lo spazio completo protetto impara una risposta, ma perde4 risposte indipendenti e un certificato: la modifica è respinta.

La misura iniziale dei gradienti esclude una spiegazione troppo semplice: acquisizione e gradiente combinato hanno coseno0,996973 nello spazio ristretto e0,999744 nel completo. Non erano opposti in quel punto. Questo non dimostra che restino allineati per tutto l'allenamento; i coefficienti della loss non equivalgono alla quota misurata di apprendimento. Guardie, LR e interferenza funzionale vanno distinti.

## 5. Dati nuovi e criteri di adozione

C ottiene2/12 sulle varianti predefinite, entrambe con valori già addestrati e formulazione nuova. Ottiene0/8 nei due gruppi con valori nuovi. A/B fanno0/12. D fa1/12 sul gruppo ormai familiare, ma **0/24** sulla conferma fresca, composta da12 coppie controfattuali e tre condizioni (valore visto/template nuovo, valore nuovo/template visto, entrambi nuovi). La baseline fa0/12 e0/24 rispettivamente. Non abbiamo dimostrato copia generale di valori nuovi o chat migliore.

Il controllo completo897 è stato eseguito su sorgente e tutti i candidati; dimostra anche perché le sole sentinelle campionate non bastano. A conserva tutti i certificati TRAIN ma perde una risposta indipendente. Per D i gate NLL authored/human non sono stati eseguiti, come previsto dalla regola di costo: risposte precedenti e ledger erano già falliti. Nessun forward del test finale sigillato. Non viene quindi dichiarato superato l'intero insieme dei gate, né pubblicato un campione nuovo.

## Che cosa cambia nella diagnosi

- Il sistema corrente aggiorna lo0,148% dei parametri: questo spazio è una restrizione importante nel confronto effettuato. Non abbiamo localizzato quali singoli blocchi, FFN o embedding siano necessari, né provato impossibilità con durate/LR diversi.
- L'aggiornamento completo risolve la piccola prova di memorizzazione, ma da solo peggiora fortemente la conservazione. Il pacchetto protettivo riduce questa perdita senza eliminarla e rallenta l'acquisizione. Non basta lasciare lo stesso ciclo acceso o togliere le protezioni.
- Le varianti mostrano un limite di trasferimento. La copertura dei dati e l'apprendimento di regole riutilizzabili restano da risolvere; non è dimostrato che aggiungere dati qualsiasi sia sufficiente.
- L'inventario contiene16.241 righe authored (15.460 sequenze uniche,8.130IT/8.111EN) e918 conversazioni umane raw. Il pool umano runtime osservato è866,18IT/848EN: non equivale a soli18 esempi italiani complessivi. I3.031.318 token supervisionati tentati nei metadati includono ripetizioni; manca un ledger dei token unici di pretraining, quindi non ne deduciamo una soglia certa di adeguatezza.

La priorità dell'intervento D è stata verificare se la maggiore plasticità potesse convivere con la conservazione prima di ampliare i dati. Non abbiamo così dichiarato risolto il ramo della generalizzazione. Questo limite è esplicito e la sperimentazione si ferma qui, senza ulteriori tentativi scelti dopo i risultati.

## Artefatti e stato del servizio

[Audit del percorso nativo](../scripts/research-training/learnability-20261115/NATIVE_PATH_AUDIT.md), [manifest e risultati A/B/C](../scripts/research-training/learnability-20261115/evidence/FIRST_THREE_ARMS_MANIFEST.json), [riproduzione](../scripts/research-training/learnability-20261115/evidence/FIRST_THREE_ARMS_REPRODUCTION.md). Gli artefatti di D sono separati nella stessa cartella evidence. I report compressi conservano generazioni raw, EOS, aggiornamenti, controlli897, provenienza e hash. Nessun peso sperimentale, stato Adam o credenziale viene incluso nel repository.

Alle17:21UTC il puntatore approvato è ancora revisione4, digest57bdb933…; il controllo remoto è abilitato, senza halt, indice272. Questa ricerca non modifica servizio, ricetta congelata, dati in produzione o criteri di adozione. I nuovi script restano strumenti di laboratorio. Il mantenimento del vecchio modello è una conseguenza dei controlli falliti, non una dichiarazione che la sua qualità sia stata risolta.
