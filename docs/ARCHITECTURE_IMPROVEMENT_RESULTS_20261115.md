# Miglioramento dell'architettura — risultati del15 novembre2026

## Componente utilizzabile

È disponibile un nuovo ingresso sperimentale: `python scripts/assistant-runtime/chat.py`.
Mantiene separati un componente software di memoria contestuale e il modello nativo
approvato, caricato soltanto quando serve il fallback. Non attiva pesi sperimentali
o provider non qualificati. [Uso e limiti](../scripts/assistant-runtime/README.md).

La memoria recupera dichiarazioni esplicite di nome, residenza e argomento, conserva
le correzioni e restituisce messaggio e intervallo di testo della fonte. Si astiene
su sintassi non supportata, conflitti non risolti, citazioni/ipotesi e ruoli non
compatibili. Non salva profili o conversazioni su disco e non condivide fatti fra
sessioni: lavora sulla cronologia fornita. `/reset` cancella la cronologia del processo.

## Misure distinte

| Misura | Solo modello nativo | Assistente con memoria |
| --- | ---: | ---: |
| Benchmark storico92 | 23/92 | 27/92 |
| Successi dei23 originali persi | — | 0 |
| Campione nuovo32, inferenza effettiva | 0/32 | 32/32 |

L'inferenza effettiva ha usato gli stessi pesi approvati revisione4, digest57bdb933…,
con caricamento verificato dal resolver originale. Sul benchmark storico i4 punti
aggiuntivi provengono dalla memoria software. **Non sono presentati come miglioramento
dei pesi neurali o prova di chat generale.** La generazione greedy non cambia l'RNG.

La verifica funzionale preregistrata V2 comprende96 casi positivi con valori nuovi:
96 corretti. Nei68 casi negativi nessuna risposta indebita. Include valori Unicode
e multiword, conflitti, ipotesi/citazioni, ruoli, domande nello stesso turno e cronologie
oltre la finestra dei token del modello. Questo misura una grammatica dichiarata,
non comprensione di tutte le formulazioni naturali.

Per i897 certificati TRAIN,20 richieste instradate alla memoria producono lo stesso
gold certificato; le877 astensioni inoltrano invariata la chiamata al modello immutato.
Nessuna perdita rilevata. È una verifica della composizione e dell'eredità del ramo
nativo, non una nuova certificazione neurale di tutti i897 esempi. Il benchmark92
è stato anche rieseguito con inferenza nativa reale, non solo riutilizzando output.

Sono passati50 test unitari/integrativi, compreso il vero comando chat, il reset,
le astensioni e l'obbligo del record hash per un checkpoint locale. I test del
parser includono150 valori casuali Unicode con correzioni e provenienza esatta.
Il controllo dei31 hash della ricetta esistente resta superato. Una nuova CI
esegue test e verifiche di preservazione a ogni modifica del componente.

## Prima versione e correzione verificata

V1 fa72/96 sui positivi (75%),0 risposte indebite: non supera la soglia80% fissata.
La mancanza riguarda le forme collettive dell'argomento in italiano/inglese, previste
dal contratto ma non implementate. È stata corretta genericamente la grammatica,
senza elenchi di valori o accesso dell'autore ai benchmark. La nuova valutazione V2
usa un seed di valori diverso, fissato prima della correzione. V1, relativo codice,
preregistrazione e risultato fallito sono conservati; non vengono retroattivamente
chiamati un successo. La validazione V2 riguarda nuovi valori nella grammatica
specificata, non domande semanticamente arbitrarie mai viste.

## Esperimento neurale più ampio

È stato eseguito un solo training effettivo su512 righe/256 valori, due lingue,
quattro lunghezze e coppie controfattuali, aggiornando tutti i50.164.416 parametri.
Dataset, LR3e-5, batch4, obiettivo nativo e split sono fissati prima degli aggiornamenti.
Il limite600s ha terminato il training a493 aggiornamenti,600,543s includendo l'ultimo
aggiornamento indivisibile:1.972 esposizioni,436 righe viste4 volte e76 viste3 volte.
Non si dichiara quindi completata la dose pianificata di512 aggiornamenti.

| Risultato del candidato neurale | Esito |
| --- | ---: |
| TRAIN con risposta completa+EOS | 391/512 |
| Finale: valori nuovi, template noto | 4/16 |
| Finale: template nuovo, valori noti | 3/16 |
| Finale: valori e template nuovi | 0/32 |
| Benchmark storico | 3/92 |
| Successi precedenti persi | 21/23 |

La sorgente fa0/64 sul nuovo finale. Il candidato mostra trasferimento parziale
con valori nuovi in forme note, ma fallisce nettamente i criteri predefiniti e
peggiora le vecchie capacità. **Candidato respinto e isolato.** Nessun parametro
o LR viene ritoccato dopo questi risultati, nessun peso viene attivato. I gate
costosi897/NLL non vengono eseguiti perché il trigger predefinito è già fallito;
non vengono dichiarati superati. Export nativo e immutabilità di sorgente/codice
sono verificati.

Un errore nella collazione batch4 è stato corretto prima di qualsiasi forward o
aggiornamento, lasciando invariati dati e disegno. Il tentativo a zero aggiornamenti
e i suoi hash sono archiviati. I risultati completi, i limiti dell'annotazione dei
bucket e le istruzioni di riproduzione sono nel
[report del curriculum](../scripts/research-training/compositional-20261115/RESULTS.md).

## Stato finale e limiti

Il [piano](ARCHITECTURE_IMPROVEMENT_PLAN_20261115.md) è stato eseguito: base stabile,
memoria, composizione, training più ampio, valutazioni e gate di attivazione. È
attivato il nuovo ingresso di chat con memoria, mentre l'esperto non qualificato
rimane escluso. Il servizio storico di training e il suo puntatore nativo non
vengono sostituiti. Il runtime/provider principale mantiene i propri gate: questa
integrazione sperimentale non ne aggira la qualificazione e non è un'installazione
remota sul PC dell'utente.

La memoria rende affidabili i casi espliciti supportati; il modello di base può
ancora produrre risposte scadenti negli altri casi. Non è stata risolta la chat
generale, né garantito che altri giorni della vecchia ricetta producano nuovi
apprendimenti. Le prove indicano che un curriculum più ampio può insegnare parte
del compito, ma ancora con troppo poco trasferimento e troppa dimenticanza.

[Manifest della composizione](../scripts/assistant-runtime/evidence/MANIFEST.json) ·
[Riepilogo delle misure](../scripts/assistant-runtime/evidence/SUMMARY.json) ·
[Manifest del candidato](../scripts/research-training/compositional-20261115/evidence/MANIFEST.json).
Sono inclusi dati sintetici, report, codice e hash; nessun peso, stato Adam,
credenziale o log di conversazioni private.
