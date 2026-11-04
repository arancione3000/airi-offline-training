# AIRI: diversità dei valori e apprendimento del contesto, 4 novembre 2026

## Esito

I nuovi confronti locali **non dimostrano nuove capacità**. La revisione 2
approvata resta a **23/92 risposte esatte e 0/32 risposte contestuali**.
I candidati addestrati su valori più diversi riducono le loss di sviluppo,
ma non imparano una nuova risposta esatta e perdono risposte già corrette.
Tutti vengono respinti dai gate originali; nessun modello è adottato e
nessun nuovo ciclo operativo è avviato.

Il [ciclo remoto precedente](CONTEXT_LEARNING_REPAIR_20261102.md) è concluso.
Il ramo congelato `native-context-learning-20261102` resta al commit
`d3f5039c35953625b059263d0be08f70adc74a01`. Le nuove opzioni descritte qui
sono disponibili nel codice, **non sono attivate** e mantengono il
comportamento precedente quando la configurazione è assente.

Il 4 novembre sono stati anche scaricati e verificati i byte degli archivi
MODEL e OPTIMIZER terminali. Il digest dei pesi caricati coincide con il
puntatore della revisione 2. Questa verifica successiva estende quella del
3 novembre, che a quella data aveva controllato soltanto i descrittori degli
archivi; la [registrazione della verifica](evidence/CONTEXT_DIVERSITY_OBSERVED_CHECKS_20261104.json)
ne distingue esplicitamente la provenienza.

Le misure esatte, i conteggi ricalcolati dalle generazioni e i limiti sono
nel [riepilogo JSON](evidence/CONTEXT_DIVERSITY_SUMMARY_20261104.json).
La [provenienza SHA256](evidence/CONTEXT_DIVERSITY_PROVENANCE_20261104.json)
identifica gli output integrali e gli script diagnostici compressi.

## Ipotesi e modifica disponibile

Il TRAIN contestuale precedente usa otto valori con iniziali A–H. Questo
permette di memorizzare una piccola scelta di risposte senza dimostrare la
copia di un valore nuovo dalla conversazione. È un limite del curriculum
da misurare; non prova da solo la causa del benchmark fermo.

Il nuovo generatore deterministico costruisce 2.048 righe TRAIN in 1.024
coppie controfattuali: ogni coppia mantiene lo stesso esercizio e cambia
il valore letterale da restituire. Il pool del confronto principale contiene
1.868 valori distinti, iniziali A–Z, cifre e valori di più parole, bilanciati
per lingua, famiglia e lunghezza. Le risposte derivano dal valore inserito,
senza etichette generate dal modello. Target e cronologie protetti vengono
esclusi prima della costruzione.

La modalità opzionale `context_acquisition` usa `version=1`,
`template_policy="matched_train"` e `paired_batch=true`. Trapianta soltanto
i valori nelle formulazioni e cronologie del TRAIN originale; i due esempi
della coppia contribuiscono a un singolo aggiornamento Adam. Questo confronto
non misura l'effetto delle nuove formulazioni presenti nel pool grezzo.
Gli esempi di conservazione e i ripassi originali restano distinti. Il digest
del curriculum comprende dati e ricetta; lo script viene identificato e
archiviato per SHA256.

Il diagnostico storico `learning_probe.py --spec` ora rifiuta esplicitamente
una configurazione `context_acquisition`: quel percorso non implementa le
coppie e ignorarla avrebbe etichettato in modo errato il risultato come
misura della nuova ricetta. I confronti qui riportati usano gli script
dedicati archiviati e non cambiano con questa protezione.

L'audit delle separazioni copre 559 righe protette, 489 cronologie, 383 valori
riservati, tutti i 256 schemi originali e 6.656 righe abbinate degli stadi 0–2.
Il controllo indipendente ricostruisce lo stesso pool e verifica i digest
degli otto file del checkpoint sorgente. Sono controlli sui dati e
sull'integrità, senza significato di successo del modello.

## Confronto controllato: 512 aggiornamenti per braccio

Entrambi i bracci partono dalla revisione 2 verificata, con seed 20261103,
LR 3e-5, conservazione/KL 8, ripasso nativo 0,25 e due memorie certificate
per lingua. Si allenano 74.304 parametri dell'ultima attenzione,
normalizzazioni e testa, con 11 blocchi di prefisso congelati. Calendario,
esempi non contestuali, memorie e riferimenti ancestrali sono condivisi e
identificati da hash. La baseline riproduce esattamente tutte le 92 risposte
e i relativi flag archiviati.

| Misura di sviluppo | Revisione 2 | Valori originali in coppia | Valori diversi in coppia |
|---|---:|---:|---:|
| Risposte esatte | 23/92 | 21/92 | 21/92 |
| Nuove risposte corrette | — | 0 | 0 |
| Risposte precedenti perse | — | 2 | 2 |
| Contesto esatto | 0/32 | 0/32 | 0/32 |
| NLL nativa italiana | 1,206915 | 1,180068 | 1,097845 |
| NLL nativa inglese | 1,037827 | 1,008012 | 0,930186 |
| NLL umana inglese | 3,330514 | 3,269671 | 3,269168 |
| Gate originale | — | Respinto | Respinto |

Entrambi perdono gli stessi indici 0 e 64: un saluto italiano e una risposta
empatica inglese. Le NLL del braccio diversificato migliorano rispetto alla
revisione 2 del 9,04% in italiano e del 10,37% in inglese; quella umana
dell'1,84%. Il miglioramento della loss non si traduce in una nuova risposta
intera corretta. Le risposte contestuali cambiano in 15/32 casi, contro 4/32
nel controllo, ma restano tutte errate.

Ogni braccio vede 256 esempi contestuali: il controllo usa otto valori e
1.083 token target inclusi gli EOS, il braccio diverso usa 229 valori e
1.960 token target. Sono uguali gli aggiornamenti e il numero di esempi,
**non i token o i FLOP**. La CPU condivisa e il riscaldamento delle cache
impediscono un confronto attendibile di velocità.

Due interpolazioni già previste del candidato diverso verso la revisione 2,
con fattori 0,5 e 0,25, restano entrambe a 21/92, zero risposte nuove,
zero contesto e gli stessi due errori introdotti. Anche i passi più piccoli
sono respinti. I checkpoint sorgente e candidato rimangono identici dopo
queste misure; non vengono valutati i set sigillati o finali.

## Dose maggiore di solo contesto: esito negativo a 512

Un secondo diagnostico usa soltanto acquisizione contestuale, conservando
quattro memorie certificate e il riferimento nativo a ogni aggiornamento.
Parte dalla stessa revisione 2 e dallo stesso ambito addestrabile, con
seed 20261104. Nei 512 aggiornamenti vede 1.024 esempi contestuali,
737 valori distinti e 7.489 token target: quattro volte gli esempi del
braccio misto precedente. Il pool di questo seed contiene 1.866 valori.

Il risultato è **18/92**, zero risposte nuove e **0/32 nel contesto**.
Perde gli indici 0, 4, 49, 60 e 64. Le NLL native scendono a 0,996024
in italiano e 0,869082 in inglese; la NLL umana inglese è 3,290774.
Il gate di adozione fallisce. I 346,59 secondi misurati sono solo tempo
di ottimizzazione, senza confronto di throughput.

Il cambio di seed e la dose di contesto diversa impediscono di trattare
questo risultato come un'ablazione causale del confronto precedente.
L'export nativo ha logits esatti sul controllo registrato; sorgente e
insegnanti congelati restano immutati. Lo stato esatto di pesi, Adam e RNG
è conservato localmente, ma la continuazione opzionale a 2.048 aggiornamenti
**non viene attivata**. Il diagnostico si ferma a 512, senza adozione.

## Cosa indicano le sonde, e cosa resta irrisolto

Sui 32 prompt contestuali originali, la probabilità media del primo token
corretto passa dall'1,19% al 2,21% e il rango medio da 21,97 a 13,41
nel braccio diverso. Il rango migliora in 25 casi e resta uguale in sette,
ma il primo token corretto non è mai la prima scelta: **0/32**. I target di
questo piccolo set sono sempre `Lidia` e `Nereo`; una maggiore copertura
delle iniziali L/N può spiegare parte dello spostamento. Questi numeri non
provano il recupero del valore giusto dalla cronologia.

La prima sonda di rappresentazione esamina 64 esempi TRAIN con gli otto
nomi originali. Un classificatore ausiliario ridge, con una formulazione
TRAIN esclusa a turno dal fit, recupera l'identità del nome al 100% dalla
coda della domanda nei blocchi 10 e 11, in entrambe le lingue, sia nel
modello originale sia nella revisione 2. È un readout addestrato a parte:
non è un adattatore d'inferenza e non certifica la generazione dell'intera
risposta o la copia generalizzata. Le probabilità registrate in questa
sonda riguardano soltanto il primo token.

Una seconda sonda usa 512 righe TRAIN `context-name`, 485 valori e tutte
le 26 iniziali. Il fit e il test del classificatore hanno valori
globalmente disgiunti, anche tra lingue, e tutte le classi sono presenti
in ogni split. Il readout del primo carattere dalla posizione sorgente è
corretto al 100% a tutti gli strati misurati: embedding e blocchi
0, 4, 8, 10 e 11. Alla coda della domanda del blocco 11 raggiunge
30/61 in italiano (49,18%) e 42/64 in inglese (65,62%), contro predittori
di maggioranza misurati del 3,28% e 3,125%.

Questo dimostra informazione decodificabile da una sonda, con quei dati e
quel readout. Non esclude un limite nella copia dell'intero valore,
nell'accesso causale all'informazione o nella decodifica della risposta.
Fit e test condividono famiglie di formulazioni TRAIN; non sono stati
eseguiti interventi causali o controlli con etichette mescolate. Entrambe
le sonde lasciano il modello immutato e non usano forward dei set di
valutazione. I problemi misurati restano la generalizzazione della copia
completa e la conservazione delle risposte già acquisite.

## Verifiche e riproducibilità dell'evidenza

I log archiviati della revisione iniziale registrano 70 test della suite pura
(uno saltato in assenza di Torch), nove test nativi dell'acquisizione e sei
di integrazione del prefisso. Dopo la protezione del diagnostico storico,
passano 71 test della suite pura (uno saltato) e dieci test nativi
dell'acquisizione, oltre alla compilazione dei cinque file e al controllo
whitespace. Questi ultimi esiti sono registrati come
[osservazioni dei tool](evidence/CONTEXT_DIVERSITY_OBSERVED_CHECKS_20261104.json),
con comandi, sessioni e provenienza; non vengono attribuiti ai vecchi log.
Passano inoltre i controlli reali di gradiente, prefisso/FFN
congelati, export, generazione KV, cache limitata, ripresa esatta di Adam e
atomicità del ripristino. Il comportamento predefinito conserva calendario,
digest e dataset precedenti. Nessuna di queste verifiche è un benchmark
di capacità conversazionale.

Il [manifesto](evidence/CONTEXT_DIVERSITY_PROVENANCE_20261104.json) conserva
SHA256 e dimensioni di ogni fonte originale e della copia gzip. Sono inclusi
gli otto JSON integrali, gli script diagnostici, gli snapshot del generatore
opzionale e del diagnostico protetto e i log dei test: circa 117 kB compressi,
senza pesi o stati Adam.
Per rieseguire i forward servono anche runtime, checkpoint e dati originali
identificati nei report; questo pacchetto conserva l'evidenza e gli script,
non distribuisce il modello. Il controllo locale
`python docs/evidence/CONTEXT_DIVERSITY_VALIDATE_20261104.py` verifica gli hash,
ricontrolla conteggi e delta dalle generazioni e controlla i link relativi.

Nessuna soglia è stata abbassata e nessuna risposta di valutazione viene
usata come nuova etichetta TRAIN. Questi confronti restano diagnostici locali
respinti; la revisione 2 approvata rimane invariata.
