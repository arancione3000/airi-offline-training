# Apprendimento AIRI: correzioni del 2 novembre, esito verificato il 3 novembre 2026

## Esito attuale

Il ciclo si è concluso: il modello offline `airi-native` è passato alla
**revisione 2**, con un miglioramento delle loss di sviluppo e tutte le
precedenti risposte corrette conservate nel campione adottato. Le risposte
esatte restano però **23/92**, i compiti di contesto restano a zero e il test
composizionale finale è **0/64**. È un progresso persistito sulle loss, non
una nuova capacità dimostrata. I gate finali nativi continuano a fallire,
come già accadeva nella baseline: nessuna qualificazione o promozione operativa.

La [release terminale](https://github.com/arancione3000/airi-offline-training/releases/tag/airi-offline-449276-41523612006-1-3)
conserva il campione selezionato. Lo stato verificato contiene `continue=false`:
il training finito non è più in esecuzione. Conteggi, digest e risultati sono
nel [riepilogo terminale](evidence/CONTEXT_TERMINAL_VERIFICATION_20261102.json).

## Stato verificato

Alla partenza, l'ultimo ciclo completo era la release `airi-offline-446076-40140414835-1-7`,
pubblicata il 25 ottobre. Ha tentato 24.896 aggiornamenti, eseguito 24 decisioni
di sviluppo e respinto sei traiettorie complete: zero adozioni. Il modello
approvato era ancora `airi-native`, revisione 1, 23/92 risposte esatte e zero nei
compiti contestuali. I successivi job dei controller non sono nuovi
aggiornamenti dei pesi. I sei rifiuti sono un arresto esplicito del budget,
non un errore di ripresa da aggirare.

Gli archivi MODEL e OPTIMIZER e lo stato sono verificati contro l'indice SHA256.
La baseline è stata riprodotta localmente: NLL italiana 1,215602964, inglese
1,050221499; sviluppo umano inglese 3,377343334, 23/92. Piccole differenze
numeriche dalla telemetria originale sono inferiori a 1e-5. L'audit ricostruisce
16.241 esempi authored TRAIN, 195 di sviluppo e gli stessi split umani filtrati
866/21/27. Il test finale non seleziona modifiche o iperparametri.

## Correzioni

Il calendario storico richiedeva il trasferimento dalla banca negli slot 3 e
11, gli unici `old-authored` e `composition`. Con la banca popolata, questi
gruppi sparivano dal training; la telemetria finale ne conferma zero
aggiornamenti. Il replay ora può sostituire solo slot in sovrannumero e deve
lasciare almeno un'acquisizione per ogni gruppo. Tutti gli slot umani e nativi
rimangono protetti. Le richieste multiple vengono considerate insieme.

La precedente testa addestrabile aveva 36.864 pesi, corpo interamente
congelato e proiezione protetta di rango 48 su 96 dimensioni. Il nuovo modo
`cached_prefix` allena attenzione e normalizzazioni dell'ultimo blocco, norma
finale e testa. Il prefisso e i grandi FFN rimangono congelati; il gradiente
attraversa l'ultimo FFN fino all'attenzione. La cache contiene solo attivazioni
del prefisso immutabile, ha limiti espliciti e rifiuta configurazioni o ambiti
addestrabili cambiati. Le chiavi identificano gli input tokenizzati; non vengono
memorizzate predizioni o etichette della loss. Il formato d'inferenza rimane
nativo. La migrazione crea Adam per il nuovo ambito una sola
volta; le riprese successive ripristinano esattamente pesi, momenti e RNG.

## Nuovo esperimento finito

Configurazione: `scripts/context-learning/SPEC.json`, ramo immutabile
`native-context-learning-20261102`, seed verificato al passo 446076. Sorgente
nativa `d078e718dfba0d68babd31de90fa55c0beac1951`, riferimento ancestrale
`e949157c88a582b7e52ca46001afcf8da8ddd22e`, Torch `2.14.1+cpu`.

Ogni 16 aggiornamenti: quattro dialoghi umani revisionati, quattro esercizi
contestuali, due composizioni, due dialoghi curati, due authored e due ripassi
nativi. Nessuno slot ordinario viene sostituito dalla banca. Due ricordi
certificati per lingua proteggono acquisizioni già corrette di TRAIN. I tre
profili esplorano LR 1e-4/3e-5/1e-5 e conservazione/KL 4/8/16. Non sono nuove
soglie di qualità: tutti i gate originali e ogni risposta di sviluppo
precedentemente corretta restano obbligatori.

Verifica ogni 256 aggiornamenti, traiettorie di 1.024, massimo 12 decisioni,
tre traiettorie respinte, 3.600 secondi di ottimizzazione e quattro job. Solo
checkpoint pubblicati e hash-committed possono proseguire sullo stesso commit;
un ramo modificato interrompe la continuazione. L'ultimo job dispone di un
chunk più ampio per finalizzare il budget. Il modello approvato sostituisce
il precedente solo dopo i gate; nessuna qualificazione generale o promozione
nel runtime operativo deriva dall'esistenza di un checkpoint.

## Verifiche e interpretazione

Passano 61 test del trasferimento, sette della continuazione, sei di
integrazione del nuovo modo e i controlli nativi di gradienti, cache limitata,
export, generazione KV e ripresa esatta di Adam. Passano anche le suite
standard del workflow `verify-offline-runner`. La raccolta indiscriminata di
tutte le directory include vecchi test deadline-extension che richiedono un
checkout storico assente; non sono stati indeboliti o modificati.

`scripts/transfer-training/learning_probe.py` esegue confronti isolati con gli
stessi pesi iniziali e dati, riportando risposte perse/guadagnate, NLL e gate.
Un confronto iniziale usa un calendario e un numero di ricordi esplicitamente
diversi dalla nuova ricetta; l'opzione `--spec` riproduce la ricetta esatta.
Gli esperimenti CPU concorrenti non giustificano confronti di velocità.
Una loss di TRAIN discendente o un test tecnico superato non costituiscono
prova di conversazione migliore. Risultati delle verifiche remote e dei
confronti reali devono essere letti separatamente prima di dichiarare un
miglioramento delle capacità.

## Primo confronto reale: nessun candidato adottabile

[Evidenza completa](evidence/CONTEXT_PAIRED_256_20261102.json): stessi pesi
iniziali, seed, 256 aggiornamenti, LR 3e-5, conservazione/KL 4 e otto ricordi
per aggiornamento. Questo è un confronto isolato tra ambiti addestrabili;
il suo calendario differisce dalla nuova ricetta dichiarata sopra. Il prefisso
allena effettivamente 74.304 parametri, contro i 36.864 della testa.

| Misura di sviluppo | Baseline | Testa | Ultima attenzione + testa |
|---|---:|---:|---:|
| Risposte esatte | 23/92 | 23/92 | 22/92 |
| Nuove risposte corrette | — | 0 | 0 |
| Precedenti risposte perse | — | 0 | 1 |
| NLL nativa italiana | 1,215603 | 1,207146 | 1,203115 |
| NLL nativa inglese | 1,050221 | 1,044242 | 1,033811 |
| NLL umana inglese | 3,377343 | 3,373881 | 3,348914 |
| Gate originale | — | Respinto | Respinto |

La testa non produce il guadagno minimo richiesto. Il candidato con attenzione
migliora le NLL native oltre l'1% in entrambe le lingue, ma perde un saluto
italiano precedentemente corretto. Nessuna risposta contestuale nuova è
acquisita. Il gate rifiuta correttamente entrambi. Tutte le 92 risposte e i
rispettivi flag della baseline coincidono con l'archivio originale; i file
del checkpoint sorgente restano identici. I candidati rimangono isolati e non
vengono copiati sul modello approvato. La conservazione delle acquisizioni
rimane un problema misurato da risolvere, non un successo da dichiarare.

Pubblicazione del codice: [PR 1](https://github.com/arancione3000/airi-offline-training/pull/1),
CI [41517383337](https://github.com/arancione3000/airi-offline-training/actions/runs/41517383337)
superata. Il preflight reale separato
[41517578143](https://github.com/arancione3000/airi-offline-training/actions/runs/41517578143)
è **superato**: 32 aggiornamenti, NLL del compito di TRAIN
0,000607238 → 0,000600461, risposte di sviluppo invariate e campione immutato.
Il guadagno insufficiente non viene adottato;
[evidenza dal log remoto](evidence/CONTEXT_PREFLIGHT_20261102.json).

## Ricetta esatta e controllo della conservazione

Il [confronto locale della ricetta esatta](evidence/CONTEXT_EXACT_RECIPE_256_20261102.json)
usa il primo profilo congelato: LR 1e-4, conservazione/KL 4 e due ricordi per
lingua. Dopo 256 aggiornamenti: **20/92**, zero risposte nuove, tre risposte
precedenti perse e contesto ancora zero. NLL native 1,189334 italiana e 1,018015
inglese; NLL umana inglese 3,285620. Il gate originale respinge correttamente
il candidato nonostante loss inferiori. Export nativo esatto e checkpoint
sorgente immutato. Non è un esito positivo sulle capacità.

L'[audit delle sole memorie TRAIN](evidence/CONTEXT_TRAIN_MARGIN_AUDIT_20261102.json)
trova 278 esempi già corretti: 144 italiani e 134 inglesi. I saluti di TRAIN
hanno margini minimi circa 8,8–8,9, superiori ai minimi delle altre memorie.
Campionare soltanto i margini più bassi non è quindi una soluzione dimostrata
per il saluto di sviluppo che regredisce.

Un [ripasso uniforme e predefinito di TRAIN](evidence/CONTEXT_RETENTION_SWEEP_128_20261102.json)
parte dal candidato isolato a 22/92 ed esegue 128 aggiornamenti, senza scegliere
righe dal benchmark. Usa Adam nuovo dichiarato, LR 1e-5, conservazione/KL 16,
il campione originale come insegnante e il riferimento ancestrale per il
ripasso nativo. Il risultato resta **22/92**: nessuna nuova risposta, il
precedente saluto rimane perso. Anche questo candidato viene respinto.
L'export nativo è esatto e il checkpoint sorgente resta immutato. Il ripasso
completo non ha risolto il divario tra TRAIN e nuove formulazioni; non è stato
inserito nella ricetta congelata. Il [riepilogo locale](evidence/CONTEXT_LOCAL_SUMMARY_20261102.json)
riporta separatamente i primi tre confronti, con calendari e profili effettivi.

## Ciclo remoto concluso

I job [41518404434](https://github.com/arancione3000/airi-offline-training/actions/runs/41518404434),
[41520044217](https://github.com/arancione3000/airi-offline-training/actions/runs/41520044217)
e [41523612006](https://github.com/arancione3000/airi-offline-training/actions/runs/41523612006)
sono proseguiti automaticamente sullo stesso commit
`d3f5039c35953625b059263d0be08f70adc74a01`, senza una nuova chiamata
all'assistente per ogni ripresa. Hanno completato 3.200 aggiornamenti tentati,
823,908 secondi di ottimizzazione e 12 decisioni di sviluppo. Il limite delle
12 valutazioni ha terminato il ciclo prima del budget di 3.600 secondi.
Si osservano tre rollback: due al termine di traiettorie respinte complete,
uno forzato su una traiettoria parziale al limite delle valutazioni.

Al passo **446588** il candidato completo perdeva risposte precedenti, mentre
l'interpolazione **alpha 0,25** ha superato i gate di sviluppo originali e
la conservazione di tutte le precedenti risposte corrette. Questa singola
adozione ha prodotto la revisione 2, salvata nella
[release accettata](https://github.com/arancione3000/airi-offline-training/releases/tag/airi-offline-446588-41518404434-1-2)
e conservata nella release terminale. Le successive proposte respinte non
l'hanno sostituita. Non sono state cambiate soglie o risposte di valutazione.

| Misura di sviluppo | Revisione 1 | Revisione 2 selezionata | Variazione |
|---|---:|---:|---:|
| Risposte esatte | 23/92 | 23/92 | Nessuna nuova o persa |
| NLL nativa italiana | 1,215603 | 1,206915 | −0,71% |
| NLL nativa inglese | 1,050222 | 1,037827 | −1,18% |
| NLL umana inglese | 3,377343 | 3,330514 | −1,39% |
| Compiti contestuali corretti | 0 | 0 | Nessun guadagno |

Tutti i 92 flag di correttezza del campione adottato coincidono con la
baseline; il totale uguale non nasconde scambi di risposte corrette e sbagliate.
La NLL umana qui misurata riguarda il sottoinsieme inglese rimasto dopo i
filtri: non dimostra un miglioramento della conversazione italiana.

Il test composizionale finale, escluso dalla selezione, rimane **0/64**.
I controlli finali nativi di conservazione e degenerazione falliscono per
accuratezza linguistica ed entropia; entrambi fallivano già prima di questo
ciclo. Non costituiscono una qualificazione superata. Il puntatore remoto
`AIRI_CURRENT.json` conferma revisione 2, identità e digest del campione offline,
con `production_qualified=false` e `live_promoted=false`.

Gli asset JSON di stato, indice e risultato finale sono stati scaricati e
verificati contro dimensioni e SHA256 remoti; lo stato è vincolato dall'indice
e l'impronta è quella della ricetta congelata. I descrittori degli archivi
MODEL/OPTIMIZER coincidono con i digest remoti: in questa verifica terminale
non sono stati riscaricati i byte degli archivi. Il
[riepilogo verificato](evidence/CONTEXT_TERMINAL_VERIFICATION_20261102.json)
conserva questa distinzione. Non sono pubblicati pesi o log duplicati nel
repository per documentare l'esito.
