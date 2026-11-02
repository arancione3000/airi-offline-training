# Apprendimento AIRI: diagnosi e correzioni del 2 novembre 2026

## Stato verificato

L'ultimo ciclo completo è la release `airi-offline-446076-40140414835-1-7`,
pubblicata il 25 ottobre. Ha tentato 24.896 aggiornamenti, eseguito 24 decisioni
di sviluppo e respinto sei traiettorie complete: zero adozioni. Il modello
approvato rimane `airi-native`, revisione 1, 23/92 risposte esatte e zero nei
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
addestrabili cambiati. Non contiene risposte o etichette e non cambia il
formato d'inferenza nativo. La migrazione crea Adam per il nuovo ambito una sola
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
