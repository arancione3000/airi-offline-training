# Acquisizione e verifica incrementale di procedure

Nel prototipo, Airi propone una formula usando operazioni e risultati già approvati. Per distinguere più formule sceglie un ingresso, osserva il risultato atteso e conserva le formule compatibili. La versione proposta entra nel grafo sperimentale soltanto quando ogni risultato coincide con il riferimento su tutto il dominio. Il verificatore può riusare un nodo quando l'operazione è immutata e tutte le sue tabelle di ingresso coincidono esattamente con quelle già usate. Il riuso è valido perché le operazioni sono deterministiche e ricevono tutti gli ingressi esplicitamente. Un aggiornamento al solo riferimento continua a richiedere il confronto. Un errore conserva l'ultima versione approvata e indica il primo controesempio trovato. [Protocollo](PROTOCOL.md)

## Verifica di ventuno versioni

La sequenza è stata preparata separatamente dal verificatore. I tre metodi ammettono le stesse 18 versioni e producono gli stessi risultati su tutte le 4.128 coppie. Rifiutano gli stessi tre errori, tra cui un cambio di convenzione nel quoziente e una nuova specifica che il codice precedente non soddisfa. Le versioni rifiutate lasciano invariato il grafo approvato. Una versione corretta successiva viene ammessa. [Riferimento indipendente](fixtures.py) · [Rapporto completo](evidence/verification.json)

| Metodo | Valutazioni di nodi sulle 17 revisioni valide dopo la prima | CPU mediana per le 17 revisioni | CPU mediana per la sequenza completa |
| --- | ---: | ---: | ---: |
| Ricalcolo completo | 2.117.664 | 337,49 ms | 404,61 ms |
| Riuso per dipendenze non cambiate | 227.040 | 37,13 ms | 62,11 ms |
| Riuso se le dipendenze producono gli stessi valori | 94.944 | 20,28 ms | 44,58 ms |

Una valutazione è l'esecuzione di un nodo su una coppia di operandi. Il nuovo metodo fa il **58,2% di valutazioni in meno** rispetto al controllo incrementale che segue le modifiche alle dipendenze. Ci sono anche 160.992 posizioni sottoposte a confronto con il riferimento e 78.432 posizioni in confronti tra tabelle di dipendenza. Questi conteggi indicano la lunghezza delle tabelle confrontate. Il tempo CPU comprende i confronti e la validazione delle tabelle da parte del verificatore. Esclude l'impronta del risultato usata dal benchmark e la costruzione del riferimento, che richiede 33,55 ms mediani per l'intera sequenza. Le misure di tempo sono mediane di cinque ripetizioni sull'ambiente di sviluppo. Il primo passaggio esegue 119.712 valutazioni in tutte le modalità.

## Apprendimento di nuove formule

Le definizioni di cinque nuove procedure sono state nascoste al sintetizzatore. Il sistema ha appreso formule equivalenti con **12 risposte** su ingressi scelti, contro una mediana di **16,5** in otto ordini casuali informativi. Le etichette provengono dall'oracolo aritmetico nel benchmark. Le due estensioni sono state approvate dopo verifica esaustiva. In ciascuna, un singolo valore alterato su un ingresso mai interrogato ha causato il rifiuto dell'intera estensione. Il ripristino del riferimento corretto ha permesso l'ammissione. [Sintesi delle procedure](synthesis.py) · [Risultati e trascrizioni](evidence/synthesis.json)

Riutilizzare le predizioni delle formule con argomenti invariati ha ridotto le valutazioni della seconda ricerca da **9.964.992 a 1.647.072**. Sull'insieme delle due estensioni si passa da 18.282.912 a 9.964.992. Il confronto produce le stesse domande, formule, ammissioni e controprove. Il conteggio riguarda le esecuzioni delle formule candidate. La ricerca continua a confrontare le tabelle e calcolare quali ingressi distinguono le alternative.

## Limiti e precedenti

Il riuso è garantito entro l'insieme degli ingressi dichiarati, con operazioni deterministiche e riferimenti attendibili. Il banco di prova contiene formule di una sola operazione in aggiunta a procedure numeriche già disponibili, in una sequenza costruita per includere modifiche equivalenti. Le tabelle riutilizzabili vivono nella sessione di ricerca. Una nuova procedura richiede un oracolo o esempi affidabili e una specifica di ammissione. La tecnologia riguarda l'apprendimento procedurale e la sua verifica.

L'invalidazione dei soli risultati cambiati ha precedenti nei sistemi di build incrementale come Bazel e Salsa. La verifica incrementale di programmi ha precedenti come SiDECAR e UpProver. Qui lo stesso principio si applica al ciclo di Airi che propone formule da pochi esempi e poi decide quali esecuzioni già verificate riusare. I risultati misurano questa applicazione e la priorità mondiale resta da accertare. [Bazel](https://bazel.build/reference/skyframe) · [Build Systems à la Carte](https://doi.org/10.1017/S0956796820000088) · [SiDECAR](https://arxiv.org/abs/1304.8034) · [UpProver](https://doi.org/10.1007/s10703-023-00423-0)

## Riproduzione

```sh
D=scripts/research-training/incremental-procedure-verification
W=$(mktemp -d)
python -m unittest discover -s "$D" -v
python "$D/benchmark.py" "$W/verification.json"
python "$D/synthesis.py" --output "$W/synthesis.json"
```

Il benchmark confronta esiti, controesempi e impronte dei risultati, verifica il recupero dopo rifiuto e controlla che la strategia per valori riduca il lavoro. I conteggi e le decisioni sono deterministici, mentre i tempi CPU possono cambiare.
