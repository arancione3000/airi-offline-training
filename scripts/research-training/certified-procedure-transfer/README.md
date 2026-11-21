# Apprendere una funzione condivisa e riusarla in sei procedure

Airi può imparare una convenzione di divisione intera da una o due risposte verificate sul quoziente e riusarla per calcolare il resto, il multiplo raggiunto, il valore assoluto del resto e il multiplo associato al quoziente successivo. La verifica della divisibilità è indipendente dalla convenzione e funziona già a zero esempi. Le sei formule sono dichiarate nel contratto. L'apprendimento identifica quale delle tre convenzioni applicare, cioè divisione per difetto, troncamento verso zero o divisione euclidea. [Contratto](contract.json) · [Protocollo](PROTOCOL.md)

L'esecutore possiede una sola funzione per il quoziente e il resto. Le sei procedure leggono risultati derivati da quella funzione. Il sistema conserva le convenzioni compatibili con le risposte osservate. Per una richiesta specifica risponde quando tutte le alternative ancora compatibili danno lo stesso risultato. La ricevuta conserva le osservazioni e l'impronta dell'esecutore. Un ricevitore ricalcola le convenzioni rimaste e confronta le osservazioni con un'impronta ricevuta dal verificatore di fiducia. L'impronta verifica l'identità dei dati. L'attendibilità della fonte e il fatto che le procedure condividano davvero la funzione sono premesse dell'esperimento. [Esecutore](transfer.py) · [Funzione di riferimento indipendente](reference.py)

## Risultati riproducibili

| Convenzione | Risposte richieste sul quoziente | Apprendimento attivo per singola procedura | Limite inferiore favorevole al metodo indipendente |
| --- | ---: | ---: | ---: |
| Divisione per difetto | 2 | 10 | 5 |
| Troncamento verso zero | 2 | 10 | 5 |
| Divisione euclidea | 1 | 5 | 5 |

Il metodo indipendente può scegliere le sue domande, sa che le tre convenzioni sono possibili per ogni procedura e riconosce le procedure invarianti. Il limite inferiore gli concede di scegliere il percorso più breve sapendo già quale convenzione è vera. Anche con quel vantaggio richiede almeno una risposta per ognuna delle cinque procedure che dipendono dalla convenzione. Il metodo condiviso usa esclusivamente domande sul quoziente. Le altre quattro procedure dipendenti dalla convenzione ricevono zero osservazioni dirette.

Il confronto con il riferimento aritmetico, scritto separatamente in Python con `Decimal`, ha ottenuto **74.304/74.304** risposte corrette. Copre tutte le 4.128 coppie del dominio, le sei procedure e le tre convenzioni. **74.274** esecuzioni usano coppie mai mostrate al rispettivo candidato. La chat sperimentale ha superato **126 richieste** con controllo del risultato e dei due operandi e **15 richieste fuori dominio** con astensione. Un confronto appaiato delle risposte della chat prima e dopo l'aggiunta ha preservato **989 controlli storici e 600 esempi delle abilità**. Nei controlli di conservazione il modello nativo è rappresentato da una risposta fissa. [Rapporto e trascrizioni delle osservazioni](evidence/results.json)

La verifica ha respinto ricevute alterate, esecutori o contratti di un'altra versione, impronte di evidenza errate e osservazioni impossibili. Un controllo avverso ha assegnato una convenzione diversa alla sola procedura del resto, lasciando valido il contratto di partenza. Il trasferimento avrebbe prodotto **1.626 errori su 4.128**. Una fonte considerata attendibile che fornisce osservazioni sbagliate ma coerenti produce lo stesso limite. La divisione è definita su numeratore da −64 a 64 e divisore da −16 a 16, escluso zero. Le sei domande sperimentali sono testuali esatte. Il risultato misura l'efficienza dell'apprendimento procedurale in questo dominio.

## Precedenti

Relish tratta la sintesi di più programmi legati da specifiche relazionali. SmartLabel usa domande mirate nella sintesi neurosimbolica e ReGAL apprende librerie di funzioni riutilizzabili con verifiche di esecuzione. Il risultato ottenuto qui è il trasferimento misurato fra procedure di Airi con una dipendenza dichiarata, una ricevuta ricalcolabile e un confronto con apprendimento indipendente. Una priorità mondiale resta da dimostrare. [Relish](https://arxiv.org/abs/1809.02283) · [SmartLabel](https://arxiv.org/abs/2508.15750) · [ReGAL](https://proceedings.mlr.press/v235/stengel-eskin24a.html)

## Riproduzione

Dalla radice del repository, con Python 3.12 e il banco approvato della revisione `40ceb0931d823d71c1b151c5a46698659d4bd58e` o una revisione con lo stesso banco.

```sh
D=scripts/research-training/certified-procedure-transfer
W=$(mktemp -d)
python -m unittest discover -s "$D" -v
python "$D/trial.py" "$W/run"
cmp "$W/run/results.json" "$D/evidence/results.json"
```

Per provare il normale percorso della chat con una ricevuta di laboratorio, fornire al comando l'impronta dell'evidenza dal rapporto del verificatore. In questa dimostrazione i messaggi devono usare il formato degli operandi e una delle sei domande previste dall'adattatore.

```sh
cat > "$W/messages.json" <<'JSON'
[{"role":"user","content":"Valori divisione: -19 e -6."},
 {"role":"user","content":"Qual è il resto della divisione?"}]
JSON
DIGEST=$(python -c 'import sys,json; print(json.load(open(sys.argv[1]))["policies"]["floor"]["shared"]["evidence_sha256"])' "$D/evidence/results.json")
python "$D/chat_demo.py" --receipt "$W/run/floor.receipt.json" --trusted-evidence-sha256 "$DIGEST" --messages "$W/messages.json"
```
