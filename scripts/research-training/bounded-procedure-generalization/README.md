# Apprendimento di procedure a più passaggi

Il candidato riceve esempi di ingresso e uscita e cerca sequenze di operazioni tipizzate. Può chiedere al massimo tre esiti a un riferimento indipendente quando due ipotesi danno risposte diverse. I risultati di qualificazione arrivano al registro dopo la scelta del programma. I casi finali vengono valutati dopo avere chiuso tutte le ammissioni. Il registro ha un’identità legata al codice dell’interprete e conserva ogni procedura ammessa con l’impronta delle verifiche. [Protocollo e impronta del corpus](PROTOCOL.md)

## Risultati del primo confronto

| Metodo | Procedure ammesse | Risposte corrette su 172 richieste valide | Astensioni corrette su 34 richieste incompatibili | Risposte errate o indebite | Etichette aggiuntive |
| --- | ---: | ---: | ---: | ---: | ---: |
| Memoria esatta | 0 | 0 | 34 | 0 | 0 |
| Una operazione, richieste per disaccordo | 0 | 0 | 34 | 0 | 0 |
| Fino a tre operazioni, richieste in ordine | 9 | 118 | 34 | 0 | 1 |
| Fino a tre operazioni, richieste per disaccordo | 9 | 118 | 34 | 0 | 1 |

Le procedure ammesse appartengono a tre domini. Ottengono 36/36 risposte valide sui tre compiti di testo, 42/42 sui tre compiti di sequenze e 40/42 sui tre compiti con record. Su otto delle nove procedure ogni esito finale è corretto. Dopo la riapertura del registro, 66 verifiche delle procedure ammesse hanno riprodotto gli esiti attesi. A ogni nuova ammissione il registro ha ricontrollato le qualificazioni già salvate. [Esiti per caso, procedure e certificati](RESULTS.json)

Il candidato ha imparato, fra le altre operazioni, a pulire un elenco separato da virgole e trasformarlo in una riga, a sommare i valori positivi di un campo e a normalizzare etichette rimuovendo i duplicati. Gli esempi portano a programmi composti da operazioni disponibili nel motore. La prova misura generalizzazione a dati nuovi dentro questo spazio. La memoria esatta parte da ingressi volutamente disgiunti, e il controllo a un passo misura il costo di togliere la composizione su compiti costruiti per richiederla.

Fra le 54 richieste valide rimaste senza risposta, 40 appartengono a tre compiti fuori dalla libreria e 12 a un compito su cui la qualificazione ha respinto un programma indistinguibile sui dati iniziali. Negli altri due casi il motore ha respinto un record con metadati annidati, anche se il campo da elaborare era valido. Una sola richiesta al riferimento ha distinto due possibili campi in un compito. I due metodi hanno scelto lo stesso caso. Questo campione non dimostra un vantaggio della selezione per disaccordo.

L’esperimento usa un ID di compito esplicito. L’interpretazione del linguaggio libero e l’acquisizione di operazioni fuori dalla libreria richiedono prove distinte. Il confronto valuta un componente simbolico su dati sintetici e lascia alla ricerca l’eventuale vantaggio rispetto a modelli neurali o automazioni reali.

## Riproduzione

Da GitHub o dalla cartella del progetto con Python 3.10 o successivo e la sola libreria standard.

```sh
D=scripts/research-training/bounded-procedure-generalization
W=$(mktemp -d)
python "$D/oracle_cases.py" "$W/CASES.json"
cmp "$D/CASES.json" "$W/CASES.json"
python "$D/run_trial.py" "$W/RESULTS.json" --cases "$W/CASES.json"
cmp "$D/RESULTS.json" "$W/RESULTS.json"
```

Il report include lo SHA-256 del corpus e dei programmi Python che hanno eseguito la prova. Il risultato archiviato conserva anche i rifiuti e le astensioni. [Corpus](CASES.json) · [Esecutore](run_trial.py)
