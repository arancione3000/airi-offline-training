# Verifica di formulazioni italiane nuove

Il candidato ricava strutture di comando da 198 formulazioni approvate e prova a riutilizzarle con 18 famiglie di oggetti. La verifica confronta la versione approvata, i due selettori precedenti e il candidato. Una variante impara la struttura senza esempi della famiglia assegnata dal parser, così da verificare il trasferimento da altre famiglie. I programmi eseguibili e il controllo della fonte restano quelli del runtime approvato.

## Risultato

| Percorso | Corrette su 36 richieste valide | Esecuzioni indebite su 36 richieste da rifiutare |
| --- | ---: | ---: |
| Approvato, formulazioni esatte | 0 | 0 |
| Interprete oggetto-azione precedente | 27 | 9 |
| Filtro sulle richieste precedente | 31 | 8 |
| Strutture apprese dagli alias | 0 | 0 |
| Strutture apprese da altre famiglie | 0 | 0 |

Una richiesta conta come corretta se due esecuzioni con valori distinti riproducono la risposta prevista, la procedura e la provenienza del dato. Tra i casi da rifiutare ci sono richieste condizionali, negazioni, operazioni parziali e domande sul significato del testo. Il candidato più conservativo si è fermato su tutte le richieste, con copertura zero. Il filtro precedente ha riconosciuto più parafrasi e ha violato il requisito di zero esecuzioni indebite in otto casi. Entrambe le direzioni rimangono sperimentali. [Esiti completi](evidence/results.json)

Le 72 formulazioni erano già presenti nel workspace prima del riavvio. Le impronte del candidato e dei testi sono fissate nel driver. La verifica non ha trovato domande normalizzate identiche nei 12 insiemi di sviluppo o confronto consultati, compresi gli alias approvati. L’origine indipendente dei testi non è stata verificata, quindi i risultati sono riportati come prova su formulazioni riservate, con una verifica di sovrapposizione. I risultati dei due vecchi corpora sono riportati separatamente. [Testi e risposte attese](blind.json)

## Conservazione e riproduzione

Ogni metodo riproduce i 198 alias approvati con due valori, conserva i 600 esempi del banco e supera i 989 contratti storici di instradamento. Quando il runtime chiede di chiarire un conflitto e non esegue una procedura, il driver conta il caso come astensione. La prima lettura con il criterio storico aveva classificato quel messaggio testuale come risposta, producendo 30 falsi scarti identici anche nella versione approvata. La correzione riguarda la misura e lascia invariati i conteggi sulle 72 formulazioni. Il fallback nativo è simulato. [Osservazioni per caso](evidence/observations.json.gz)

Dalla radice del repository, con Python 3.12 e gli snapshot indicati dal driver.

```sh
D=scripts/research-training/prospective-language
W=$(mktemp -d)
python "$D/readout.py" "$W/risultati"
cmp "$D/evidence/results.json" "$W/risultati/results.json"
cmp "$D/evidence/observations.json.gz" "$W/risultati/observations.json.gz"
```

Se il banco o gli alias sono stati aggiornati, usa un checkout temporaneo del commit `452f23ed3d6fd5f055ec9d15cff2b27bbc488ae6` e copia in quel checkout questa cartella prima di eseguire i comandi.
