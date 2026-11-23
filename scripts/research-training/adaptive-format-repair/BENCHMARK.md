# Prova di adattamento a due nuovi formati

La prova simula la riconciliazione delle quantità spedite e stornate. Parte dallo schema `spedite/stornate`. Due fornitori simulati cambiano i nomi dei campi in sequenza, prima in `consegnate/rese`, poi in `evase/annullate`. Nel corpus variano l’ordine dei campi, il separatore, le maiuscole e gli spazi. Le quantità e i resoconti sono sintetici. [Corpus congelato](CASES.json)

Per ogni cambiamento, Airi riceve un primo resoconto mai visto e chiede quale etichetta indica le unità spedite. Riceve una sola risposta a questa domanda per schema e un insieme separato di 14 verifiche etichettate. Otto verifiche richiedono un saldo, sei richiedono astensione. Due verifiche oppongono lo schema candidato a quello già approvato. Il candidato deve preservare anche le verifiche del primo aggiornamento prima che il secondo sia attivato. Il programma riapre il registro dopo ciascuna attivazione.

Lo script mantenuto accetta lo stesso insieme di strutture sintattiche del corpus. Un operatore simulato dichiara una regola di associazione per ciascuno schema. Fornisce allo script la stessa informazione sul significato ricevuta da Airi. Lo script deve superare gli stessi esempi di ammissione e i casi protetti. Lo script con configurazione iniziale misura quante richieste gestirebbe lasciando la configurazione invariata.

I 54 casi finali restano fuori dall’attivazione e vengono valutati dopo il cambiamento corrispondente. Includono resoconti coerenti ripetuti, conflitti anche con saldo coincidente, dati incompleti e formati irrisolti. Il primo aggiornamento verifica 22 casi protetti della configurazione iniziale. Il secondo ripete quei 22 casi e le 14 verifiche del primo aggiornamento.

## Esito

| Metodo | Saldi corretti su 40 | Astensioni corrette su 14 | Saldi errati o indebiti |
| --- | ---: | ---: | ---: |
| Airi | 40 | 14 | 0 |
| Script mantenuto | 40 | 14 | 0 |
| Script con configurazione iniziale | 4 | 14 | 0 |

Airi ha risposto correttamente anche ai due resoconti che avevano innescato le domande. Sui 22 casi protetti, tutte le decisioni sono rimaste uguali e corrette. Nei due casi finali con un unico schema sconosciuto e privo di una conferma esterna, Airi ha proposto un chiarimento e ha lasciato il saldo in sospeso. [Risultati per caso](RESULTS.json)

Entrambi i metodi aggiornati richiedono **due conferme del significato delle etichette** e le stesse **28 verifiche etichettate**. Airi costruisce le due voci di configurazione da sola. Nello script l’operatore deve dichiararle. Il confronto conta le informazioni e gli interventi previsti dal flusso. I minuti di lavoro, il costo di produrre le verifiche e il tempo necessario per l’integrazione reale restano da misurare. I risultati si riferiscono a cambiamenti di formato in questa grammatica e a dati sintetici congelati. Il confronto con il modello linguistico è documentato a parte. [Prova Qwen](QWEN.md)

## Riproduzione

Dalla radice del repository, Python 3.10 o successivo e sola libreria standard.

```sh
D=scripts/research-training/adaptive-format-repair
W=$(mktemp -d)
python "$D/cases.py" "$W/CASES.json"
cmp "$D/CASES.json" "$W/CASES.json"
python "$D/benchmark.py" "$W/RESULTS.json"
cmp "$D/RESULTS.json" "$W/RESULTS.json"
```

Il generatore usa il seme `429017`. L’esecutore controlla lo SHA-256 del corpus prima dell’esecuzione. Gli insiemi di arrivo, ammissione e prova finale sono distinti. Le quantità sono nuove, salvo le ripetizioni intenzionali che verificano la coerenza fra resoconti. Il risultato è stabile byte per byte e omette le misure di latenza.
