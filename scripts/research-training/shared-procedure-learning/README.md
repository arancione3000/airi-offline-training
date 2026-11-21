# Verifiche che possono insegnare più procedure ad Airi

## Ipotesi

Oggi Airi verifica una formulazione o una procedura alla volta. In questo prototipo cerca risultati che possano chiarire più procedure insieme. Rappresenta ogni possibile configurazione del gruppo e conserva quelle compatibili con le risposte confermate. La prossima domanda è quella che massimizza il numero atteso di procedure risolte dalla risposta. L'aspettativa attribuisce peso uguale alle configurazioni rimaste. In caso di conflitto osservato il prototipo torna a ipotesi locali, mantenendo le risposte già verificate. Una verifica supplementare può controllare una conclusione trasferita contro le alternative rimaste possibili per la singola procedura. [Protocollo](PROTOCOL.md) · [Implementazione](trial.py)

Per esempio, due procedure possono usare la stessa convenzione per trattare un risultato negativo. Un esempio che produce un risultato positivo spesso lascia aperte entrambe le interpretazioni. Un esempio di segno opposto può chiarire la regola per più procedure, purché la condivisione sia applicabile. Il prototipo rappresenta tre scelte binarie che modificano gli ingressi o il segno, esegue le sette procedure numeriche del banco e consulta una funzione di riferimento sintetica. La simulazione valuta 72 casi e ne archivia tutte le domande e le risposte. Il verificatore simulato restituisce un solo risultato per verifica. Tutti i confronti ricevono gli stessi tipi di ingressi e lo stesso budget.

## Risultati

| Situazione | Ricerca indipendente | Ipotesi condivise con selezione per impatto |
| --- | --- | --- |
| Stessa configurazione per tutte, 8 casi | 7 verifiche mediane per tutte | 7 verifiche mediane per tutte |
| Una procedura fa eccezione, 32 casi | 4 procedure certe a 4 verifiche. Mediana di 7 per tutte | 6,06 procedure corrette in media a 4 verifiche. Mediana di 3 per tutte |
| Due procedure fanno eccezione, 32 casi | 32/32 casi completati senza errori | 26/32 casi con almeno una procedura erroneamente considerata risolta |

La scelta per entropia della risposta ha ottenuto gli stessi risultati della variante per impatto negli scenari misurati. Le 3 verifiche mediane misurano il vantaggio della condivisione quando il vincolo è corretto. Con la conferma indipendente di ciascuna procedura il costo totale torna a 7 nel caso con una sola eccezione. Con due eccezioni la conferma mirata ha bloccato le proposte errate in tutti i 26 casi in cui il metodo per impatto ne ha emesse. Il controllo fisso supplementare ha rilevato gli stessi 26 casi in questa simulazione. [Risultati per caso e per domanda](results.json.gz)

In una verifica separata il selettore per impatto ha risolto tutte le sette procedure in ciascuna delle 400 configurazioni compatibili con il vincolo. La conferma locale ha accettato tutte le proposte in quei casi.

La ragione del pareggio con la conferma locale emerge anche senza simulazione. Se tutte le risposte osservate sono compatibili con la stessa regola e il contratto ammette un'eccezione in qualsiasi procedura, ogni procedura ancora senza osservazioni potrebbe essere proprio l'eccezione. Per autorizzare il riuso con meno verifiche, occorre una fonte indipendente che attesti quali procedure condividono davvero la regola, per esempio una specifica unica con campi e funzioni comuni. Senza una simile evidenza, la condivisione è utile per anticipare proposte da verificare. Il prototipo resta in ricerca. Le convenzioni simulate e i risultati riguardano il banco procedurale, con la stessa base testuale del progetto.

## Ricerca di precedenti

DreamCoder apprende componenti riutilizzabili da programmi risolti. SmartLabel sintetizza domande mirate nella sintesi neurosimbolica. La ricerca sull'apprendimento attivo con trasferimento ha già studiato quante etichette risparmia la condivisione, e un altro lavoro studia insiemi di ipotesi condivise fra compiti. Il confronto di Airi valuta insieme ipotesi condivise, priorità del prossimo esempio, errore dovuto a eccezioni e costo della verifica locale. La ricerca mirata ha identificato precedenti per le idee di base. La priorità scientifica della combinazione specifica resta da accertare. [DreamCoder](https://doi.org/10.1145/3453483.3454080) · [SmartLabel](https://arxiv.org/abs/2508.15750) · [Transfer learning e apprendimento attivo](https://doi.org/10.1007/s10994-012-5310-y) · [Ipotesi condivise](https://papers.neurips.cc/paper_files/paper/2012/hash/9c82c7143c102b71c593d98d96093fde-Abstract.html)

## Riproduzione

L'esperimento usa soltanto Python e il banco approvato in `40ceb0931d823d71c1b151c5a46698659d4bd58e`. Il comando richiede un percorso di uscita nuovo.

```sh
python scripts/research-training/shared-procedure-learning/trial.py /tmp/airi-shared-procedure.json
```

## Continuazione

La prova successiva usa una dipendenza esplicita dal medesimo quoziente e misura il trasferimento a quattro altre procedure. [Esperimento sul contratto comune](../certified-procedure-transfer/README.md)
