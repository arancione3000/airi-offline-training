# Espansione della capacità nativa con un ramo inizialmente neutro

L’esperimento estende l’ultimo blocco SwiGLU del modello nativo approvato, revisione 4. Il calcolo precedente continua attraverso la stessa proiezione, con gli stessi tensori. Un secondo ramo da 64 unità aggiunge 18.432 pesi, pari a tre matrici di dimensioni 64 × 96. L’inizializzazione mette a zero soltanto la proiezione di uscita. Il primo passo aggiorna quella proiezione e i passi seguenti possono aggiornare anche i pesi di ingresso del ramo.

La prova confronta i logits del runtime vero su 30 sequenze TRAIN, la generazione deterministica su 24 richieste in due campioni e tutti i certificati TRAIN tramite riproduzione dell’ultimo blocco. Congela i pesi precedenti, addestra solo il ramo aggiunto con 64 passi e usa esempi selezionati dal TRAIN e distillazione sui vecchi certificati. I campioni di generazione servono per la verifica, con un limite di 24 nuovi token per risposta. L’ampiezza della prova consente di rilevare regressioni ma non di autorizzare un aggiornamento della chat. Il modello approvato è identificato dall’impronta dei tensori e gli input sono fissati da hash nel codice.

## Riproduzione

Richiede Torch CPU, il sorgente nativo e gli asset locali del checkpoint descritti nella [prova di revisione 4](../native-projection-20261120/README.md). Usare un percorso di uscita nuovo.

```sh
PYTHONPATH=/workspace/airi-lab/native-source/computer \
  /workspace/airi-lab/venv/bin/python -u scripts/research-training/native-capacity-growth/trial.py \
  /workspace/airi-lab/rev4/checkpoint /workspace/airi-lab/rev4/experiment \
  /tmp/airi-native-capacity-reproduction
```

L’uscita contiene il protocollo eseguito, i risultati numerici e i pesi sperimentali del ramo. L’esportazione richiede ancora il modello di origine e il codice che installa il ramo. Il pilot riguarda un’aggiunta locale a un blocco e il confronto con le regole di ammissione. L’uso del ramo in produzione richiederebbe supporto esplicito del formato e prove più ampie.

## Esito

Il numero di parametri sale da 50.164.416 a 50.182.848. Prima dell’addestramento, la massima differenza misurata tra logits nativi è zero sulle 30 sequenze e 1.668 posizioni. Le 24 generazioni coincidono token per token. Restano corretti tutti i 897 certificati memorizzati. Il gradiente della nuova proiezione di uscita è diverso da zero al primo passo. Quello dei nuovi pesi di ingresso riceve un segnale al secondo passo.

Dopo 64 aggiornamenti il candidato conserva 889 certificati su 897. Nel campione di generazione passa da 3 a 1 risposte corrette su 24. La prova ha confermato che i pesi aggiunti si addestrano e che l’esportazione del ramo riproduce numericamente il candidato. Il candidato non ha superato la verifica e rimane scartato. [Protocollo eseguito](PLAN.json) · [Misure](SUMMARY.json)

L’ammissione sperimentale richiede inizializzazione neutra, esportazione esatta, 897 certificati conservati, zero risposte precedentemente corrette perse e almeno una nuova risposta corretta nel campione. La prova indipendente su un insieme riservato resta necessaria anche se questi controlli preliminari passano. Il controllo riproducibile ricalcola la decisione dal rapporto pubblico senza riaddestrare il candidato né usare dati riservati. [Decisione](SELECTION.json)

```sh
python scripts/research-training/native-capacity-growth/selection.py
```

Per verificare anche l’impronta del ramo originale locale, passare `--candidate /tmp/airi-native-capacity-20261121/candidate-branch.pt`. Il ramo è un artefatto di laboratorio conservato fuori dal repository.

## Limitare il ramo addestrato

Una seconda prova sceglie il peso del ramo fra potenze decrescenti di due, usando esclusivamente i 897 certificati TRAIN. La scala più grande fra quelle provate che conserva ogni certificato è 1/32. Poi la prova rilegge le stesse 24 richieste diagnostiche. Le risposte corrette restano 3 su 24, con zero acquisizioni e zero perdite. Una sequenza di token cambia senza cambiare l'esito della verifica. La proiezione riduce la regressione osservata e resta sperimentale. [Rapporto](PROJECTION.json)

```sh
PYTHONPATH=/workspace/airi-lab/native-source/computer \
  /workspace/airi-lab/venv/bin/python -u scripts/research-training/native-capacity-growth/guarded_projection.py \
  /workspace/airi-lab/rev4/checkpoint /workspace/airi-lab/rev4/experiment \
  /tmp/airi-native-capacity-20261121/candidate-branch.pt /tmp/airi-guarded-projection-new.json
```
