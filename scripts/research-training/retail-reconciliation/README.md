# Riconciliazione di quantità vendute e stornate

Abbiamo provato ad automatizzare il calcolo delle unità spedite al netto degli storni da brevi resoconti. I valori sono aggregati del dataset Online Retail. I resoconti e le domande in italiano sono simulati. Il [protocollo](PROTOCOL.md) fissa il compito, i dati e i confronti.

## Riesecuzione dopo la correzione dei conflitti

Con entrambi i formati appresi, tutti i 24 casi da rivedere sono classificati correttamente nel test. Airi chiede chiarimenti nei quattro conflitti fra formati che prima producevano un numero. Le 28 risposte corrette e le 20 astensioni sulle richieste valide sono invariate. Nel banco con un solo formato il secondo schema rimane sconosciuto e i quattro casi misti non vengono riconosciuti. Il nuovo prototipo di riparazione manda a revisione ogni gruppo con una fonte sconosciuta. [Nuovo esito per caso](AFTER_CONFLICT_FIX.json) · [Prototipo](../adaptive-format-repair/README.md)

## Esito originario sui 72 casi finali

| Metodo | Saldi corretti su 48 | Richieste valide da rivedere | Numeri indebiti su 24 | Mediana per richiesta |
| --- | ---: | ---: | ---: | ---: |
| Airi, due procedure apprese | 28 | 20 | 4 | 0,551 ms |
| Qwen2.5-1.5B-Instruct Q4_K_M con sottrazione esatta | 26 | 22 | 5 | 722 ms |
| Script Python con i due formati configurati | 48 | 0 | 0 | 0,0039 ms |

Ogni sistema ha prodotto saldi corretti quando ha restituito un numero su una richiesta valida. La quarta colonna conta i casi in cui doveva astenersi. I risultati e le singole risposte sono in [RESULTS.json](RESULTS.json), nello stesso ordine di [CASES.json](CASES.json). Latenze su AMD EPYC 9V74 e Python 3.12.14, modello Qwen già caricato e due thread. Le mediane coprono il costo di risposta del metodo. [Dati e metodo](PROTOCOL.md)

Airi ha appreso il secondo formato da quattro esempi etichettati e lo ha attivato dopo dodici verifiche separate. Le automazioni corrette sono salite da 14 a 28. Tutti i 24 esiti precedentemente corretti sul primo gruppo sono rimasti corretti. I due aggiornamenti del banco hanno impiegato circa 20 e 22 ms di elaborazione, compreso il controllo di 989 contratti storici di instradamento. Il tempo per procurarsi e controllare le etichette è fuori dalla misura.

Airi ha risposto a tutte le 28 domande nel formato esatto. Ha passato a revisione tutte le 12 parafrasi e gli 8 resoconti con impaginazione diversa. Nelle quattro conversazioni con due resoconti incompatibili scritti nei due formati, prende il valore della prima procedura che trova una fonte valida. Ogni procedura verifica i conflitti nel proprio formato. Lo script raccoglie invece le coppie da entrambi i formati prima di decidere. Qwen ha risolto 13 domande con parafrasi o impaginazione diversa e ha prodotto cinque numeri indebiti nei casi con dato mancante o fonte dell’assistente.

Per questi due formati, lo script ha avuto la copertura completa con meno errori e minore latenza. La capacità di Airi che emerge è aggiungere una procedura da esempi con verifica e mantenere gli esiti precedenti. La prova non misura il tempo umano di configurazione, l’accuratezza su documenti di un cliente o la domanda di mercato. Il confronto riguarda un compito e un solo Transformer quantizzato.

## Riproduzione

Da eseguire dalla radice del repository. La prima riga usa il corpus congelato incluso. Per il confronto completo usare anche `--qwen`, con `AIRI_JOINT_LLM_CACHE` impostata come nel [benchmark Qwen precedente](../joint-action-20261121/README.md). Il modello e il binario sono verificati per hash.

```sh
D=scripts/research-training/retail-reconciliation
python "$D/run.py" "$D/CASES.json" /tmp/airi-retail-results.json
python "$D/run.py" "$D/CASES.json" /tmp/airi-retail-qwen.json --qwen
```

Per rigenerare il corpus da fonte servono OpenPyXL 3.1.5 e l’archivio [Online Retail](https://archive.ics.uci.edu/static/public/352/online+retail.zip). Il generatore verifica lo SHA-256 dell’archivio prima di leggerlo.

```sh
python "$D/prepare.py" /percorso/online-retail.zip /tmp/airi-retail-cases.json
cmp "$D/CASES.json" /tmp/airi-retail-cases.json
```

Attribuzione dei dati. Daqing Chen, Online Retail (2015), UCI Machine Learning Repository, [DOI 10.24432/C5BW33](https://doi.org/10.24432/C5BW33), [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Sono stati selezionati e aggregati i conteggi per prodotto e poi usati in messaggi di prova sintetici. Il corpus conserva solo quantità aggregate, senza identificativi cliente.
