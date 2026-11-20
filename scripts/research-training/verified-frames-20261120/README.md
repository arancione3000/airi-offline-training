# Trasferimento composizionale con budget di verifica

La prima prova usa 84 richieste, 42 valide e 42 da respingere. I selettori ordinano solo il testo e gli identificatori opachi. Dopo il congelamento delle graduatorie il verificatore consulta le etichette. Lo stesso meccanismo trasforma ogni richiesta confermata in una struttura applicabile a un altro oggetto con uguale operazione e classe grammaticale. La verifica è simulata su etichette curate. I tempi di selezione non includono il costo umano dell'annotazione.

A 16 verifiche, V5 trova in media 12 strutture valide e permette 6 ricombinazioni frase-oggetto, contro 8,875 strutture e 7,125 ricombinazioni con selezione casuale. La baseline TF-IDF si astiene sulla selezione delle 84 proposte e ricade nello stesso ordine casuale. La grammatica produce 7,5 ricombinazioni. Nel primo ordine fissato, tutti gli approcci si astengono sui 144 negativi indipendenti e sui 360 casi sfavorevoli di ricombinazione. Superano anche 600 controlli memorizzati e 989 storici. Ottengono tutti 0/72 sui positivi indipendenti, quindi il componente resta sperimentale.

Una diagnostica successiva massimizza il prodotto fra la probabilità stimata di convalida e il numero di oggetti compatibili. Sullo stesso corpus produce 20,125 ricombinazioni, rispetto a 18,625 con il solo numero di oggetti compatibili. Il conteggio è per frase-oggetto. Le due varianti di valore nel test di esecuzione sono osservazioni dipendenti. Gli otto ordini di spareggio riusano il corpus e non vanno interpretati come otto repliche indipendenti.

## Conferma su proposte nuove

Un secondo corpus di 56 richieste, scritto senza conoscere le graduatorie, verifica il confronto con lo stesso budget di 16 etichette. Il selettore basato sulla compatibilità degli oggetti produce in media 13,875 ricombinazioni, mentre l'aggiunta del punteggio V5 porta la media a 15,75. Il vantaggio misurato è del 13,5%. Nel primo ordine fissato, l'esecuzione sul banco passa da 28 a 30 risposte corrette su 38, con zero risposte indebite su 242 richieste incompatibili e zero regressioni sui controlli del banco. Il risultato riguarda nuove combinazioni di frasi e oggetti compatibili. [Graduatorie ed esecuzione](CONFIRMATION_EXECUTION.json)

## Riproduzione

Il banco e le formulazioni iniziali provengono dal commit `07424b75b806f5ae9712c1a8fb8ed53df63a15aa`. Il runner congela una copia degli artefatti presenti prima della valutazione. Per confrontare esattamente le misure archiviate, estrarre gli artefatti di quel commit in un checkout isolato del codice dell'esperimento. Usare l'ambiente encoder locale del [protocollo V5](../typed-routing-20261118/PROTOCOL.md).

```sh
experiment=scripts/research-training/verified-frames-20261120
python "$experiment/trial.py" freeze /tmp/airi-new-composition
python "$experiment/trial.py" qualify /tmp/airi-new-composition
python "$experiment/transfer_priority.py" /tmp/airi-new-priority.json
python "$experiment/priority_confirmation.py" /tmp/airi-new-confirmation
python "$experiment/bench_bank.py" 07424b75b806f5ae9712c1a8fb8ed53df63a15aa /tmp/airi-new-speed.json
```

## Verifica della velocità

Su 180 turni per versione con banco a 30 esperti e 32 formulazioni, il percorso di instradamento passa da 91,58 a 6,63 millisecondi di mediana a caldo e da 112,64 a 10,16 millisecondi al 95º percentile. La prova alterna competenze esatte, formulazioni aggiunte e astensione. Tutti i risultati sono identici. Il fallback è una stringa fissa. La misura riguarda il percorso di selezione sullo stesso ambiente CPU. Il costo della generazione neurale richiede una misura distinta. [Misure](BANK_SPEED.json)
