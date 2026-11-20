# Come Airi seleziona e riusa parti di candidati

Abbiamo confrontato i candidati sulla stessa chat con 30 esperti, 72 domande alternative già approvate e due valori per ogni formulazione. Per ottenere un punto devono coincidere la procedura, la risposta e la provenienza del dato in entrambe le esecuzioni. Il fallback nativo chiude il percorso con un segnaposto e conta come astensione. Le misure riguardano il riconoscimento di istruzioni italiane per sette operazioni. [Protocollo](PROTOCOL.md)

## Cosa si trasferisce

Il selettore ha imparato dai risultati su 216 formulazioni precedenti. Ha assegnato somma e moltiplicazione a V6 e le altre cinque operazioni al nuovo parser. Sulle 72 formulazioni positive di quei dati ottiene 40 risposte corrette, una in più del singolo donatore scelto con la stessa procedura. Sulle 28 richieste positive scritte successivamente entrambi ne risolvono una. Il vantaggio della selezione per operazione sui dati precedenti non si ripete sulla prova indipendente. Tutti i 70 testi di questa prova sono distinti dalle domande degli insiemi precedenti anche dopo la normalizzazione. [Selezione congelata](evidence/freeze.json)

| Metodo | Corrette sui 72 positivi usati per scegliere i donatori | Corrette sui 28 positivi nuovi | Attivazioni sui 42 negativi nuovi |
| --- | ---: | ---: | ---: |
| Chat approvata, prima del fallback | 0 | 0 | 0 |
| Grammatica V6 | 27 | 0 | 0 |
| Nuovo parser | 39 | 1 | 0 |
| Unione delle grammatiche | 42 | 1 | 0 |
| Classificatore con TRAIN iniziale | 5 | 1 | 0 |
| Stesso classificatore con 140 etichette in più | 13 | 0 | 0 |
| Unione delle grammatiche e classificatore con TRAIN iniziale | 44 | 2 | 0 |
| Unione delle grammatiche e classificatore con 140 etichette in più | 44 | 1 | 0 |
| MiniLM V4 | 10 | 3 | 1 |
| MiniLM V5 | 23 | 2 | 2 |
| Parti scelte per operazione | 40 | 1 | 0 |

Le 140 etichette migliorano il risultato del classificatore su materiale precedente. Sulle formulazioni nuove perde l'unica risposta che prima gestiva. Sul nuovo test tutti i metodi in tabella hanno conservato i 600 esempi del banco e i 989 contratti storici. Il controllo sui 989 è una verifica dei percorsi di risposta. I pesi nativi sono gli stessi in tutti i confronti. La soglia per passare alla verifica finale era di 14 richieste corrette su 28 e zero errori. [Metriche complete per operazione](evidence/results.json.gz) · [Esiti per formulazione](evidence/observations.json.gz)

Su 28 richieste valide, il nuovo parser perde 15 richieste quando cerca l'oggetto nominato, tre al controllo della sintassi e nove quando interpreta l'azione. Questo ha motivato una seconda prova dedicata ai nomi alternativi. [Diagnostica](BREAKDOWN.json)

### Inferenza testuale

Il confronto complementare usa un cross-encoder MiniLM multilingue addestrato su MNLI/XNLI. La prima calibrazione si è interrotta prima della valutazione. La generazione dei controesempi aveva assegnato un'etichetta negativa a tre coppie di frase e operazione già confermate nel TRAIN. Abbiamo corretto la generazione, mantenuto il controllo sulle vere contraddizioni delle etichette e congelato il codice corretto. Su 240 coppie di calibrazione, 66 delle quali positive, ogni coppia di soglie ammissibili della griglia produceva errori. La regola prestabilita ha selezionato l'astensione. Sulle 824 esecuzioni delle quattro prove, il risultato coincide con il percorso approvato. [Calibrazione e impronte delle due versioni](evidence/nli/freeze.json) · [Esiti](evidence/nli/results.json.gz)

### Nomi alternativi degli oggetti

Abbiamo ricavato un peso per ogni parola del catalogo in base al numero di oggetti in cui compare. Il componente usa questi pesi per proporre un oggetto, poi il parser deve interpretare il resto della richiesta. Su una seconda prova scritta indipendentemente, con 72 testi senza duplicati letterali rispetto agli insiemi precedenti, il riconoscimento degli oggetti è passato da 0 a 30 su 54 riferimenti validi. Su 18 riferimenti estranei, il componente ha proposto un oggetto sbagliato tre volte. Il parser ha respinto tutte e tre le proposte.

Sulla stessa prova entrambi i sistemi hanno risolto 0 dei 36 comandi validi e si sono astenuti su tutti i 36 comandi incompatibili. Sul primo insieme di 28, già esaminato per decidere l'intervento, il risultato è passato da 1 a 3. La seconda prova mostra che l'identificazione di un oggetto può migliorare senza rendere eseguibile la richiesta completa. Nella chat distribuita è attiva la correzione al controllo delle fonti. I candidati linguistici restano nel banco di ricerca. [Prova sui nomi alternativi](ground_holdout.json) · [Risultati per frase e per operazione](evidence/ground.json.gz)

## Controllo delle fonti nella chat

Abbiamo riprodotto un errore che compare dopo aver aggiunto una procedura al banco. Davanti a due fonti compatibili il router semantico poteva restituire il valore della prima perché il suo classificatore conosceva solo gli esperti più vecchi. Ora controlla le fonti di tutti gli esperti installati. Le domande alternative della stessa procedura conservano la precedenza e la provenienza. In 100 confronti appaiati su CPU, il costo mediano di questo controllo è sceso da 0,488 a 0,043 ms su due messaggi e da 5,093 a 0,666 ms su 64 messaggi. La prova usa una decisione di classificazione fissa e confronta l'intera risposta, provenienza inclusa. [Misure](GUARD_SPEED.json)

## Riproduzione

Usare Python 3.12, Git, le [dipendenze ONNX e il MiniLM con revisione fissata](../semantic-routing-20261116/REPRODUCTION.md). Gli aggiornamenti automatici modificano il numero di domande approvate. Recuperare i dati della revisione usata durante l'esperimento prima di eseguire il confronto. Eseguire questi comandi dalla radice del repository e scegliere una cartella nuova.

```sh
D=scripts/research-training/intent-transfer-20261120
W=$(mktemp -d)
BASE=55f835f203beb1e3d67be7bef9bd76b8734dda38
git show "$BASE:scripts/assistant-runtime/skills/approved-bank.json" > "$W/bank.json"
git show "$BASE:scripts/assistant-runtime/queries/approved-aliases.json" > "$W/aliases.json"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
# AIRI_QUERY_ENCODER_CACHE punta alla cartella contenente il MiniLM verificato.
python "$D/evaluate.py" --learned --transformers --select-parts --bank "$W/bank.json" --aliases "$W/aliases.json" --holdout "$D/prospective.json" --holdout-sha256 8e7bb5d6803fcaca13ec866aa1774d0222584ef3eb1df03324ea7b5bdf952825 --output "$W/first"
python "$D/diagnose.py" "$D/prospective.json" "$W/first/freeze.json" "$W/breakdown.json"
python "$D/bench_guard.py" "$W/cpu.json"
python "$D/ground_trial.py" "$D/ground_holdout.json" e818658ca35a3746547cef573af1eef78d34a44e2d08da8b589d9931ea3253c7 b7f579b6ba2c99bfff964bb27a85f057fd15022d28dc40905ed310fe9a9ec7ad "$W/ground.json" --bank "$W/bank.json" --aliases "$W/aliases.json"
```

Per il confronto NLI, copiare i tre file `model.onnx`, `tokenizer.json` e `config.json` nella stessa cartella locale e impostare `AIRI_NLI_CACHE` al suo percorso. Il programma ne verifica le impronte prima della calibrazione. [File ONNX del modello originale con revisione fissata](https://huggingface.co/MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli/tree/0a71e92a985b6e1ad1828cf67ce9c459639c1dca/onnx)

```sh
python "$D/nli_trial.py" --reference "$D/evidence/freeze.json" --holdout "$D/prospective.json" --bank "$W/bank.json" --aliases "$W/aliases.json" --output "$W/nli"
```
