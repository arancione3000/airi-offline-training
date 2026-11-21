# Verifica delle affermazioni su Airi

La verifica usa il codice pubblicato in `331ee5c34cd470e1ff3c9c0cb9f67330cad1ffc3`. La domanda è quali risultati si riproducono, su quali compiti valgono e se l'architettura rivendicata abbia precedenti pubblici.

## Risultati verificati

- Il benchmark di aggiornamento ha rigenerato tutte le decisioni, i controesempi e le impronte dei risultati. Nel caso di prova, il verificatore per valori impiega 94.944 valutazioni per le 17 revisioni valide dopo la prima, contro 227.040 del controllo incrementale strutturale e 2.117.664 del ricalcolo completo. Le tre modalità concordano sulle 18 ammissioni e sui tre rifiuti. La sintesi delle cinque formule ha rigenerato lo stesso rapporto byte per byte. [Esperimento pubblicato](../README.md)
- Un autore separato ha definito dieci funzioni aritmetiche per controllare i limiti sulla stessa griglia di 4.128 coppie. Il codice ha appreso e qualificato cinque funzioni, si è astenuto su due e il verificatore ha rifiutato tre formule candidate. Le cinque funzioni fallite sono assenti dallo spazio attuale di formule a una sola operazione. Lo conferma il confronto con tutte le tabelle candidate sull'intero dominio. Il test include sia casi semplici ammessi sia controesempi costruiti per mettere in evidenza quel limite. [Prova](scope_stress.py) · [Rapporto](scope_stress.json)
- Il confronto diretto già disponibile riguarda l'instradamento di comandi italiani e condivide il banco di procedure. L'interprete sperimentale ha risolto 30/54 richieste valide e ha eseguito 6 dei 54 casi da respingere. Qwen2.5-1.5B-Instruct quantizzato ha ottenuto 22/54 e 22/54 sugli stessi casi. Ho rieseguito Qwen con gli stessi pesi e lo stesso esecutore. Le osservazioni per formulazione coincidono byte per byte con quelle archiviate. Il confronto riguarda un singolo piccolo modello e una singola configurazione. [Risultati e riproduzione](../../joint-action-20261121/README.md)

## Novità scientifica

L'idea generale è già presente nella letteratura. SkillCompiler accumula abilità compilando le parti ben specificate in programmi validati e riusa i programmi nelle abilità successive. SmartLabel sceglie domande per distinguere programmi candidati. Bazel propaga gli aggiornamenti lungo le dipendenze e riusa le parti a valle quando il valore resta invariato. Relish applica la sintesi a più programmi con vincoli condivisi. Questi precedenti rendono insostenibile la rivendicazione che Airi sia l'unico progetto con un approccio simile. Il contributo verificato qui è la combinazione concreta di domande scalari, controllo esaustivo e riuso di tabelle finite nel ciclo sperimentale di Airi. La priorità di quella combinazione precisa resta aperta. [SkillCompiler](https://arxiv.org/abs/2610.39508) · [SmartLabel](https://arxiv.org/abs/2508.15750) · [Bazel Skyframe](https://bazel.build/reference/skyframe) · [Relish](https://arxiv.org/abs/1809.02283)

## Confronto con modelli di frontiera

Le metriche sulle 17 revisioni misurano il costo di verifica dello stesso programma con tre algoritmi. Il test aritmetico aggiuntivo misura la copertura del sintetizzatore. Una prova di superiorità nell'apprendimento continuo richiede le stesse sequenze di compiti, gli stessi strumenti e fonti di risposte, un budget comune per esempi e calcolo e test indipendenti dopo intervalli di inattività. Continual-ARC misura queste proprietà su sequenze con compiti nuovi e compiti che ritornano. SkillLearnBench aggiunge procedure tratte da compiti reali. Il dossier di Airi non contiene un punteggio su queste prove insieme ai sistemi di frontiera. [Continual-ARC](https://arxiv.org/abs/2611.14505) · [SkillLearnBench](https://arxiv.org/abs/2604.20087) · [Esempio di confronto pubblico su ARC-AGI](https://arcprize.org/results/openai-gpt-6-2)

## Riproduzione della prova di copertura

```sh
D=scripts/research-training/incremental-procedure-verification
W=$(mktemp -d)
PYTHONPATH="$D" python "$D/audit/scope_stress.py" "$W/scope_stress.json"
cmp "$W/scope_stress.json" "$D/audit/scope_stress.json"
```
