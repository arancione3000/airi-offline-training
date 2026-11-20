# Riproduzione

La sorgente è la revisione 4 del checkpoint `airi-offline-469820-42263066151-1-1`. Servono gli asset di modello, stato e dati dell'esperimento originale, verificati contro il relativo manifest. La prova accetta soltanto il digest dei tensori approvato. Usa l'ambiente CPU e il runtime nativo del [precedente esperimento](../ffn-output-20261112/README.md).

```sh
PYTHONPATH=/workspace/airi-lab/native-source/computer \
  /workspace/airi-lab/venv/bin/python scripts/research-training/native-projection-20261120/trial.py \
  /workspace/airi-lab/rev4/checkpoint /workspace/airi-lab/rev4/experiment \
  /tmp/airi-native-projection-reproduction
```

La directory di destinazione deve essere nuova. Il file `PLAN.json` riporta gli hash fissati all'avvio e `REPORT.json` registra le risposte e le misure per entrambe le versioni. `head-update.pt` contiene solo la correzione candidata, il fattore applicato e l'hash della sorgente. Il repository conserva il riepilogo e il codice della prova.
