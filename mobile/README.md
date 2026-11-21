# Airi Chat per Android

Airi Chat 1.1 richiede Android 8.0 o successivo. Al primo avvio scarica automaticamente da GitHub il modello nativo verificato. Dopo il download, le risposte vengono generate sul telefono e la chat funziona anche offline. L’app verifica gli aggiornamenti mentre è aperta e conserva l’ultimo modello valido se la rete o l’aggiornamento falliscono. La cronologia usata per generare le risposte resta nella sessione sul dispositivo.

Il workflow `publish-android-model.yml` segue il puntatore `airi-model-current`. Esporta i pesi approvati in ONNX, confronta risposte e valori di uscita contro il runtime nativo e pubblica il modello mobile con il relativo SHA-256. I candidati in addestramento arrivano al telefono dopo la promozione e la verifica dell’esportazione. Aggiornare i pesi finché il formato resta compatibile non richiede di reinstallare l’APK.

L’app usa il modello nativo in locale con un contesto fino a 256 token per contenere il costo sul telefono. Il servizio HTTP opzionale descritto sotto resta disponibile per chi usa l’API Python con il banco delle conoscenze e le procedure verificate.

## Servizio su un PC o su una macchina Linux

Usare un PC sempre acceso se si vuole poter chattare in qualsiasi momento. Per l'uso in rete locale il telefono e il PC devono raggiungersi, per esempio sulla stessa Wi-Fi. Il progetto non configura in automatico un host Internet. Per l'uso da remoto dell’API, configurare HTTPS sul proprio reverse proxy. Usare HTTP in rete locale su una rete fidata.

Servono Git, GitHub CLI, Python 3.12, Torch CPU e il runtime Python nativo di Airi. Il setup qui sotto usa il sorgente nativo fissato dal progetto. È concepito per Linux o WSL. Se l'ambiente nativo è già disponibile, si possono riusare il suo Python e la directory `computer`.

```sh
git clone https://github.com/arancione3000/airi-offline-training.git airi-mobile-server
cd airi-mobile-server
AIRI_MOBILE_ROOT="$HOME/.local/share/airi-mobile"
mkdir -p "$AIRI_MOBILE_ROOT"
git clone --filter=blob:none --no-checkout https://github.com/arancione3000/airi-pc-bootstrap.git "$AIRI_MOBILE_ROOT/native"
git -C "$AIRI_MOBILE_ROOT/native" sparse-checkout set computer
git -C "$AIRI_MOBILE_ROOT/native" checkout --detach d078e718dfba0d68babd31de90fa55c0beac1951
python3.12 -m venv "$AIRI_MOBILE_ROOT/venv"
"$AIRI_MOBILE_ROOT/venv/bin/pip" install torch==2.14.1+cpu --index-url https://download.pytorch.org/whl/cpu
gh auth status
export PYTHONPATH="$AIRI_MOBILE_ROOT/native/computer"
export AIRI_SERVER_PYTHON="$AIRI_MOBILE_ROOT/venv/bin/python"
(umask 077 && [ -s "$AIRI_MOBILE_ROOT/chat-token" ] || (umask 077 && python3 -c 'import secrets; print(secrets.token_urlsafe(32))' > "$AIRI_MOBILE_ROOT/chat-token"))
export AIRI_CHAT_TOKEN="$(cat "$AIRI_MOBILE_ROOT/chat-token")"
mobile/server/follow-main.sh --bind 0.0.0.0 --port 8787
```

In rete locale l'indirizzo dell’API ha la forma `http://IP_DEL_PC:8787`. Passare il contenuto del file privato `chat-token` nella testata `Authorization` come token Bearer. Al primo avvio il server scarica il modello approvato. Ascoltare su `127.0.0.1` quando si usa un reverse proxy sullo stesso host. Per una prova con un checkpoint già scaricato, `server.py` accetta insieme `--checkpoint`, `--model-record` e `--offline`.

## Compilazione e firma

Servono SDK Android 35, build tools 35, Java e una chiave di firma. Conservare la stessa chiave per aggiornare il pacchetto Android esistente. Le credenziali di accesso ad Airi non fanno parte dell'APK.

```sh
export ANDROID_BUILD_TOOLS=/percorso/android-sdk/build-tools/35.0.0
export ANDROID_PLATFORM_JAR=/percorso/android-sdk/platforms/android-35/android.jar
export AIRI_ANDROID_KEYSTORE=/percorso/airi-chat.p12
export AIRI_ANDROID_KEYPASS
mobile/android/build.sh /percorso/Airi-Chat.apk
```

La build supporta anche ECJ impostando `AIRI_ECJ_JAR`, utile se è installato soltanto il JRE. Per la compilazione sono stati usati i build tools Android 35.0.0, API 35 ed ECJ 3.38.0.

Per rendere disponibili al servizio HTTP anche procedure approvate da un registro indipendente, passare al server `--procedures`, `--procedure-reference` e `--procedure-reference-sha256` insieme. Il server usa lo stesso controllo della chat del repository e accoglie nuovi snapshot verificati durante la sessione. [Registro procedurale](../scripts/research-training/procedural-capacity-growth/README.md)
