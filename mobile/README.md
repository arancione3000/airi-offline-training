# Airi Chat per Android

Airi Chat usa un servizio con l'API `/v1/chat`. Il telefono invia la conversazione al server e riceve le risposte di Airi. L'APK richiede Android 8.0 o successivo. Al primo avvio, aprire **Server** e inserire l'indirizzo e la chiave generata per il proprio servizio. La conversazione resta nella sessione dell'app. La chiave e l'indirizzo sono salvati nello spazio privato dell'app, con backup disabilitato.

Il server usa la memoria contestuale, il banco approvato e il modello nativo di Airi. Controlla periodicamente il puntatore approvato `airi-model-current`, verifica i file e carica una revisione più recente quando è disponibile. Se il download o la convalida non riescono, la revisione già caricata rimane disponibile. `follow-main.sh` consente di aggiornare anche le modifiche al runtime nel ramo `main` senza cambiare l'APK. Gli aggiornamenti ordinari del modello restano compatibili finché il server offre la stessa API. Eventuali modifiche al client, ai requisiti Android o al protocollo potranno richiedere un APK successivo.

## Servizio su un PC o su una macchina Linux

Usare un PC sempre acceso se si vuole poter chattare in qualsiasi momento. Per l'uso in rete locale il telefono e il PC devono raggiungersi, per esempio sulla stessa Wi-Fi. Il progetto non configura in automatico un host Internet. Per l'uso da remoto, configurare HTTPS sul proprio reverse proxy e usare l'indirizzo pubblico HTTPS nell'app. Il client accetta HTTP solo per indirizzi IP privati o loopback. HTTP in rete locale va usato su una rete fidata.

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

In rete locale l'indirizzo ha la forma `http://IP_DEL_PC:8787`. Usare come chiave nell'app il contenuto del file privato `chat-token`. Al primo avvio il server scarica il modello approvato. Ascoltare su `127.0.0.1` quando si usa un reverse proxy sullo stesso host. Per una prova con un checkpoint già scaricato, `server.py` accetta insieme `--checkpoint`, `--model-record` e `--offline`.

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

Per rendere disponibili in app anche procedure approvate da un registro indipendente, passare al server `--procedures`, `--procedure-reference` e `--procedure-reference-sha256` insieme. Il server usa lo stesso controllo della chat del repository e accoglie nuovi snapshot verificati durante la sessione. [Registro procedurale](../scripts/research-training/procedural-capacity-growth/README.md)
