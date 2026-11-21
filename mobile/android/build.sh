#!/usr/bin/env bash
set -euo pipefail

: "${ANDROID_BUILD_TOOLS:?Path to Android SDK build-tools, for example /opt/android/build-tools/35.0.0}"
: "${ANDROID_PLATFORM_JAR:?Path to Android SDK platforms/android-35/android.jar}"
: "${AIRI_ANDROID_KEYSTORE:?Path to the signing keystore}"
: "${AIRI_ANDROID_KEYPASS:?Signing keystore password}"
ROOT="$(cd "$(dirname "$0")" && pwd)"
OUT="${1:-$ROOT/airi-chat.apk}"
OUT="$(realpath -m "$OUT")"
mkdir -p "$(dirname "$OUT")"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

TOOLS="$ANDROID_BUILD_TOOLS"
mkdir -p "$WORK/classes" "$WORK/dex"
"$TOOLS/aapt2" compile --dir "$ROOT/res" -o "$WORK/resources.zip"
"$TOOLS/aapt2" link -I "$ANDROID_PLATFORM_JAR" --manifest "$ROOT/AndroidManifest.xml" \
  -o "$WORK/unsigned.apk" "$WORK/resources.zip"

if [[ -n "${AIRI_ECJ_JAR:-}" ]]; then
  java -jar "$AIRI_ECJ_JAR" -1.8 -proc:none -classpath "$ANDROID_PLATFORM_JAR" \
    -d "$WORK/classes" "$ROOT/src/com/airi/chat/AiriConnection.java" "$ROOT/src/com/airi/chat/MainActivity.java"
else
  javac -source 8 -target 8 -bootclasspath "$ANDROID_PLATFORM_JAR:$TOOLS/core-lambda-stubs.jar" -d "$WORK/classes" \
    "$ROOT/src/com/airi/chat/AiriConnection.java" "$ROOT/src/com/airi/chat/MainActivity.java"
fi
mapfile -d '' CLASSES < <(find "$WORK/classes" -name '*.class' -print0)
"$TOOLS/d8" --min-api 26 --lib "$ANDROID_PLATFORM_JAR" --output "$WORK/dex" "${CLASSES[@]}"
(cd "$WORK/dex" && zip -q "$WORK/unsigned.apk" classes.dex)
"$TOOLS/zipalign" -f 4 "$WORK/unsigned.apk" "$WORK/aligned.apk"
"$TOOLS/apksigner" sign --ks "$AIRI_ANDROID_KEYSTORE" --ks-pass env:AIRI_ANDROID_KEYPASS \
  --ks-key-alias "${AIRI_ANDROID_KEYALIAS:-airi}" --out "$WORK/signed.apk" "$WORK/aligned.apk"
"$TOOLS/apksigner" verify "$WORK/signed.apk"
cp "$WORK/signed.apk" "$OUT"
sha256sum "$OUT"
