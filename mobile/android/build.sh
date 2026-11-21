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
ORT_VERSION=1.23.2
ORT_SHA256=82048d1f462218adae4ba76477089ab0ba76093d84f733540066db1a8ba6b827
ORT_AAR="${AIRI_ANDROID_ORT_AAR:-$WORK/onnxruntime.aar}"
if [[ -z "${AIRI_ANDROID_ORT_AAR:-}" ]]; then
  curl --fail --location --silent --show-error --retry 2 --max-time 120 \
    "https://repo.maven.apache.org/maven2/com/microsoft/onnxruntime/onnxruntime-android/$ORT_VERSION/onnxruntime-android-$ORT_VERSION.aar" \
    -o "$ORT_AAR"
fi
printf '%s  %s\n' "$ORT_SHA256" "$ORT_AAR" | sha256sum --check --status
ORT_JAR="$WORK/onnxruntime.jar"
unzip -p "$ORT_AAR" classes.jar > "$ORT_JAR"
mapfile -d '' SOURCES < <(find "$ROOT/src" -name '*.java' -print0)
"$TOOLS/aapt2" compile --dir "$ROOT/res" -o "$WORK/resources.zip"
"$TOOLS/aapt2" link -I "$ANDROID_PLATFORM_JAR" --manifest "$ROOT/AndroidManifest.xml" \
  -o "$WORK/unsigned.apk" "$WORK/resources.zip"

if [[ -n "${AIRI_ECJ_JAR:-}" ]]; then
  java -jar "$AIRI_ECJ_JAR" -1.8 -proc:none -classpath "$ANDROID_PLATFORM_JAR:$ORT_JAR" \
    -d "$WORK/classes" "${SOURCES[@]}"
else
  javac -source 8 -target 8 -bootclasspath "$ANDROID_PLATFORM_JAR:$TOOLS/core-lambda-stubs.jar" \
    -classpath "$ORT_JAR" -d "$WORK/classes" "${SOURCES[@]}"
fi
mapfile -d '' CLASSES < <(find "$WORK/classes" -name '*.class' -print0)
"$TOOLS/d8" --min-api 26 --lib "$ANDROID_PLATFORM_JAR" --output "$WORK/dex" "${CLASSES[@]}" "$ORT_JAR"
(cd "$WORK/dex" && zip -q "$WORK/unsigned.apk" classes.dex)
(cd "$ROOT" && zip -q "$WORK/unsigned.apk" licenses/*.txt)
mkdir -p "$WORK/native"
(
  cd "$WORK/native"
  unzip -q "$ORT_AAR" 'jni/*/*.so'
  mv jni lib
  zip -qr "$WORK/unsigned.apk" lib
)
"$TOOLS/zipalign" -P 16 -f 4 "$WORK/unsigned.apk" "$WORK/aligned.apk"
"$TOOLS/apksigner" sign --ks "$AIRI_ANDROID_KEYSTORE" --ks-pass env:AIRI_ANDROID_KEYPASS \
  --ks-key-alias "${AIRI_ANDROID_KEYALIAS:-airi}" --out "$WORK/signed.apk" "$WORK/aligned.apk"
"$TOOLS/apksigner" verify "$WORK/signed.apk"
cp "$WORK/signed.apk" "$OUT"
sha256sum "$OUT"
