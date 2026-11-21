#!/usr/bin/env bash
# Run in a dedicated, clean clone owned by the server operator.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
: "${AIRI_CHAT_TOKEN:?Set AIRI_CHAT_TOKEN}"
AIRI_PYTHON="${AIRI_SERVER_PYTHON:-python3}"
AIRI_POLL="${AIRI_SERVER_UPDATE_SECONDS:-300}"
if ! [[ "$AIRI_POLL" =~ ^[0-9]{2,4}$ ]] || (( AIRI_POLL < 30 || AIRI_POLL > 3600 )); then
  echo "Set an update interval from 30 to 3600 seconds." >&2
  exit 2
fi
runner=""
start() {
  "$AIRI_PYTHON" "$ROOT/mobile/server/server.py" "$@" &
  runner=$!
}
stop() {
  if [[ -n "$runner" ]]; then
    kill "$runner" 2>/dev/null || true
    wait "$runner" 2>/dev/null || true
  fi
}
trap 'stop' EXIT
trap 'exit 0' INT TERM
start "$@"
while true; do
  sleep "$AIRI_POLL" &
  wait "$!" || true
  if ! kill -0 "$runner" 2>/dev/null; then
    wait "$runner"
    exit 1
  fi
  if ! timeout 45 git fetch --quiet origin refs/heads/main; then
    continue
  fi
  current=$(git rev-parse HEAD)
  next=$(git rev-parse FETCH_HEAD)
  if [[ "$current" == "$next" ]] || ! git merge-base --is-ancestor "$current" "$next"; then
    continue
  fi
  if [[ -n "$(git status --porcelain --untracked-files=no)" ]] ||
     ! git cat-file -e "$next:mobile/server/server.py"; then
    continue
  fi
  stop
  if ! git merge --ff-only "$next"; then
    echo "Impossibile applicare l'aggiornamento; riavvio il checkout corrente." >&2
  fi
  start "$@"
done
