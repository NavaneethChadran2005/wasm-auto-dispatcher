#!/usr/bin/env bash
# One command from your laptop (WSL): start the controller, open an SSH reverse tunnel to the Pi,
# launch a headless browser on the Pi that pulls payloads from the controller, wait until all are done.
#
#   bash scripts/run-on-pi.sh pi@192.168.1.50 pi4            # label "pi4"
#   bash scripts/run-on-pi.sh pi@192.168.1.50 pi4 --quick    # extra args go to controller.js
#   PORT=9000 bash scripts/run-on-pi.sh ...                  # use another port
set -uo pipefail
TARGET=${1:?usage: scripts/run-on-pi.sh user@pi-address [label] [controller options]}
LABEL=${2:-pi}
shift $(( $# >= 2 ? 2 : 1 ))
PORT=${PORT:-8080}
cd "$(dirname "$0")/.."

# --- pre-flight checks (clear errors instead of a silent hang) ---
if [ "$(id -u)" = "0" ]; then echo "Do not run this with sudo/root (it would use root's SSH keys and create root-owned results)." >&2; exit 1; fi
command -v node >/dev/null || { echo "node not found in this shell. Install Node 20+ (e.g. via nvm) inside WSL." >&2; exit 1; }
if ss -ltn 2>/dev/null | grep -q ":$PORT "; then echo "Port $PORT is already in use here; set another with PORT=9000 $0 ..." >&2; exit 1; fi
if ! ssh -o BatchMode=yes -o ConnectTimeout=8 "$TARGET" true 2>/dev/null; then
  echo "Cannot SSH to $TARGET without a password prompt. One-time fix:  ssh-copy-id $TARGET" >&2
  echo "(then re-run). If ssh-copy-id works but this still fails, test:  ssh $TARGET echo ok" >&2
  exit 1
fi

node controller.js --port "$PORT" --label "$LABEL" --exit-when-done "$@" &
CTRL=$!
SSH=
trap 'kill $CTRL 2>/dev/null; [ -n "$SSH" ] && kill $SSH 2>/dev/null' EXIT
sleep 1

# Runs ON the Pi. "localhost" through the tunnel is a secure context => threads (SharedArrayBuffer) work.
REMOTE='
URL="http://localhost:'"$PORT"'/worker.html"
PROFILE=$(mktemp -d)
for c in chromium-browser chromium; do command -v $c >/dev/null && B=$c && break; done
if [ -n "${B:-}" ]; then
  exec "$B" --headless=new --no-sandbox --disable-gpu --disable-renderer-backgrounding \
       --disable-background-timer-throttling --user-data-dir="$PROFILE" "$URL"
fi
for c in firefox firefox-esr; do command -v $c >/dev/null && F=$c && break; done
if [ -n "${F:-}" ]; then exec "$F" --headless --no-remote --profile "$PROFILE" "$URL"; fi
echo "no chromium or firefox found on the Pi" >&2; exit 1'

ssh -tt -o ExitOnForwardFailure=yes -o ServerAliveInterval=15 -R "$PORT:localhost:$PORT" "$TARGET" "$REMOTE" >/dev/null &
SSH=$!

wait $CTRL; code=$?
exit $code