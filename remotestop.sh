#!/usr/bin/env bash
# Stops what remotedev.sh started: the Rust backend and the Next.js frontend.
#   FRONTEND_PORT=11002  BACKEND_PORT=4000
cd "$(dirname "$0")"
FRONTEND_PORT="${FRONTEND_PORT:-11002}"
BACKEND_PORT="${BACKEND_PORT:-4000}"

# the launcher itself first, so its trap doesn't matter and nothing respawns
if [ -f .run/remotedev.pid ]; then
  kill "$(cat .run/remotedev.pid)" 2>/dev/null && echo "stopped launcher"
  rm -f .run/remotedev.pid
fi
for p in "$BACKEND_PORT" "${NEXT_PORT:-3000}" "$FRONTEND_PORT"; do
  if fuser -k "$p"/tcp >/dev/null 2>&1; then echo "stopped :$p"; else echo "nothing on :$p"; fi
done
pkill -f modelbuilder-backend 2>/dev/null || true
pkill -f "$PWD/.run/nginx.conf" 2>/dev/null || true
