#!/usr/bin/env bash
# Starts the Rust backend (:4000) and the Next.js frontend (:3000). Ctrl+C stops both.
set -euo pipefail
cd "$(dirname "$0")"

(cd backend && cargo build --release)
[ -d frontend/node_modules ] || (cd frontend && npm install)

trap 'kill 0' EXIT INT TERM
# HOST=0.0.0.0 so the backend is reachable from other devices on the LAN too (the frontend's
# next.config.ts already allows that origin) — 127.0.0.1-only was unreachable from anywhere but
# this machine, showing as "Backend offline" the moment you opened the app via a LAN IP.
(cd backend && HOST="${HOST:-0.0.0.0}" PORT="${PORT:-4000}" ./target/release/modelbuilder-backend) &
(cd frontend && npm run dev) &
wait
