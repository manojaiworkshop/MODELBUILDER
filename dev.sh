#!/usr/bin/env bash
# Starts the Rust backend (:4000) and the Next.js frontend (:3000). Ctrl+C stops both.
set -euo pipefail
cd "$(dirname "$0")"

(cd backend && cargo build --release)
[ -d frontend/node_modules ] || (cd frontend && npm install)

trap 'kill 0' EXIT INT TERM
(cd backend && PORT="${PORT:-4000}" ./target/release/modelbuilder-backend) &
(cd frontend && npm run dev) &
wait
