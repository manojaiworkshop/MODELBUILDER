#!/usr/bin/env bash
# Remote (Ubuntu/Debian, root) setup + run: installs anything missing, builds, then starts the Rust
# backend (:4000) and the Next.js frontend (:11002). Safe to re-run — installed tools are skipped,
# the builds only redo work when sources changed, and old instances on the ports are stopped first.
# Ctrl+C stops both.
#   FRONTEND_PORT=11002 (public, nginx)  NEXT_PORT=3000  BACKEND_PORT=4000  (the last two are localhost-only)
#   PUBLIC_URL=https://<pod>-<port>.proxy.runpod.net   (what the browser opens; the page talks to the backend through the same URL, via the nginx proxy)
set -euo pipefail
cd "$(dirname "$0")"

FRONTEND_PORT="${FRONTEND_PORT:-11002}"
BACKEND_PORT="${BACKEND_PORT:-4000}"
NEXT_PORT="${NEXT_PORT:-3000}"
HOST="${HOST:-0.0.0.0}"
NODE_MAJOR=22

say() { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }
have() { command -v "$1" >/dev/null 2>&1; }
SUDO=""; [ "$(id -u)" -eq 0 ] || SUDO="sudo"

# ---- system packages ----
need_pkgs=()
for p in curl ca-certificates build-essential pkg-config libssl-dev psmisc nginx libvulkan1; do
  dpkg -s "$p" >/dev/null 2>&1 || need_pkgs+=("$p")
done
if [ ${#need_pkgs[@]} -gt 0 ]; then
  say "Installing system packages: ${need_pkgs[*]}"
  $SUDO apt-get update -y
  $SUDO env DEBIAN_FRONTEND=noninteractive apt-get install -y "${need_pkgs[@]}"
fi

# ---- Node.js ----
if ! have node || [ "$(node -p 'process.versions.node.split(".")[0]')" -lt 20 ]; then
  say "Installing Node.js ${NODE_MAJOR}"
  curl -fsSL "https://deb.nodesource.com/setup_${NODE_MAJOR}.x" | $SUDO bash -
  $SUDO apt-get install -y nodejs
fi

# ---- CUDA compiler bits for GPU training ----
# The NVIDIA driver ships with the pod, but the CUDA backend compiles its kernels at runtime and
# needs NVRTC + the CUDA headers. Vulkan can't see MIG GPU slices, so CUDA is the GPU path there.
# 12.8 is the first release that targets Blackwell. SKIP_CUDA=1 skips this (GPU runs then use the CPU).
CUDA_VER="${CUDA_VER:-12-8}"
if [ -z "${SKIP_CUDA:-}" ] && have nvidia-smi && [ ! -f /usr/local/cuda/include/nvrtc.h ]; then
  . /etc/os-release
  repo="ubuntu${VERSION_ID//./}"
  say "Installing CUDA ${CUDA_VER//-/.} NVRTC + headers (for GPU training)"
  tmp="$(mktemp -d)"
  curl -fsSL -o "$tmp/keyring.deb" "https://developer.download.nvidia.com/compute/cuda/repos/${repo}/x86_64/cuda-keyring_1.1-1_all.deb"
  $SUDO dpkg -i "$tmp/keyring.deb"
  rm -rf "$tmp"
  $SUDO apt-get update -y
  $SUDO env DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    "cuda-nvrtc-dev-${CUDA_VER}" "cuda-cudart-dev-${CUDA_VER}" "cuda-crt-${CUDA_VER}" "cuda-cccl-${CUDA_VER}"
fi
[ -d /usr/local/cuda ] && export CUDA_PATH=/usr/local/cuda LD_LIBRARY_PATH="/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# ---- Rust ----
[ -f "$HOME/.cargo/env" ] && . "$HOME/.cargo/env"
if ! have cargo; then
  say "Installing Rust (rustup)"
  curl -fsSL https://sh.rustup.rs | sh -s -- -y --profile minimal
  . "$HOME/.cargo/env"
fi

# ---- backend build (cargo is incremental) ----
say "Building backend"
(cd backend && cargo build --release)

# ---- frontend install + build ----
if [ ! -d frontend/node_modules ] || [ frontend/package-lock.json -nt frontend/node_modules ]; then
  say "Installing frontend dependencies"
  (cd frontend && npm ci && touch node_modules)
fi

# NEXT_PUBLIC_* is baked in at build time, so rebuild when it changes too.
env_stamp="frontend/.next/.backend-url"
# RunPod sets RUNPOD_POD_ID for the container's main process; SSH shells don't inherit it, so read PID 1's env.
POD_ID="${RUNPOD_POD_ID:-$(tr '\0' '\n' < /proc/1/environ 2>/dev/null | sed -n 's/^RUNPOD_POD_ID=//p' || true)}"
[ -n "$POD_ID" ] || [ -n "${PUBLIC_URL:-}${NEXT_PUBLIC_BACKEND_URL:-}" ] || { echo "Cannot detect the RunPod pod id; set PUBLIC_URL=https://<pod>-${FRONTEND_PORT}.proxy.runpod.net" >&2; exit 1; }
want_url="${NEXT_PUBLIC_BACKEND_URL:-${PUBLIC_URL:-https://${POD_ID}-${FRONTEND_PORT}.proxy.runpod.net}}"
if [ ! -f frontend/.next/BUILD_ID ] \
   || [ "$(cat "$env_stamp" 2>/dev/null || true)" != "$want_url" ] \
   || [ -n "$(find frontend/src frontend/public frontend/package.json frontend/next.config.ts -newer frontend/.next/BUILD_ID -print -quit 2>/dev/null)" ]; then
  say "Building frontend"
  (cd frontend && NEXT_PUBLIC_BACKEND_URL="$want_url" npm run build)
  printf '%s' "$want_url" > "$env_stamp"
else
  say "Frontend build is up to date"
fi

# ---- free the ports if an older instance is running ----
for p in "$BACKEND_PORT" "$NEXT_PORT" "$FRONTEND_PORT"; do
  fuser -k "$p"/tcp >/dev/null 2>&1 || true
done
sleep 1

# ---- run ----
# Hosting proxies (RunPod etc.) expose only one port, so $FRONTEND_PORT is a small nginx that serves the
# page (Next.js on 127.0.0.1:$NEXT_PORT) and forwards /socket.io, /api and /health to the backend
# (127.0.0.1:$BACKEND_PORT) - the browser then only ever talks to one origin.
mkdir -p .run
echo $$ > .run/remotedev.pid   # remotestop.sh kills exactly this launcher
cat > .run/nginx.conf <<NGINX
worker_processes 1;
pid $PWD/.run/nginx.pid;
error_log $PWD/.run/nginx-error.log warn;
events { worker_connections 1024; }
http {
  access_log off;
  client_body_temp_path $PWD/.run/body; proxy_temp_path $PWD/.run/proxy; fastcgi_temp_path $PWD/.run/fcgi;
  uwsgi_temp_path $PWD/.run/uwsgi; scgi_temp_path $PWD/.run/scgi;
  map \$http_upgrade \$conn_upgrade { default upgrade; '' close; }
  server {
    listen $FRONTEND_PORT;
    client_max_body_size 0;            # uploads are streamed to disk by the backend
    proxy_request_buffering off;
    proxy_read_timeout 3600s;
    proxy_http_version 1.1;
    proxy_set_header Host \$host;
    proxy_set_header Upgrade \$http_upgrade;
    proxy_set_header Connection \$conn_upgrade;
    location /socket.io/ { proxy_pass http://127.0.0.1:$BACKEND_PORT; }
    location /api/       { proxy_pass http://127.0.0.1:$BACKEND_PORT; }
    location = /health   { proxy_pass http://127.0.0.1:$BACKEND_PORT; }
    location /           { proxy_pass http://127.0.0.1:$NEXT_PORT; }
  }
}
NGINX
nginx -t -c "$PWD/.run/nginx.conf"

say "Starting backend :$BACKEND_PORT, frontend :$NEXT_PORT, proxy :$FRONTEND_PORT"
trap 'kill 0' EXIT INT TERM
(cd backend && HOST=127.0.0.1 PORT="$BACKEND_PORT" ./target/release/modelbuilder-backend) &
(cd frontend && npx next start -H 127.0.0.1 -p "$NEXT_PORT") &
nginx -c "$PWD/.run/nginx.conf" -g 'daemon off;' &
wait
