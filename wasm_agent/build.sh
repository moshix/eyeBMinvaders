#!/bin/bash
# Build the wasm_agent crate for browser use via wasm-pack.
# Output goes to wasm_agent/pkg/ and is also copied to the game root
# so it can be served alongside index.html without extra config.

set -euo pipefail

cd "$(dirname "$0")"

echo "=== eyeBMinvaders WASM Agent Build ==="

# ------------------------------------------------------------------
# 1. Check / install wasm-pack
# ------------------------------------------------------------------
if ! command -v wasm-pack &> /dev/null; then
    echo "[*] wasm-pack not found. Installing via cargo..."
    if ! command -v cargo &> /dev/null; then
        echo "[!] cargo is not installed. Please install Rust first: https://rustup.rs"
        exit 1
    fi
    cargo install wasm-pack
    echo "[+] wasm-pack installed."
fi

# ------------------------------------------------------------------
# 2. Build the WASM target
# ------------------------------------------------------------------
echo "[*] Building wasm_agent (release, target=web)..."
wasm-pack build --target web --release --out-dir pkg

echo "[+] WASM build succeeded. Output in wasm_agent/pkg/"

# ------------------------------------------------------------------
# 3. Copy pkg to game root for easy serving
# ------------------------------------------------------------------
echo "[*] Copying pkg to ../wasm_agent_pkg/ ..."
rm -rf ../wasm_agent_pkg
cp -r pkg ../wasm_agent_pkg

echo "[+] Build complete."
echo ""
echo "Files:"
ls -lh pkg/*.wasm pkg/*.js 2>/dev/null || true
echo ""
echo "To test: serve the repo root with any HTTP server and open wasm_agent/www/index_wasm.html"
