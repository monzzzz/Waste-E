#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if ! command -v ffprobe >/dev/null 2>&1; then
  echo "[manual_sim] Installing ffmpeg for ffprobe..."
  apt-get update
  apt-get install -y ffmpeg
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "[manual_sim] Installing uv..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

echo "[manual_sim] Syncing Python dependencies..."
uv sync --project "$SCRIPT_DIR"

cat <<EOF
[manual_sim] Ready.

Example:
  bash $SCRIPT_DIR/run_offline_eval.sh \\
    $SCRIPT_DIR/../../hardware/recordings/2026-05-07_08-11-52 \\
    --policy constant_velocity \\
    --overwrite
EOF
