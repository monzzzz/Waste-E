#!/usr/bin/env bash
# Start the CARLA 0.9.15 server headlessly and wait until it's ready.
# Usage:  bash start_carla.sh [carla_server_dir]

set -euo pipefail

CARLA_DIR="${1:-/workspace/carla_server}"
CARLA_SH="$CARLA_DIR/CarlaUE4.sh"

if [[ ! -f "$CARLA_SH" ]]; then
    echo "ERROR: $CARLA_SH not found."
    echo "  Check that /workspace/CARLA_0.9.15.tar.gz was fully extracted:"
    echo "    tar -xzf /workspace/CARLA_0.9.15.tar.gz -C /workspace/carla_server/"
    exit 1
fi

# Virtual display
export DISPLAY="${DISPLAY:-:99}"
if ! DISPLAY=$DISPLAY xdpyinfo &>/dev/null; then
    echo "[start_carla] Starting Xvfb on $DISPLAY …"
    Xvfb "$DISPLAY" -screen 0 1280x1024x24 &
    sleep 2
fi

echo "[start_carla] Launching CARLA server from $CARLA_DIR …"
mkdir -p /tmp/runtime-dir
chmod 700 /tmp/runtime-dir
cd "$CARLA_DIR"
env DISPLAY=$DISPLAY \
    XDG_RUNTIME_DIR=/tmp/runtime-dir \
    LD_PRELOAD=/tmp/fakeroot.so \
    "$CARLA_DIR/CarlaUE4/Binaries/Linux/CarlaUE4-Linux-Shipping" \
    CarlaUE4 -RenderOffScreen -opengl -nosound -carla-server -fps=20 \
    > /tmp/carla_server.log 2>&1 &
CARLA_PID=$!
echo "[start_carla] CARLA PID: $CARLA_PID"

# Wait for the port to open
echo -n "[start_carla] Waiting for port 2000 "
for i in $(seq 1 60); do
    if python3 -c "import socket; s=socket.socket(); s.settimeout(1); s.connect(('localhost',2000)); s.close()" 2>/dev/null; then
        echo " ready!"
        break
    fi
    echo -n "."
    sleep 2
done

# Quick connectivity test
python3 - <<'PYEOF'
import carla, sys
try:
    c = carla.Client("localhost", 2000)
    c.set_timeout(10.0)
    print(f"[start_carla] Server version: {c.get_server_version()}")
except Exception as e:
    print(f"[start_carla] WARNING: could not connect: {e}", file=sys.stderr)
PYEOF

echo "[start_carla] Done. CARLA server running (PID $CARLA_PID)."
echo "  Run simulation with:"
echo "    python3 /workspace/Waste-E/carla/run_carla_sim.py --steps 300"
