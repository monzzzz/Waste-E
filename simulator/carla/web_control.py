"""
Waste-E live control panel — Flask web server.

Usage:
    python3 web_control.py [--port 7860] [--steps 300] [--num-scenes 1]

Open in browser:  http://localhost:7860  (or via cloudflared tunnel)
"""

from __future__ import annotations

import argparse
import glob
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

from flask import Flask, Response, jsonify, request

app = Flask(__name__)

NAV_FILE      = "/tmp/waste_e_nav.txt"
REPO_ROOT     = Path(__file__).resolve().parent.parent
SIM_SCRIPT    = str(Path(__file__).parent / "run_carla_sim.py")

_output_dir: Path  = REPO_ROOT / "output_carla"
_frames_dir: Path  = _output_dir / "frames"
_sim_proc: subprocess.Popen | None = None
_sim_args: list[str] = []   # filled from CLI at startup


# ── simulation process management ────────────────────────────────────────────

def _sim_running() -> bool:
    return _sim_proc is not None and _sim_proc.poll() is None


def _stop_sim():
    global _sim_proc
    if _sim_proc and _sim_proc.poll() is None:
        _sim_proc.terminate()
        try:
            _sim_proc.wait(timeout=8)
        except subprocess.TimeoutExpired:
            _sim_proc.kill()
    _sim_proc = None


def _start_sim():
    global _sim_proc
    _stop_sim()
    cmd = [sys.executable, SIM_SCRIPT] + _sim_args
    _sim_proc = subprocess.Popen(cmd, stdout=sys.stdout, stderr=sys.stderr)
    print(f"[web_control] Simulation started (pid {_sim_proc.pid})", flush=True)


def _latest_frame() -> Path | None:
    frames = sorted(_frames_dir.glob("**/*.jpg"))
    return frames[-1] if frames else None


# ── HTML ──────────────────────────────────────────────────────────────────────
_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Waste-E Control Panel</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    background: #0f1117; color: #e8eaf0;
    font-family: 'Segoe UI', system-ui, sans-serif;
    min-height: 100vh;
    display: flex; flex-direction: column; align-items: center;
    padding: 20px; gap: 16px;
  }
  h1 { font-size: 1.4rem; letter-spacing: .08em; color: #7eb8f7; margin-top: 4px; }
  #stream-box {
    width: 100%; max-width: 1200px;
    background: #1a1d27; border-radius: 12px; overflow: hidden;
    border: 1px solid #2a2d3a;
  }
  #feed { width: 100%; display: block; }
  #no-signal {
    display: flex; align-items: center; justify-content: center;
    height: 300px; color: #555; font-size: 1.1rem;
    flex-direction: column; gap: 10px;
  }
  #status-bar {
    width: 100%; max-width: 1200px;
    background: #1a1d27; border: 1px solid #2a2d3a;
    border-radius: 10px; padding: 10px 20px;
    display: flex; justify-content: space-between; align-items: center; gap: 12px;
    flex-wrap: wrap;
  }
  #current-cmd { color: #ffe082; font-weight: 700; }
  #sim-status  { font-size: .9rem; }
  #sim-status.running { color: #4caf50; }
  #sim-status.stopped { color: #ef5350; }
  .panels {
    width: 100%; max-width: 1200px;
    display: grid; grid-template-columns: 1fr 1fr; gap: 16px;
  }
  @media(max-width:700px){ .panels { grid-template-columns: 1fr; } }
  .card {
    background: #1a1d27; border: 1px solid #2a2d3a;
    border-radius: 12px; padding: 16px;
    display: flex; flex-direction: column; gap: 10px;
  }
  .card h2 { font-size: .9rem; color: #90a4ae; letter-spacing: .06em; text-transform: uppercase; }
  .btn-row { display: flex; gap: 8px; flex-wrap: wrap; justify-content: center; }
  button {
    padding: 12px 20px; border: none; border-radius: 9px;
    font-size: .95rem; font-weight: 600; cursor: pointer;
    transition: transform .08s; min-width: 110px;
  }
  button:active { transform: scale(.95); }
  .btn-straight { background: #2e7d32; color: #fff; }
  .btn-left     { background: #1565c0; color: #fff; }
  .btn-right    { background: #6a1a9a; color: #fff; }
  .btn-sleft    { background: #0d47a1; color: #fff; }
  .btn-sright   { background: #4a148c; color: #fff; }
  .btn-slow     { background: #e65100; color: #fff; }
  .btn-stop     { background: #b71c1c; color: #fff; min-width: 120px; }
  .btn-restart  { background: #f57f17; color: #fff; }
  .btn-save     { background: #00695c; color: #fff; }
  .btn-misc     { background: #37474f; color: #fff; }
  .btn-active   { outline: 3px solid #ffe082; }
  input[type=text], input[type=number] {
    background: #12151f; border: 1px solid #2a2d3a;
    border-radius: 8px; color: #e8eaf0;
    padding: 9px 12px; font-size: .9rem; outline: none; width: 100%;
  }
  input:focus { border-color: #7eb8f7; }
  label { font-size: .85rem; color: #90a4ae; }
  #save-msg { font-size: .85rem; color: #80cbc4; min-height: 1.2em; }
  .row { display: flex; gap: 8px; align-items: center; }
  .row label { white-space: nowrap; }
</style>
</head>
<body>
<h1>&#x1F916; Waste-E Live Control</h1>

<div id="stream-box">
  <img id="feed" src="/stream" alt="live feed"
       onerror="this.style.display='none';document.getElementById('no-signal').style.display='flex'">
  <div id="no-signal" style="display:none">
    <span style="font-size:2.5rem">&#x1F4F9;</span>
    Simulation not running — press Restart
  </div>
</div>

<div id="status-bar">
  <span>Nav: <span id="current-cmd">—</span></span>
  <span id="sim-status" class="stopped">&#9679; Stopped</span>
</div>

<div class="panels">

  <!-- Navigation -->
  <div class="card">
    <h2>Navigation</h2>
    <div class="btn-row">
      <button class="btn-straight" onclick="nav('Go straight')">&#x2B06; Straight</button>
    </div>
    <div class="btn-row">
      <button class="btn-sleft"  onclick="nav('Sharp left')">&#x21B0; Sharp L</button>
      <button class="btn-left"   onclick="nav('Turn left')">&#x2190; Left</button>
      <button class="btn-right"  onclick="nav('Turn right')">Right &#x2192;</button>
      <button class="btn-sright" onclick="nav('Sharp right')">Sharp R &#x21B1;</button>
    </div>
    <div class="btn-row">
      <button class="btn-slow" onclick="nav('Slow down')">&#x1F40C; Slow</button>
      <button class="btn-stop" onclick="nav('Stop')">&#x23F9; STOP</button>
    </div>
    <div class="row">
      <input id="custom-input" type="text" placeholder="Custom command…"
             onkeydown="if(event.key==='Enter') navCustom()">
      <button class="btn-misc" style="min-width:60px" onclick="navCustom()">Send</button>
    </div>
  </div>

  <!-- Simulation control -->
  <div class="card">
    <h2>Simulation</h2>
    <div class="row">
      <label>Steps</label>
      <input id="cfg-steps" type="number" value="300" min="10" max="2000">
    </div>
    <div class="row">
      <label>Scenes</label>
      <input id="cfg-scenes" type="number" value="1" min="1" max="20">
    </div>
    <div class="row">
      <label>Default nav</label>
      <input id="cfg-nav" type="text" value="Go straight">
    </div>
    <div class="btn-row">
      <button class="btn-restart" onclick="restartSim()">&#x21BA; Restart</button>
      <button class="btn-stop"    onclick="stopSim()">&#x25A0; Stop</button>
    </div>
    <div class="btn-row">
      <button class="btn-save" onclick="saveVideo()">&#x1F4BE; Save Video</button>
    </div>
    <div id="save-msg"></div>
  </div>

  <!-- Domain adapter -->
  <div class="card" style="grid-column: 1 / -1;">
    <h2>&#x2728; Sim-to-Real Domain Adapter (SDXL-Turbo)</h2>
    <div style="font-size:.85rem;color:#90a4ae;margin-bottom:4px;">
      Enable with <code style="color:#80cbc4">--use-adapter</code> when starting the sim.
      Adjust strength live while running.
    </div>
    <div class="row" style="gap:16px;flex-wrap:wrap;">
      <label style="white-space:nowrap">Strength: <span id="strength-val">0.35</span></label>
      <input id="strength-slider" type="range" min="0" max="1" step="0.05" value="0.35"
             style="flex:1;min-width:200px;accent-color:#7eb8f7"
             oninput="updateStrength(this.value)">
    </div>
    <div style="font-size:.8rem;color:#607d8b;">
      0 = raw CARLA &nbsp;|&nbsp; 0.25–0.40 = recommended (preserves structure) &nbsp;|&nbsp; 0.7+ = highly stylised
    </div>
    <div class="row" style="gap:8px;flex-wrap:wrap;">
      <label style="white-space:nowrap;min-width:80px">Prompt</label>
      <input id="adapter-prompt" type="text"
             value="photorealistic urban sidewalk, city street, real photograph, DSLR camera, sharp focus, natural lighting"
             style="flex:1">
      <button class="btn-misc" style="min-width:70px" onclick="sendPrompt()">Apply</button>
    </div>
  </div>

</div>

<script>
async function nav(cmd) {
  document.querySelectorAll('button').forEach(b => b.classList.remove('btn-active'));
  event.target.classList.add('btn-active');
  await fetch('/nav', {method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({cmd})});
  document.getElementById('current-cmd').textContent = cmd;
}

function navCustom() {
  const v = document.getElementById('custom-input').value.trim();
  if (v) nav(v);
}

async function restartSim() {
  const steps  = document.getElementById('cfg-steps').value;
  const scenes = document.getElementById('cfg-scenes').value;
  const navTxt = document.getElementById('cfg-nav').value;
  const r = await fetch('/sim/restart', {method:'POST',
    headers:{'Content-Type':'application/json'},
    body: JSON.stringify({steps: +steps, num_scenes: +scenes, nav_text: navTxt})});
  const d = await r.json();
  setSaveMsg(d.ok ? 'Restarting…' : 'Error: ' + d.error);
  // reconnect feed
  setTimeout(() => {
    const f = document.getElementById('feed');
    f.src = '/stream?' + Date.now();
    f.style.display = '';
    document.getElementById('no-signal').style.display = 'none';
  }, 2000);
}

async function stopSim() {
  await fetch('/sim/stop', {method:'POST'});
}

async function saveVideo() {
  setSaveMsg('Stitching video…');
  const r = await fetch('/sim/save', {method:'POST'});
  const d = await r.json();
  setSaveMsg(d.ok ? '&#x2714; Saved: ' + d.path : 'Error: ' + d.error);
}

function setSaveMsg(txt) {
  document.getElementById('save-msg').innerHTML = txt;
}

function updateStrength(val) {
  document.getElementById('strength-val').textContent = parseFloat(val).toFixed(2);
  fetch('/adapter/strength', {method:'POST',
    headers:{'Content-Type':'application/json'},
    body: JSON.stringify({strength: parseFloat(val)})});
}

function sendPrompt() {
  const p = document.getElementById('adapter-prompt').value.trim();
  fetch('/adapter/prompt', {method:'POST',
    headers:{'Content-Type':'application/json'},
    body: JSON.stringify({prompt: p})});
}

async function pollStatus() {
  try {
    const r = await fetch('/status');
    const d = await r.json();
    document.getElementById('current-cmd').textContent = d.cmd || '—';
    const el = document.getElementById('sim-status');
    if (d.sim_running) {
      el.className = 'running';
      el.textContent = '\u25CF Running';
    } else {
      el.className = 'stopped';
      el.textContent = '\u25CF Stopped';
    }
  } catch {}
}

setInterval(pollStatus, 1500);
pollStatus();
</script>
</body>
</html>
"""


# ── routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return _HTML


@app.route("/nav", methods=["POST"])
def set_nav():
    data = request.get_json(force=True, silent=True) or {}
    cmd = str(data.get("cmd", "Go straight")).strip()
    Path(NAV_FILE).write_text(cmd)
    return jsonify({"ok": True, "cmd": cmd})


@app.route("/status")
def status():
    try:
        cmd = Path(NAV_FILE).read_text().strip()
    except OSError:
        cmd = "—"
    frame = _latest_frame()
    sim_running = _sim_running()
    if not sim_running and frame:
        age = time.time() - frame.stat().st_mtime
        sim_running = age < 5.0
    return jsonify({"cmd": cmd, "sim_running": sim_running})


@app.route("/sim/restart", methods=["POST"])
def sim_restart():
    global _sim_args
    data = request.get_json(force=True, silent=True) or {}
    steps      = int(data.get("steps", 300))
    num_scenes = int(data.get("num_scenes", 1))
    nav_text   = str(data.get("nav_text", "Go straight"))

    # Clear old frames so the stream shows fresh content
    if _frames_dir.exists():
        shutil.rmtree(_frames_dir)
    _frames_dir.mkdir(parents=True, exist_ok=True)

    try:
        strength = float(Path("/tmp/waste_e_adapter_strength").read_text())
    except (OSError, ValueError):
        strength = 0.0
    try:
        prompt = Path("/tmp/waste_e_adapter_prompt").read_text().strip()
    except OSError:
        prompt = ""

    _sim_args = [
        "--steps", str(steps),
        "--num-scenes", str(num_scenes),
        "--nav-text", nav_text,
    ]
    if strength > 0.0:
        _sim_args += ["--use-adapter",
                      "--adapter-strength", str(strength)]
    if prompt:
        _sim_args += ["--adapter-prompt", prompt]
    try:
        _start_sim()
        return jsonify({"ok": True})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/sim/stop", methods=["POST"])
def sim_stop():
    _stop_sim()
    return jsonify({"ok": True})


@app.route("/sim/save", methods=["POST"])
def sim_save():
    """Stitch all scene frame dirs into mp4 files."""
    saved = []
    errors = []
    scene_dirs = sorted(_frames_dir.glob("scene_*"))
    if not scene_dirs:
        # flat frames (no scene subdirs)
        scene_dirs = [_frames_dir] if any(_frames_dir.glob("frame_*.jpg")) else []

    for sd in scene_dirs:
        frames = sorted(sd.glob("frame_*.jpg"))
        if not frames:
            continue
        out_path = _output_dir / f"{sd.name}.mp4"
        result = subprocess.run([
            "ffmpeg", "-y", "-framerate", "20",
            "-i", str(sd / "frame_%06d.jpg"),
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            str(out_path),
        ], capture_output=True)
        if result.returncode == 0:
            saved.append(str(out_path))
        else:
            errors.append(result.stderr.decode()[-200:])

    if saved:
        return jsonify({"ok": True, "path": ", ".join(saved)})
    return jsonify({"ok": False, "error": errors[0] if errors else "No frames found"}), 400


@app.route("/adapter/strength", methods=["POST"])
def adapter_strength():
    data = request.get_json(force=True, silent=True) or {}
    strength = float(data.get("strength", 0.35))
    strength = max(0.0, min(1.0, strength))
    Path("/tmp/waste_e_adapter_strength").write_text(str(strength))
    return jsonify({"ok": True, "strength": strength})


@app.route("/adapter/prompt", methods=["POST"])
def adapter_prompt():
    data = request.get_json(force=True, silent=True) or {}
    prompt = str(data.get("prompt", "")).strip()
    if prompt:
        Path("/tmp/waste_e_adapter_prompt").write_text(prompt)
    return jsonify({"ok": True})


@app.route("/stream")
def stream():
    def generate():
        while True:
            f = _latest_frame()
            if f:
                try:
                    data = f.read_bytes()
                    yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                           + data + b"\r\n")
                except OSError:
                    pass
            time.sleep(0.05)

    return Response(generate(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--output", default=str(REPO_ROOT / "output_carla"))
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--num-scenes", type=int, default=1)
    args = parser.parse_args()

    _output_dir = Path(args.output)
    _frames_dir = _output_dir / "frames"
    _frames_dir.mkdir(parents=True, exist_ok=True)

    _sim_args = ["--steps", str(args.steps), "--num-scenes", str(args.num_scenes)]

    print(f"[web_control] http://0.0.0.0:{args.port}")
    app.run(host="0.0.0.0", port=args.port, debug=False, threaded=True)
