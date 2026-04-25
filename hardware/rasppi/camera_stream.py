from flask import Flask, Response, jsonify, render_template_string, request
import sys
import os
import re
import subprocess
import time
import threading

app = Flask(__name__)
CAM_PORT = int(os.getenv("CAM_PORT", "5000"))

_VIDEO_RE = re.compile(r"^video\d+$")

_procs: dict[str, subprocess.Popen] = {}
_procs_lock = threading.Lock()


def _is_primary_node(path: str) -> bool:
    name = os.path.basename(path)
    index_file = f"/sys/class/video4linux/{name}/index"
    try:
        with open(index_file) as f:
            return f.read().strip() == "0"
    except OSError:
        return False


def detect_cameras(timeout=3.0) -> list[str]:
    found = []
    for i in range(20):
        path = f"/dev/video{i}"
        if not os.path.exists(path):
            continue
        if not _is_primary_node(path):
            continue
        # Quick check: try opening with v4l2 to confirm capture capability
        result: dict = {}
        def _probe(p=path, r=result):
            try:
                proc = subprocess.run(
                    ["ffprobe", "-v", "quiet", "-f", "v4l2", "-i", p],
                    timeout=timeout, capture_output=True,
                )
                r["ok"] = proc.returncode == 0 or True  # ffprobe exits non-zero but device works
                r["ok"] = True
            except Exception:
                pass
        t = threading.Thread(target=_probe, daemon=True)
        t.start()
        t.join(timeout=timeout)
        if result.get("ok"):
            found.append(path)
            print(f"Found capture device: {path}")
    return found


def _start_ffmpeg(device: str) -> subprocess.Popen:
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-fflags", "nobuffer",
        "-flags", "low_delay",
        "-f", "v4l2", "-input_format", "mjpeg",
        "-framerate", "15",
        "-video_size", "640x480",
        "-i", device,
        "-f", "mjpeg", "-q:v", "5",
        "pipe:1",
    ]
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=0)


def _get_proc(device: str) -> subprocess.Popen:
    with _procs_lock:
        p = _procs.get(device)
        if p is None or p.poll() is not None:
            print(f"[camera_stream] Starting ffmpeg for {device}")
            _procs[device] = _start_ffmpeg(device)
        return _procs[device]


args = sys.argv[1:]
if args:
    cameras = []
    for idx, arg in enumerate(args):
        device = arg if not arg.isdigit() else f"/dev/video{arg}"
        cameras.append({"id": idx, "device": device, "open": os.path.exists(device)})
else:
    devices = detect_cameras()
    if not devices:
        print("No capture devices found, falling back to /dev/video0,1,2")
        devices = [f"/dev/video{i}" for i in range(3) if os.path.exists(f"/dev/video{i}")]
    cameras = [{"id": idx, "device": d, "open": os.path.exists(d)} for idx, d in enumerate(devices)]
    print(f"Using cameras: {[c['device'] for c in cameras]}")


def generate_frames(camera_id):
    device = cameras[camera_id]["device"]
    frame_count = 0
    while True:
        p = _get_proc(device)
        buf = b""
        try:
            while True:
                chunk = p.stdout.read(4096)
                if not chunk:
                    break
                buf += chunk
                while True:
                    soi = buf.find(b"\xff\xd8")
                    if soi == -1:
                        break
                    eoi = buf.find(b"\xff\xd9", soi + 2)
                    if eoi == -1:
                        break
                    frame = buf[soi:eoi + 2]
                    buf = buf[eoi + 2:]
                    frame_count += 1
                    if frame_count % 30 == 0:
                        print(f"camera {camera_id} ({device}) sent {frame_count} frames")
                    yield (b"--frame\r\n"
                           b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
        except Exception as exc:
            print(f"camera {camera_id} ({device}) ffmpeg error: {exc}")
        print(f"camera {camera_id} ({device}) restarting ffmpeg...")
        time.sleep(1)


@app.route('/')
def index():
    page = '''
    <html>
    <head>
        <title>Multi-Camera Stream</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            .camera { display: inline-block; margin: 10px; vertical-align: top; }
            .camera img { border: 1px solid #444; }
            .status { margin-bottom: 6px; }
        </style>
    </head>
    <body>
        <h1>Camera Feed</h1>
        {% for cam in cameras %}
        <div class="camera">
            <div class="status">
                <strong>Camera {{ cam.id }}</strong> ({{ cam.device }}) -
                {% if cam.open %}
                    <span style="color:green">open</span>
                {% else %}
                    <span style="color:red">not open</span>
                {% endif %}
            </div>
            <img src="/video_feed/{{ cam.id }}" width="640" height="480"
                 onerror="this.style.display='none'; this.nextElementSibling.style.display='flex'">
            <div style="width:640px; height:480px; background:#000; color:#fff; display:none; align-items:center; justify-content:center;">
                Camera not available
            </div>
        </div>
        {% endfor %}
    </body>
    </html>
    '''
    return render_template_string(page, cameras=cameras)


@app.route('/video_feed/<int:camera_id>')
def video_feed(camera_id):
    if camera_id < 0 or camera_id >= len(cameras):
        return 'Camera not found', 404
    return Response(generate_frames(camera_id), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/api/power', methods=['POST'])
def api_power():
    body = request.get_json(silent=True) or {}
    action = str(body.get('action') or '').strip().lower()
    if action not in ('shutdown', 'reboot'):
        return jsonify({'error': 'invalid action'}), 400

    def _run():
        time.sleep(1.5)
        if action == 'shutdown':
            subprocess.run(['shutdown', '-h', 'now'])
        else:
            subprocess.run(['reboot'])

    threading.Thread(target=_run, daemon=True).start()
    print(f'[power] {action} requested')
    return jsonify({'ok': True, 'action': action})


@app.route('/api/cameras')
def api_cameras():
    return jsonify([
        {
            'id': cam['id'],
            'device': cam['device'],
            'open': os.path.exists(cam['device']),
        }
        for cam in cameras
    ])


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=CAM_PORT, debug=False)
