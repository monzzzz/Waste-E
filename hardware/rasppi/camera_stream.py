from flask import Flask, Response, jsonify, render_template_string
import cv2
import sys
import os
import time
import threading

app = Flask(__name__)
CAM_PORT = int(os.getenv("CAM_PORT", "5000"))

def open_camera(device):
    for backend in (cv2.CAP_V4L2, cv2.CAP_ANY):
        cap = cv2.VideoCapture(device, backend)
        if cap.isOpened():
            # Force MJPEG to reduce USB bandwidth (~10x less than raw YUYV).
            # Without this, 3 cameras together exceed the Pi's shared USB 2.0 bus.
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 15)
            return cap
        cap.release()
    return cv2.VideoCapture(device)

def _test_camera(path, result):
    """Try to read a frame; store cap in result if successful."""
    cap = open_camera(path)
    if not cap.isOpened():
        cap.release()
        return
    for _ in range(5):
        ret, frame = cap.read()
        if ret and frame is not None:
            result['cap'] = cap
            return
    cap.release()

def detect_cameras(timeout=3.0):
    """Return list of {device, cap} for devices that can capture frames.
    Each device is tested in a thread with a timeout to avoid blocking on hung devices."""
    found = []
    for i in range(20):
        path = f'/dev/video{i}'
        if not os.path.exists(path):
            continue
        result = {}
        t = threading.Thread(target=_test_camera, args=(path, result), daemon=True)
        t.start()
        t.join(timeout=timeout)
        if 'cap' in result:
            found.append({'device': path, 'cap': result['cap']})
            print(f'Found capture device: {path}')
        else:
            print(f'Skipping {path} (timed out or cannot capture frames)')
    return found


args = sys.argv[1:]
if args:
    # Manual device list — open each one
    cameras = []
    for idx, arg in enumerate(args):
        device = arg if not arg.isdigit() else f'/dev/video{arg}'
        cap = open_camera(device)
        cameras.append({'id': idx, 'device': device, 'cap': cap, 'open': cap.isOpened()})
        if not cap.isOpened():
            print(f'WARNING: could not open camera {device}')
else:
    detected = detect_cameras()
    if not detected:
        print('No capture devices found, falling back to indices 0,1,2')
        detected = [{'device': f'/dev/video{i}', 'cap': open_camera(i)} for i in range(3)]
    cameras = [
        {'id': idx, 'device': d['device'], 'cap': d['cap'], 'open': d['cap'].isOpened()}
        for idx, d in enumerate(detected)
    ]
    print(f'Using cameras: {[c["device"] for c in cameras]}')


def _reopen_camera(camera):
    old_cap = camera['cap']
    try:
        old_cap.release()
    except Exception:
        pass
    cap = open_camera(camera['device'])
    camera['cap'] = cap
    camera['open'] = cap.isOpened()
    return cap


def generate_frames(camera_id):
    camera = cameras[camera_id]
    device = camera['device']
    cap = camera['cap']
    frame_count = 0
    read_failures = 0
    while True:
        if not cap.isOpened():
            print(f'camera {camera_id} ({device}) reconnecting...')
            cap = _reopen_camera(camera)
            if not cap.isOpened():
                time.sleep(2)
                continue

        success, frame = cap.read()
        if not success or frame is None:
            read_failures += 1
            print(f'camera {camera_id} ({device}) read failed, retrying...')
            # Some V4L2 devices stay "opened" but stop delivering frames.
            # Force a reopen after repeated read failures.
            if read_failures >= 8:
                print(f'camera {camera_id} ({device}) forcing reopen after repeated read failures')
                cap = _reopen_camera(camera)
                read_failures = 0
            time.sleep(0.5)
            continue

        read_failures = 0
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue

        frame_count += 1
        if frame_count % 30 == 0:
            print(f'camera {camera_id} ({device}) sent {frame_count} frames')

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')


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


@app.route('/api/cameras')
def api_cameras():
    return jsonify([
        {
            'id': cam['id'],
            'device': cam['device'],
            'open': bool(cam['cap'] and cam['cap'].isOpened()),
        }
        for cam in cameras
    ])


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=CAM_PORT, debug=False)
