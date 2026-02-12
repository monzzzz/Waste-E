from flask import Flask, Response, request
import cv2
import numpy as np
import threading
import time

app = Flask(__name__)

latest_jpg = None
lock = threading.Lock()

@app.route("/upload", methods=["POST"])
def upload():
    global latest_jpg
    # Orange Pi sends raw JPEG bytes
    data = request.data

    if not data:
        return "No data", 400
    
    print(len(data), "bytes received")
    
    with open("latest.jpg", "wb") as f:
        f.write(data)

    with lock:
        latest_jpg = data

    return "OK", 200

def mjpeg_stream():
    global latest_jpg
    while True:
        with lock:
            jpg = latest_jpg
        if jpg is None:
            time.sleep(0.01)
            continue

        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n"
               b"Content-Length: " + str(len(jpg)).encode() + b"\r\n\r\n" +
               jpg + b"\r\n")

@app.route("/stream")
def stream():
    return Response(mjpeg_stream(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/")
def index():
    return """
    <h2>Orange Pi 2-Camera Stream</h2>
    <img src="/stream" />
    """

if __name__ == "__main__":
    print("🚀 Laptop server running at: http://0.0.0.0:5000")
    app.run(host="0.0.0.0", port=5000, threaded=True)