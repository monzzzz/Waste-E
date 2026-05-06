#!/usr/bin/env python3
"""
make_composite.py — Build a composite video from a Waste-E recording session.

Layout (1280×720):
  ┌────────┬────────┬────────┬────────┐
  │ cam 0  │ cam 1  │ cam 2  │ cam 3  │  ← OrangePi cameras
  ├────────┼────────┼────────┼────────┤
  │ cam 4  │ cam 5  │ cam 6  │ SENSOR │  ← RaspPi cameras + sensor panel
  ├────────┴────────┴────────┴────────┤
  │         TELEMETRY  STRIP          │  ← GPS / IMU / Encoders
  └───────────────────────────────────┘

Usage:
    python3 hardware/make_composite.py <session_dir> [output.mp4]
    python3 hardware/make_composite.py ~/...recordings/2026-05-01_17-38-13
"""

import argparse
import bisect
import csv
import json
import math
import pathlib
import subprocess
import sys
from typing import Optional

import cv2
import numpy as np

# ── Layout ──────────────────────────────────────────────────────────────────
CELL_W, CELL_H = 320, 240   # each camera/panel cell
COLS, ROWS = 4, 2
TBAR_H = 240                # telemetry strip height
OUT_W = CELL_W * COLS       # 1280
OUT_H = CELL_H * ROWS + TBAR_H  # 720

FPS = 30
FONT = cv2.FONT_HERSHEY_SIMPLEX

# Colors (BGR)
C_BG       = (13, 6, 3)
C_GRID     = (45, 45, 45)
C_LABEL    = (160, 200, 230)
C_VALUE    = (40, 210, 165)
C_DIM      = (70, 70, 70)
C_WARN     = (50, 100, 230)
C_OK       = (40, 190, 90)
C_NORTH    = (60, 60, 210)
C_BAR_FWD  = (30, 170, 110)
C_BAR_REV  = (170, 60, 60)


# ── Helpers ─────────────────────────────────────────────────────────────────

def _text(img, text: str, x: int, y: int, scale: float, color, thickness: int = 1):
    cv2.putText(img, text, (x, y), FONT, scale, (0, 0, 0), thickness + 1, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), FONT, scale, color, thickness, cv2.LINE_AA)


def letterbox(frame: np.ndarray, w: int, h: int) -> np.ndarray:
    fh, fw = frame.shape[:2]
    scale = min(w / fw, h / fh)
    nw, nh = int(fw * scale), int(fh * scale)
    resized = cv2.resize(frame, (nw, nh))
    out = np.zeros((h, w, 3), dtype=np.uint8)
    y0 = (h - nh) // 2
    x0 = (w - nw) // 2
    out[y0:y0 + nh, x0:x0 + nw] = resized
    return out


def _safe_float(v, default: float = 0.0) -> float:
    try:
        return float(v) if v not in (None, "", "None") else default
    except (ValueError, TypeError):
        return default


def _safe_bool(v) -> bool:
    return str(v).strip().lower() == "true"


# ── Telemetry ────────────────────────────────────────────────────────────────

def load_telemetry(path: pathlib.Path) -> tuple[list[float], list[dict]]:
    """Return (timestamps, rows) sorted by timestamp."""
    timestamps, rows = [], []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            try:
                ts = float(row["timestamp"])
            except (KeyError, ValueError):
                continue
            timestamps.append(ts)
            rows.append(row)
    return timestamps, rows


def telem_at(timestamps: list[float], rows: list[dict], abs_ts: float) -> dict:
    if not rows:
        return {}
    idx = bisect.bisect_right(timestamps, abs_ts)
    idx = max(0, min(idx, len(rows) - 1))
    return rows[idx]


# ── Drawing ──────────────────────────────────────────────────────────────────

def draw_compass(img, cx: int, cy: int, radius: int, heading: float):
    cv2.circle(img, (cx, cy), radius, C_GRID, 1)
    for label, angle in (("N", 0), ("E", 90), ("S", 180), ("W", 270)):
        rad = math.radians(angle - 90)
        lx = int(cx + (radius + 11) * math.cos(rad))
        ly = int(cy + (radius + 11) * math.sin(rad))
        col = C_NORTH if label == "N" else C_DIM
        _text(img, label, lx - 4, ly + 4, 0.28, col)

    rad = math.radians(heading - 90)
    tip_x = int(cx + radius * 0.82 * math.cos(rad))
    tip_y = int(cy + radius * 0.82 * math.sin(rad))
    tail_x = int(cx - radius * 0.4 * math.cos(rad))
    tail_y = int(cy - radius * 0.4 * math.sin(rad))
    cv2.line(img, (tail_x, tail_y), (tip_x, tip_y), C_VALUE, 2, cv2.LINE_AA)
    cv2.circle(img, (cx, cy), 3, C_VALUE, -1)
    _text(img, f"{heading:.0f}\xb0", cx - 14, cy + radius + 20, 0.32, C_VALUE)


def draw_rpm_bar(img, x: int, y: int, w: int, h: int, rpm: float, label: str):
    cv2.rectangle(img, (x, y), (x + w, y + h), (25, 25, 25), -1)
    cv2.rectangle(img, (x, y), (x + w, y + h), C_GRID, 1)
    frac = min(1.0, abs(rpm) / 120.0)
    fw = int(w * frac)
    if fw > 0:
        col = C_BAR_FWD if rpm >= 0 else C_BAR_REV
        cv2.rectangle(img, (x + 1, y + 1), (x + fw, y + h - 1), col, -1)
    _text(img, label, x - 18, y + h - 1, 0.30, C_LABEL)
    _text(img, f"{rpm:+.0f}", x + w + 4, y + h - 1, 0.30, C_VALUE)


def draw_sensor_panel(img, t: dict, x0: int, y0: int):
    """8th grid slot — compact sensor summary."""
    cv2.rectangle(img, (x0, y0), (x0 + CELL_W, y0 + CELL_H), (8, 14, 22), -1)
    cv2.rectangle(img, (x0, y0), (x0 + CELL_W, y0 + CELL_H), C_GRID, 1)

    heading = _safe_float(t.get("imu_heading"))
    roll    = _safe_float(t.get("imu_roll"))
    pitch   = _safe_float(t.get("imu_pitch"))
    l_rpm   = _safe_float(t.get("enc_left_rpm"))
    r_rpm   = _safe_float(t.get("enc_right_rpm"))
    l_dist  = _safe_float(t.get("enc_left_dist_m"))
    r_dist  = _safe_float(t.get("enc_right_dist_m"))
    online  = _safe_bool(t.get("motor_online"))

    _text(img, "SENSORS", x0 + 4, y0 + 14, 0.38, C_LABEL)

    # Compass (left half)
    draw_compass(img, x0 + 58, y0 + 85, 46, heading)

    # Roll / Pitch (right half)
    _text(img, f"Roll  {roll:+.1f}\xb0", x0 + 122, y0 + 50, 0.38, C_VALUE)
    _text(img, f"Pitch {pitch:+.1f}\xb0", x0 + 122, y0 + 70, 0.38, C_VALUE)

    # RPM bars
    draw_rpm_bar(img, x0 + 22, y0 + 148, 130, 10, l_rpm, "L")
    draw_rpm_bar(img, x0 + 22, y0 + 168, 130, 10, r_rpm, "R")

    # Distance
    _text(img, f"L {l_dist:.2f}m  R {r_dist:.2f}m", x0 + 6, y0 + 202, 0.32, C_LABEL)

    # Motor status
    mc = C_OK if online else C_WARN
    _text(img, "● MOTOR " + ("ONLINE" if online else "OFFLINE"), x0 + 6, y0 + 222, 0.33, mc)


def draw_telemetry_bar(img, t: dict, elapsed: float):
    """Full-width strip at the bottom of the canvas."""
    y0 = CELL_H * ROWS
    cv2.rectangle(img, (0, y0), (OUT_W, OUT_H), (6, 10, 18), -1)
    cv2.line(img, (0, y0), (OUT_W, y0), C_GRID, 1)

    # Column dividers
    for cx in (320, 640, 960):
        cv2.line(img, (cx, y0 + 20), (cx, OUT_H - 20), C_GRID, 1)

    # ── Col 0: Time + GPS ─────────────────────────────
    mins, secs = divmod(elapsed, 60)
    _text(img, f"{int(mins):02d}:{secs:05.2f}", 10, y0 + 28, 0.7, (210, 210, 210), 1)

    lat  = t.get("gps_lat") or ""
    lon  = t.get("gps_lon") or ""
    spd  = _safe_float(t.get("gps_speed"))
    alt  = _safe_float(t.get("gps_alt"))
    sats = t.get("gps_satellites") or "0"
    fix  = _safe_bool(t.get("gps_fix"))
    gps_col = C_VALUE if fix else C_DIM

    _text(img, "GPS", 10, y0 + 56, 0.38, C_LABEL)
    if lat and lon:
        _text(img, f"{_safe_float(lat):.5f}\xb0N", 10, y0 + 76, 0.40, gps_col)
        _text(img, f"{_safe_float(lon):.5f}\xb0E", 10, y0 + 96, 0.40, gps_col)
    else:
        _text(img, "No fix", 10, y0 + 76, 0.40, C_DIM)
    _text(img, f"Speed {spd:.1f} km/h", 10, y0 + 116, 0.34, gps_col)
    _text(img, f"Alt   {alt:.1f} m", 10, y0 + 134, 0.34, gps_col)
    _text(img, f"Sats  {sats}", 10, y0 + 152, 0.34, gps_col)

    # ── Col 1: IMU ────────────────────────────────────
    heading = _safe_float(t.get("imu_heading"))
    roll    = _safe_float(t.get("imu_roll"))
    pitch   = _safe_float(t.get("imu_pitch"))
    temp    = _safe_float(t.get("imu_temp"))
    cal_sys   = t.get("imu_cal_sys") or "0"
    cal_gyro  = t.get("imu_cal_gyro") or "0"
    cal_accel = t.get("imu_cal_accel") or "0"
    cal_mag   = t.get("imu_cal_mag") or "0"

    _text(img, "IMU", 330, y0 + 56, 0.38, C_LABEL)
    _text(img, f"Heading {heading:.1f}\xb0", 330, y0 + 76, 0.42, C_VALUE)
    _text(img, f"Roll    {roll:+.1f}\xb0", 330, y0 + 96, 0.40, C_VALUE)
    _text(img, f"Pitch   {pitch:+.1f}\xb0", 330, y0 + 116, 0.40, C_VALUE)
    _text(img, f"Temp    {temp:.1f}\xb0C", 330, y0 + 136, 0.38, C_LABEL)
    _text(img, f"Cal sys:{cal_sys} gyro:{cal_gyro}", 330, y0 + 156, 0.32, C_DIM)
    _text(img, f"    acc:{cal_accel}  mag:{cal_mag}", 330, y0 + 172, 0.32, C_DIM)

    # ── Col 2: Encoders ───────────────────────────────
    l_rpm   = _safe_float(t.get("enc_left_rpm"))
    r_rpm   = _safe_float(t.get("enc_right_rpm"))
    l_dist  = _safe_float(t.get("enc_left_dist_m"))
    r_dist  = _safe_float(t.get("enc_right_dist_m"))
    l_angle = _safe_float(t.get("enc_left_angle"))
    r_angle = _safe_float(t.get("enc_right_angle"))
    online  = _safe_bool(t.get("motor_online"))

    _text(img, "ENCODERS", 650, y0 + 56, 0.38, C_LABEL)
    _text(img, f"L RPM  {l_rpm:+7.1f}", 650, y0 + 76, 0.42, C_VALUE)
    _text(img, f"R RPM  {r_rpm:+7.1f}", 650, y0 + 96, 0.42, C_VALUE)
    _text(img, f"L dist {l_dist:.3f} m", 650, y0 + 116, 0.38, C_LABEL)
    _text(img, f"R dist {r_dist:.3f} m", 650, y0 + 134, 0.38, C_LABEL)
    _text(img, f"L ang  {l_angle:.1f}\xb0", 650, y0 + 152, 0.34, C_DIM)
    _text(img, f"R ang  {r_angle:.1f}\xb0", 650, y0 + 168, 0.34, C_DIM)

    # RPM bars (wide)
    draw_rpm_bar(img, 668, y0 + 190, 240, 11, l_rpm, "L")
    draw_rpm_bar(img, 668, y0 + 210, 240, 11, r_rpm, "R")

    # ── Col 3: Compass + Motor ────────────────────────
    mc = C_OK if online else C_WARN
    _text(img, "● MOTOR", 970, y0 + 56, 0.40, C_LABEL)
    _text(img, "ONLINE" if online else "OFFLINE", 970, y0 + 78, 0.50, mc)

    draw_compass(img, 1130, y0 + 145, 72, heading)


# ── Main ─────────────────────────────────────────────────────────────────────

def open_cap(videos_dir: pathlib.Path, cam_id: str) -> Optional[cv2.VideoCapture]:
    for ext in ("mp4", "webm"):
        p = videos_dir / f"{cam_id}.{ext}"
        if p.exists():
            cap = cv2.VideoCapture(str(p))
            if cap.isOpened():
                return cap
    return None


RECORDINGS_DIR = pathlib.Path(
    "~/Library/CloudStorage/GoogleDrive-elmond.pattanan@gmail.com/My Drive/Waste-E/recordings"
).expanduser()


def pick_session() -> pathlib.Path:
    """Interactively list recordings and let the user pick one."""
    if not RECORDINGS_DIR.exists():
        sys.exit(f"Recordings dir not found: {RECORDINGS_DIR}")

    sessions = sorted(
        [d for d in RECORDINGS_DIR.iterdir() if d.is_dir() and (d / "session.json").exists()],
        reverse=True,
    )
    if not sessions:
        sys.exit("No recordings found.")

    print("Available recordings:\n")
    for i, s in enumerate(sessions, 1):
        try:
            meta = json.loads((s / "session.json").read_text())
            dur = meta.get("duration_s", 0)
            n_cams = len(meta.get("cameras", []))
            n_rows = meta.get("telemetry_rows", 0)
            composite = " [composite exists]" if (s / "composite.mp4").exists() else ""
            mins, secs = divmod(dur, 60)
            print(f"  [{i}] {s.name}  {int(mins):02d}:{secs:04.1f}  {n_cams} cams  {n_rows} telem rows{composite}")
        except Exception:
            print(f"  [{i}] {s.name}")

    print()
    while True:
        raw = input("Pick a recording (number): ").strip()
        if raw.isdigit() and 1 <= int(raw) <= len(sessions):
            return sessions[int(raw) - 1]
        print(f"  Please enter a number between 1 and {len(sessions)}.")


def main():
    ap = argparse.ArgumentParser(description="Composite Waste-E recording into one video")
    ap.add_argument("session_dir", nargs="?", help="Path to session directory (omit to pick interactively)")
    ap.add_argument("output", nargs="?", help="Output path (default: <session>/composite.mp4)")
    ap.add_argument("--fps", type=int, default=FPS)
    args = ap.parse_args()

    if args.session_dir:
        session_dir = pathlib.Path(args.session_dir).expanduser().resolve()
        if not session_dir.exists():
            sys.exit(f"Session dir not found: {session_dir}")
    else:
        session_dir = pick_session()

    output = pathlib.Path(args.output).expanduser() if args.output else session_dir / "composite.mp4"

    # ── Session metadata ───────────────────────────────
    meta = json.loads((session_dir / "session.json").read_text())
    session_start: float = meta["started_at"]
    duration: float = meta.get("duration_s", 0.0)
    cam_ids: list[str] = [c["id"] for c in meta.get("cameras", [])]

    print(f"Session : {meta['id']}")
    print(f"Duration: {duration:.1f}s")
    print(f"Cameras : {', '.join(cam_ids)}")

    # ── Telemetry ──────────────────────────────────────
    telem_path = session_dir / "telemetry.csv"
    if telem_path.exists():
        ts_list, telem_rows = load_telemetry(telem_path)
        print(f"Telemetry: {len(telem_rows)} rows")
    else:
        ts_list, telem_rows = [], []
        print("Telemetry: none")

    # ── Open videos ────────────────────────────────────
    # cam_offset: how many seconds into the session this camera's video starts.
    # Computed as (session_duration - video_duration), assuming all cameras
    # stopped at the same moment. A camera with a short video started late.
    videos_dir = session_dir / "videos"
    caps: list[Optional[cv2.VideoCapture]] = []
    cam_offsets: list[float] = []   # seconds to skip at the start of each video
    labels: list[str] = []
    for cam_id in cam_ids:
        cap = open_cap(videos_dir, cam_id)
        caps.append(cap)
        labels.append(cam_id)
        if cap:
            src_fps = cap.get(cv2.CAP_PROP_FPS)
            src_dur = cap.get(cv2.CAP_PROP_FRAME_COUNT) / src_fps if src_fps else duration
            offset = max(0.0, duration - src_dur)
            cam_offsets.append(offset)
            print(f"  {cam_id}: ok  ({src_fps:.1f}fps, {src_dur:.1f}s, offset={offset:.3f}s)")
        else:
            cam_offsets.append(0.0)
            print(f"  {cam_id}: MISSING")

    # Pad to 7 slots (slot 8 = sensor panel)
    while len(caps) < 7:
        caps.append(None)
        cam_offsets.append(0.0)
        labels.append("")

    fps = args.fps
    total_frames = max(1, int(duration * fps))

    # ── ffmpeg pipe ────────────────────────────────────
    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo", "-vcodec", "rawvideo",
        "-s", f"{OUT_W}x{OUT_H}",
        "-pix_fmt", "bgr24",
        "-r", str(fps),
        "-i", "pipe:0",
        "-vcodec", "libx264", "-preset", "fast", "-crf", "22",
        "-pix_fmt", "yuv420p",
        str(output),
    ]
    print(f"\nOutput : {output}")
    print(f"Size   : {OUT_W}x{OUT_H} @ {fps}fps  ({total_frames} frames)")
    print("Encoding...")

    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
    blank = np.zeros((CELL_H, CELL_W, 3), dtype=np.uint8)

    for fi in range(total_frames):
        elapsed = fi / fps
        abs_ts  = session_start + elapsed
        t = telem_at(ts_list, telem_rows, abs_ts)

        canvas = np.full((OUT_H, OUT_W, 3), C_BG, dtype=np.uint8)

        # ── Camera grid (4×2) ──────────────────────────
        for slot in range(8):
            col = slot % COLS
            row = slot // COLS
            x0  = col * CELL_W
            y0  = row * CELL_H

            if slot == 7:
                draw_sensor_panel(canvas, t, x0, y0)
                continue

            cap = caps[slot] if slot < len(caps) else None
            if cap:
                seek_ms = (elapsed - cam_offsets[slot]) * 1000
                if seek_ms < 0:
                    ok, frame = False, None   # camera hadn't started yet
                else:
                    cap.set(cv2.CAP_PROP_POS_MSEC, seek_ms)
                    ok, frame = cap.read()
            else:
                ok, frame = False, None
            if ok and frame is not None:
                cell = letterbox(frame, CELL_W, CELL_H)
            else:
                cell = blank.copy()
                _text(cell, "NO SIGNAL", 85, 128, 0.50, (40, 40, 40))

            canvas[y0:y0 + CELL_H, x0:x0 + CELL_W] = cell

            # Camera label badge
            lbl = labels[slot] if slot < len(labels) else ""
            if lbl:
                (tw, _th), _ = cv2.getTextSize(lbl, FONT, 0.36, 1)
                bw = tw + 10
                cv2.rectangle(canvas, (x0, y0), (x0 + bw, y0 + 17), (0, 0, 0), -1)
                _text(canvas, lbl, x0 + 5, y0 + 13, 0.36, C_LABEL)

        # Grid lines
        for c in range(1, COLS):
            cv2.line(canvas, (c * CELL_W, 0), (c * CELL_W, CELL_H * ROWS), C_GRID, 1)
        cv2.line(canvas, (0, CELL_H), (OUT_W, CELL_H), C_GRID, 1)

        # Telemetry bar
        draw_telemetry_bar(canvas, t, elapsed)

        proc.stdin.write(canvas.tobytes())

        if fi % fps == 0 or fi == total_frames - 1:
            pct = 100 * (fi + 1) / total_frames
            print(f"\r  {pct:3.0f}%  ({fi + 1}/{total_frames})", end="", flush=True)

    proc.stdin.close()
    proc.wait()
    print(f"\nDone → {output}")

    for cap in caps:
        if cap:
            cap.release()


if __name__ == "__main__":
    main()
