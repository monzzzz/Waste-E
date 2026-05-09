#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import median
from typing import Any, Callable, Optional

DEFAULT_RECORDINGS_DIR = Path(__file__).resolve().parents[2] / "hardware" / "recordings"
DEFAULT_VIDEO_STEM = "orangepi-video0"
VIDEO_EXTS = (".mp4", ".mkv", ".mov", ".webm")
EARTH_RADIUS_M = 6_378_137.0


@dataclass(frozen=True)
class EvalConfig:
    horizon_seconds: float
    step_seconds: float
    steps: int
    history_seconds: float


@dataclass(frozen=True)
class TelemetrySample:
    index: int
    timestamp: float
    relative_time: float
    east_m: float
    north_m: float
    heading_deg: float
    speed_mps: float
    yaw_rate_dps: float
    gps_fix: bool
    motor_online: bool


@dataclass(frozen=True)
class Observation:
    sample: TelemetrySample
    history_local_xy: list[list[float]]
    frame_rgb: Optional[Any]
    frame_index: int
    video_path: Path
    session_dir: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replay a Waste-E recording session, run a lightweight offline "
            "trajectory policy, and score it against the robot's future path."
        )
    )
    parser.add_argument(
        "session",
        nargs="?",
        help=(
            "Session folder under hardware/recordings or a direct path. "
            "Defaults to the latest available session."
        ),
    )
    parser.add_argument(
        "--recordings-dir",
        type=Path,
        default=DEFAULT_RECORDINGS_DIR,
        help=f"Root recordings directory (default: {DEFAULT_RECORDINGS_DIR})",
    )
    parser.add_argument(
        "--video",
        default=DEFAULT_VIDEO_STEM,
        help=(
            "Video filename, stem, or absolute path to use as the primary view "
            f"(default: {DEFAULT_VIDEO_STEM})."
        ),
    )
    parser.add_argument(
        "--policy",
        choices=("stop", "constant_velocity", "oracle", "external"),
        default="constant_velocity",
        help="Built-in policy to evaluate (default: constant_velocity).",
    )
    parser.add_argument(
        "--policy-hook",
        help=(
            "External policy hook in the form path/to/module.py:function. "
            "Used when --policy external."
        ),
    )
    parser.add_argument(
        "--horizon-seconds",
        type=float,
        default=3.0,
        help="Prediction horizon in seconds (default: 3.0).",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=12,
        help="Number of predicted waypoints per sample (default: 12).",
    )
    parser.add_argument(
        "--history-seconds",
        type=float,
        default=1.5,
        help="Past context window exposed to policies (default: 1.5).",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Evaluate every Nth telemetry row (default: 1).",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Optional cap on evaluated samples for quick tests (default: 0 = all).",
    )
    parser.add_argument(
        "--render-video",
        action="store_true",
        help="Render an overlay video at telemetry rate.",
    )
    parser.add_argument(
        "--load-frames",
        action="store_true",
        help="Load RGB frames and pass them into the observation for external policies.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Override output directory. Default: <session>/offline_eval/<policy_name>",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing output directory.",
    )
    return parser.parse_args()


def resolve_session_dir(args: argparse.Namespace) -> Path:
    recordings_dir = args.recordings_dir.resolve()
    if args.session:
        candidate = Path(args.session)
        if not candidate.exists():
            candidate = recordings_dir / args.session
        candidate = candidate.resolve()
        if (candidate / "telemetry.csv").exists() and (candidate / "videos").is_dir():
            return candidate
        raise SystemExit(f"Session directory is missing telemetry.csv or videos/: {candidate}")

    sessions = sorted(
        path
        for path in recordings_dir.iterdir()
        if path.is_dir() and (path / "telemetry.csv").exists() and (path / "videos").is_dir()
    )
    if not sessions:
        raise SystemExit(f"No valid recording sessions found in {recordings_dir}")
    return sessions[-1]


def resolve_video_path(session_dir: Path, raw_video: str) -> Path:
    candidate = Path(raw_video)
    if candidate.is_absolute() and candidate.exists():
        return candidate

    videos_dir = session_dir / "videos"
    if candidate.exists():
        return candidate.resolve()

    direct = videos_dir / raw_video
    if direct.exists():
        return direct.resolve()

    stem_matches = sorted(
        path for path in videos_dir.iterdir()
        if path.is_file() and path.stem == raw_video
    )
    if stem_matches:
        return stem_matches[0].resolve()

    if raw_video == DEFAULT_VIDEO_STEM:
        preferred = sorted(
            path for path in videos_dir.iterdir()
            if path.is_file() and path.stem.startswith("orangepi-video0")
        )
        if preferred:
            return preferred[0].resolve()

    available = sorted(
        path for path in videos_dir.iterdir()
        if path.is_file() and path.suffix.lower() in VIDEO_EXTS
    )
    if not available:
        raise SystemExit(f"No video files found in {videos_dir}")
    raise SystemExit(
        f"Could not resolve video {raw_video!r} in {videos_dir}. "
        f"Available: {[path.name for path in available]}"
    )


def parse_float(raw: str | None) -> float:
    if raw is None:
        return float("nan")
    text = str(raw).strip()
    if not text:
        return float("nan")
    try:
        return float(text)
    except ValueError:
        return float("nan")


def parse_bool(raw: str | None) -> bool:
    if raw is None:
        return False
    return str(raw).strip().lower() in {"1", "true", "yes", "y"}


def is_finite(value: float) -> bool:
    return math.isfinite(value)


def load_telemetry_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="") as fh:
        reader = csv.DictReader(fh)
        rows = [dict(row) for row in reader]
    if not rows:
        raise SystemExit(f"Telemetry file is empty: {path}")
    return rows


def latlon_to_local_xy(latitudes: list[float], longitudes: list[float]) -> tuple[list[float], list[float]]:
    lat0 = math.radians(latitudes[0])
    lon0 = math.radians(longitudes[0])
    east: list[float] = []
    north: list[float] = []
    for lat_deg, lon_deg in zip(latitudes, longitudes):
        lat = math.radians(lat_deg)
        lon = math.radians(lon_deg)
        mean_lat = 0.5 * (lat + lat0)
        east.append((lon - lon0) * math.cos(mean_lat) * EARTH_RADIUS_M)
        north.append((lat - lat0) * EARTH_RADIUS_M)
    return east, north


def heading_from_displacement(east: list[float], north: list[float]) -> list[float]:
    heading: list[float] = []
    count = len(east)
    for idx in range(count):
        prev_idx = max(idx - 1, 0)
        next_idx = min(idx + 1, count - 1)
        dx = east[next_idx] - east[prev_idx]
        dy = north[next_idx] - north[prev_idx]
        if abs(dx) < 1e-6 and abs(dy) < 1e-6:
            heading.append(heading[-1] if heading else 0.0)
            continue
        angle_math = math.degrees(math.atan2(dy, dx))
        heading.append((90.0 - angle_math) % 360.0)
    return heading


def shortest_heading_delta_deg(current_deg: float, prev_deg: float) -> float:
    return ((current_deg - prev_deg + 180.0) % 360.0) - 180.0


def speed_from_displacement(times: list[float], east: list[float], north: list[float]) -> list[float]:
    speed: list[float] = []
    count = len(times)
    for idx in range(count):
        prev_idx = max(idx - 1, 0)
        next_idx = min(idx + 1, count - 1)
        dt = times[next_idx] - times[prev_idx]
        if dt <= 1e-6:
            speed.append(speed[-1] if speed else 0.0)
            continue
        dx = east[next_idx] - east[prev_idx]
        dy = north[next_idx] - north[prev_idx]
        speed.append(math.hypot(dx, dy) / dt)
    return speed


def build_samples(rows: list[dict[str, str]]) -> list[TelemetrySample]:
    filtered_rows: list[tuple[int, dict[str, str], float, float, float]] = []
    for idx, row in enumerate(rows):
        timestamp = parse_float(row.get("timestamp"))
        lat = parse_float(row.get("gps_lat"))
        lon = parse_float(row.get("gps_lon"))
        if is_finite(timestamp) and is_finite(lat) and is_finite(lon):
            filtered_rows.append((idx, row, timestamp, lat, lon))

    if not filtered_rows:
        raise SystemExit("Telemetry does not contain usable timestamp/GPS rows.")

    timestamps = [item[2] for item in filtered_rows]
    latitudes = [item[3] for item in filtered_rows]
    longitudes = [item[4] for item in filtered_rows]
    rel_times = [timestamp - timestamps[0] for timestamp in timestamps]
    east_m, north_m = latlon_to_local_xy(latitudes, longitudes)
    derived_heading = heading_from_displacement(east_m, north_m)
    derived_speed = speed_from_displacement(rel_times, east_m, north_m)

    samples: list[TelemetrySample] = []
    prev_heading = derived_heading[0]
    prev_yaw_rate = 0.0
    for local_idx, (raw_idx, row, timestamp, _, _) in enumerate(filtered_rows):
        imu_heading = parse_float(row.get("imu_heading"))
        gps_heading = parse_float(row.get("gps_heading"))
        gps_speed = parse_float(row.get("gps_speed"))

        heading = imu_heading if is_finite(imu_heading) else gps_heading if is_finite(gps_heading) else derived_heading[local_idx]
        speed = gps_speed if is_finite(gps_speed) else derived_speed[local_idx]
        dt = rel_times[local_idx] - rel_times[local_idx - 1] if local_idx > 0 else 0.0
        if local_idx == 0 or dt <= 1e-6:
            yaw_rate = prev_yaw_rate
        else:
            yaw_rate = shortest_heading_delta_deg(heading, prev_heading) / dt

        sample = TelemetrySample(
            index=raw_idx,
            timestamp=timestamp,
            relative_time=rel_times[local_idx],
            east_m=east_m[local_idx],
            north_m=north_m[local_idx],
            heading_deg=heading % 360.0,
            speed_mps=max(0.0, speed),
            yaw_rate_dps=yaw_rate,
            gps_fix=parse_bool(row.get("gps_fix")),
            motor_online=parse_bool(row.get("motor_online")),
        )
        samples.append(sample)
        prev_heading = sample.heading_deg
        prev_yaw_rate = yaw_rate
    return samples


def world_to_local(points_en: list[list[float]], origin_en: tuple[float, float], heading_deg: float) -> list[list[float]]:
    theta = math.radians(heading_deg)
    forward_x = math.sin(theta)
    forward_y = math.cos(theta)
    left_x = -math.cos(theta)
    left_y = math.sin(theta)
    origin_east, origin_north = origin_en

    local_points: list[list[float]] = []
    for east_m, north_m in points_en:
        dx = east_m - origin_east
        dy = north_m - origin_north
        x_forward = dx * forward_x + dy * forward_y
        y_left = dx * left_x + dy * left_y
        local_points.append([x_forward, y_left])
    return local_points


def interpolate_series(query_t: float, times: list[float], values: list[float]) -> float:
    if query_t <= times[0]:
        return values[0]
    if query_t >= times[-1]:
        return values[-1]

    lo = 0
    hi = len(times) - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if times[mid] <= query_t:
            lo = mid
        else:
            hi = mid

    t0 = times[lo]
    t1 = times[hi]
    v0 = values[lo]
    v1 = values[hi]
    if t1 <= t0:
        return v0
    alpha = (query_t - t0) / (t1 - t0)
    return v0 + alpha * (v1 - v0)


def get_ground_truth_future(
    sample: TelemetrySample,
    config: EvalConfig,
    all_times: list[float],
    east: list[float],
    north: list[float],
) -> Optional[list[list[float]]]:
    future_times = [
        sample.relative_time + (step_idx + 1) * config.step_seconds
        for step_idx in range(config.steps)
    ]
    if future_times[-1] > all_times[-1]:
        return None

    future_world = [
        [
            interpolate_series(t, all_times, east),
            interpolate_series(t, all_times, north),
        ]
        for t in future_times
    ]
    return world_to_local(future_world, (sample.east_m, sample.north_m), sample.heading_deg)


def get_history_local(sample_idx: int, samples: list[TelemetrySample], history_seconds: float) -> list[list[float]]:
    current = samples[sample_idx]
    start_time = current.relative_time - history_seconds
    history_points = [
        [sample.east_m, sample.north_m]
        for sample in samples
        if start_time <= sample.relative_time <= current.relative_time
    ]
    if not history_points:
        return []
    return world_to_local(history_points, (current.east_m, current.north_m), current.heading_deg)


def resample_prediction(points: Any, steps: int) -> list[list[float]]:
    if points is None:
        return [[0.0, 0.0] for _ in range(steps)]

    flattened: list[list[float]] = []
    for item in list(points):
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            flattened.append([float(item[0]), float(item[1])])

    if not flattened:
        return [[0.0, 0.0] for _ in range(steps)]
    if len(flattened) == steps:
        return flattened
    if len(flattened) == 1:
        return [flattened[0][:] for _ in range(steps)]

    result: list[list[float]] = []
    src_positions = [idx / (len(flattened) - 1) for idx in range(len(flattened))]
    for dst_idx in range(steps):
        dst_pos = dst_idx / (steps - 1) if steps > 1 else 0.0
        lo = 0
        hi = len(src_positions) - 1
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if src_positions[mid] <= dst_pos:
                lo = mid
            else:
                hi = mid
        p0 = flattened[lo]
        p1 = flattened[hi]
        s0 = src_positions[lo]
        s1 = src_positions[hi]
        alpha = 0.0 if s1 <= s0 else (dst_pos - s0) / (s1 - s0)
        result.append([
            p0[0] + alpha * (p1[0] - p0[0]),
            p0[1] + alpha * (p1[1] - p0[1]),
        ])
    return result


def round_points(points: list[list[float]], digits: int = 6) -> list[list[float]]:
    return [[round(point[0], digits), round(point[1], digits)] for point in points]


def point_errors(predicted: list[list[float]], ground_truth: list[list[float]]) -> list[float]:
    errors: list[float] = []
    for pred, truth in zip(predicted, ground_truth):
        errors.append(math.hypot(pred[0] - truth[0], pred[1] - truth[1]))
    return errors


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def predict_stop(_: Observation, config: EvalConfig) -> list[list[float]]:
    return [[0.0, 0.0] for _ in range(config.steps)]


def predict_oracle(_: Observation, __: EvalConfig, gt_future_local: list[list[float]]) -> list[list[float]]:
    return [point[:] for point in gt_future_local]


def predict_constant_velocity(obs: Observation, config: EvalConfig) -> list[list[float]]:
    x = 0.0
    y = 0.0
    yaw = 0.0
    speed = max(0.0, float(obs.sample.speed_mps))
    yaw_rate_rad = -math.radians(float(obs.sample.yaw_rate_dps))
    points: list[list[float]] = []
    for _ in range(config.steps):
        yaw += yaw_rate_rad * config.step_seconds
        x += speed * math.cos(yaw) * config.step_seconds
        y += speed * math.sin(yaw) * config.step_seconds
        points.append([x, y])
    return points


def load_external_policy(raw_hook: str) -> Callable[[Observation, EvalConfig], Any]:
    if ":" not in raw_hook:
        raise SystemExit("External policy hook must be in the form path/to/module.py:function_name")
    module_path_raw, func_name = raw_hook.split(":", 1)
    module_path = Path(module_path_raw).resolve()
    if not module_path.exists():
        raise SystemExit(f"External policy module not found: {module_path}")

    spec = importlib.util.spec_from_file_location("manual_sim_external_policy", module_path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"Could not load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    func = getattr(module, func_name, None)
    if func is None or not callable(func):
        raise SystemExit(f"Function {func_name!r} not found in {module_path}")
    return func


def resolve_policy(args: argparse.Namespace) -> tuple[str, Callable[[Observation, EvalConfig], Any], bool]:
    if args.policy == "external":
        if not args.policy_hook:
            raise SystemExit("--policy external requires --policy-hook")
        return ("external", load_external_policy(args.policy_hook), False)
    if args.policy_hook:
        raise SystemExit("--policy-hook is only valid with --policy external")
    if args.policy == "stop":
        return ("stop", predict_stop, False)
    if args.policy == "constant_velocity":
        return ("constant_velocity", predict_constant_velocity, False)
    if args.policy == "oracle":
        return ("oracle", predict_stop, False)
    raise SystemExit(f"Unsupported policy: {args.policy}")


def probe_video(path: Path) -> tuple[float, float, int, int]:
    if shutil.which("ffprobe"):
        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=r_frame_rate,width,height",
            "-show_entries",
            "format=duration",
            "-of",
            "json",
            str(path),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if proc.returncode == 0:
            payload = json.loads(proc.stdout or "{}")
            streams = payload.get("streams") or [{}]
            fmt = payload.get("format") or {}
            stream = streams[0]
            rate = str(stream.get("r_frame_rate") or "30/1")
            num, den = rate.split("/", 1)
            fps = float(num) / float(den) if float(den) != 0.0 else 30.0
            duration = float(fmt.get("duration") or 0.0)
            width = int(stream.get("width") or 0)
            height = int(stream.get("height") or 0)
            if fps > 0.0 and duration > 0.0 and width > 0 and height > 0:
                return fps, duration, width, height

    try:
        import cv2  # type: ignore
    except ImportError as exc:
        raise SystemExit(
            "Need either ffprobe on PATH or opencv-python installed to inspect video metadata."
        ) from exc

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise SystemExit(f"Failed to open video: {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0) or 30.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    duration = frame_count / fps if frame_count > 0 and fps > 0.0 else 0.0
    cap.release()
    return fps, duration, width, height


class SampledVideoReader:
    def __init__(self, video_path: Path):
        try:
            import cv2  # type: ignore
        except ImportError as exc:
            raise SystemExit("opencv-python is required for frame sampling/rendering.") from exc

        self.cv2 = cv2
        self.cap = cv2.VideoCapture(str(video_path))
        if not self.cap.isOpened():
            raise SystemExit(f"Failed to open video for sampling: {video_path}")
        self.current_index = -1
        self.current_frame: Optional[Any] = None

    def get(self, target_index: int) -> Any:
        if target_index < 0:
            target_index = 0
        if self.current_index > target_index:
            self.cap.set(self.cv2.CAP_PROP_POS_FRAMES, target_index)
            self.current_index = target_index - 1

        while self.current_index < target_index:
            ok, frame = self.cap.read()
            if not ok:
                if self.current_frame is None:
                    raise RuntimeError("Reached end of video before requested frame.")
                return self.current_frame.copy()
            self.current_index += 1
            self.current_frame = frame

        if self.current_frame is None:
            raise RuntimeError("Video returned no frames.")
        return self.current_frame.copy()

    def close(self) -> None:
        self.cap.release()


def draw_bev_inset(
    pred_local_xy: list[list[float]],
    gt_local_xy: list[list[float]],
    history_local_xy: list[list[float]],
) -> Any:
    try:
        import cv2  # type: ignore
        import numpy as np  # type: ignore
    except ImportError as exc:
        raise RuntimeError("opencv-python and numpy are required for rendering overlays.") from exc

    inset_size = 240
    inset = np.full((inset_size, inset_size, 3), 18, dtype=np.uint8)
    origin_x = inset_size // 2
    origin_y = inset_size - 24
    scale = (inset_size - 36) / 12.0

    for meter in range(-6, 7, 2):
        x = int(origin_x - meter * scale)
        cv2.line(inset, (x, 16), (x, inset_size - 16), (45, 45, 45), 1, cv2.LINE_AA)
    for meter in range(0, 7, 2):
        y = int(origin_y - meter * scale)
        cv2.line(inset, (16, y), (inset_size - 16, y), (45, 45, 45), 1, cv2.LINE_AA)

    def project(points: list[list[float]]) -> Any:
        if not points:
            return np.zeros((0, 2), dtype=np.int32)
        projected = []
        for forward_m, left_m in points:
            px = int(origin_x - left_m * scale)
            py = int(origin_y - forward_m * scale)
            projected.append([px, py])
        return np.asarray(projected, dtype=np.int32)

    hist_pts = project(history_local_xy)
    gt_pts = project(gt_local_xy)
    pred_pts = project(pred_local_xy)

    if len(hist_pts) >= 2:
        cv2.polylines(inset, [hist_pts], False, (120, 120, 120), 2, cv2.LINE_AA)
    if len(gt_pts) >= 2:
        cv2.polylines(inset, [gt_pts], False, (0, 200, 0), 2, cv2.LINE_AA)
    if len(pred_pts) >= 2:
        cv2.polylines(inset, [pred_pts], False, (255, 220, 0), 2, cv2.LINE_AA)

    robot = np.asarray(
        [
            [origin_x, origin_y - 14],
            [origin_x - 8, origin_y + 8],
            [origin_x + 8, origin_y + 8],
        ],
        dtype=np.int32,
    )
    cv2.fillConvexPoly(inset, robot, (0, 255, 255))
    cv2.putText(inset, "GT", (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1, cv2.LINE_AA)
    cv2.putText(inset, "Pred", (54, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 220, 0), 1, cv2.LINE_AA)
    return inset


def render_overlay_frame(
    frame_bgr: Any,
    sample: TelemetrySample,
    pred_local_xy: list[list[float]],
    gt_local_xy: list[list[float]],
    history_local_xy: list[list[float]],
    ade_m: float,
    fde_m: float,
    policy_name: str,
) -> Any:
    try:
        import cv2  # type: ignore
    except ImportError as exc:
        raise RuntimeError("opencv-python is required for rendering overlays.") from exc

    output = frame_bgr.copy()
    overlay = output.copy()
    cv2.rectangle(overlay, (12, 12), (390, 122), (0, 0, 0), -1)
    output = cv2.addWeighted(overlay, 0.45, output, 0.55, 0.0)

    lines = [
        f"policy: {policy_name}",
        f"t={sample.relative_time:6.2f}s  speed={sample.speed_mps:4.2f} m/s",
        f"heading={sample.heading_deg:6.1f} deg  yaw_rate={sample.yaw_rate_dps:5.2f} deg/s",
        f"ADE={ade_m:5.2f} m  FDE={fde_m:5.2f} m",
        f"gps_fix={sample.gps_fix}  motor_online={sample.motor_online}",
    ]
    for idx, line in enumerate(lines):
        cv2.putText(
            output,
            line,
            (24, 36 + idx * 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (235, 235, 235),
            1,
            cv2.LINE_AA,
        )

    inset = draw_bev_inset(pred_local_xy, gt_local_xy, history_local_xy)
    height, width = output.shape[:2]
    inset_h, inset_w = inset.shape[:2]
    y0 = max(12, height - inset_h - 12)
    x0 = max(12, width - inset_w - 12)
    output[y0:y0 + inset_h, x0:x0 + inset_w] = inset
    return output


def ensure_output_dir(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise SystemExit(f"Output directory already exists: {path}. Pass --overwrite to reuse it.")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def main() -> int:
    args = parse_args()
    if args.steps <= 0:
        raise SystemExit("--steps must be positive")
    if args.horizon_seconds <= 0.0:
        raise SystemExit("--horizon-seconds must be positive")
    if args.history_seconds < 0.0:
        raise SystemExit("--history-seconds must be non-negative")
    if args.stride <= 0:
        raise SystemExit("--stride must be positive")

    session_dir = resolve_session_dir(args)
    telemetry_path = session_dir / "telemetry.csv"
    session_json_path = session_dir / "session.json"
    video_path = resolve_video_path(session_dir, args.video)

    rows = load_telemetry_rows(telemetry_path)
    samples = build_samples(rows)

    fps, duration_s, video_w, video_h = probe_video(video_path)
    usable_samples = [sample for sample in samples if sample.relative_time <= duration_s + 1e-6]
    if not usable_samples:
        raise SystemExit(f"No telemetry samples overlap the video duration of {duration_s:.2f}s.")

    sample_times = [sample.relative_time for sample in usable_samples]
    sample_dt = median(
        [sample_times[idx] - sample_times[idx - 1] for idx in range(1, len(sample_times))]
    ) if len(sample_times) > 1 else 0.1

    config = EvalConfig(
        horizon_seconds=float(args.horizon_seconds),
        step_seconds=float(args.horizon_seconds) / float(args.steps),
        steps=int(args.steps),
        history_seconds=float(args.history_seconds),
    )

    policy_name, policy_fn, needs_frames = resolve_policy(args)
    output_dir = args.output_dir.resolve() if args.output_dir else (session_dir / "offline_eval" / policy_name)
    ensure_output_dir(output_dir, args.overwrite)

    all_times = [sample.relative_time for sample in usable_samples]
    east = [sample.east_m for sample in usable_samples]
    north = [sample.north_m for sample in usable_samples]

    target_indices = list(range(0, len(usable_samples), args.stride))
    if args.max_samples > 0:
        target_indices = target_indices[:args.max_samples]
    if not target_indices:
        raise SystemExit("No samples selected for evaluation.")

    sampled_reader: Optional[SampledVideoReader] = None
    video_writer = None
    cv2 = None
    if args.render_video or args.load_frames or needs_frames:
        sampled_reader = SampledVideoReader(video_path)
    if args.render_video:
        try:
            import cv2 as _cv2  # type: ignore
        except ImportError as exc:
            raise SystemExit("--render-video requires opencv-python.") from exc
        cv2 = _cv2
        overlay_path = output_dir / "overlay.mp4"
        writer_fps = max(1.0, 1.0 / max(sample_dt * args.stride, 1e-6))
        video_writer = cv2.VideoWriter(
            str(overlay_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            writer_fps,
            (video_w, video_h),
        )
        if not video_writer.isOpened():
            raise SystemExit(f"Failed to open overlay writer: {overlay_path}")

    per_frame_rows: list[dict[str, float | int | str | bool]] = []
    predictions_path = output_dir / "predictions.jsonl"
    metrics_path = output_dir / "per_frame_metrics.csv"
    summary_path = output_dir / "summary.json"

    with predictions_path.open("w", encoding="utf-8") as pred_fh:
        evaluated = 0
        for idx in target_indices:
            sample = usable_samples[idx]
            gt_future_local = get_ground_truth_future(sample, config, all_times, east, north)
            if gt_future_local is None:
                continue

            frame_rgb: Optional[Any] = None
            frame_bgr: Optional[Any] = None
            frame_index = min(int(round(sample.relative_time * fps)), max(0, int(round(duration_s * fps)) - 1))
            if sampled_reader is not None:
                frame_bgr = sampled_reader.get(frame_index)
                frame_rgb = frame_bgr[:, :, ::-1].copy()

            obs = Observation(
                sample=sample,
                history_local_xy=get_history_local(idx, usable_samples, config.history_seconds),
                frame_rgb=frame_rgb,
                frame_index=frame_index,
                video_path=video_path,
                session_dir=session_dir,
            )

            if args.policy == "oracle":
                pred_local_xy = predict_oracle(obs, config, gt_future_local)
            else:
                pred_local_xy = resample_prediction(policy_fn(obs, config), config.steps)

            errors = point_errors(pred_local_xy, gt_future_local)
            ade_m = mean(errors)
            fde_m = errors[-1]
            evaluated += 1

            metrics_row: dict[str, float | int | str | bool] = {
                "sample_index": evaluated - 1,
                "telemetry_index": sample.index,
                "relative_time_s": round(sample.relative_time, 6),
                "frame_index": frame_index,
                "speed_mps": round(sample.speed_mps, 6),
                "heading_deg": round(sample.heading_deg, 6),
                "yaw_rate_dps": round(sample.yaw_rate_dps, 6),
                "ade_m": round(ade_m, 6),
                "fde_m": round(fde_m, 6),
                "gps_fix": sample.gps_fix,
                "motor_online": sample.motor_online,
            }
            per_frame_rows.append(metrics_row)

            pred_record = {
                "sample_index": evaluated - 1,
                "telemetry_index": sample.index,
                "timestamp": sample.timestamp,
                "relative_time_s": sample.relative_time,
                "frame_index": frame_index,
                "pred_local_xy_m": round_points(pred_local_xy),
                "gt_local_xy_m": round_points(gt_future_local),
                "ade_m": ade_m,
                "fde_m": fde_m,
            }
            pred_fh.write(json.dumps(pred_record) + "\n")

            if video_writer is not None and frame_bgr is not None and cv2 is not None:
                render = render_overlay_frame(
                    frame_bgr=frame_bgr,
                    sample=sample,
                    pred_local_xy=pred_local_xy,
                    gt_local_xy=gt_future_local,
                    history_local_xy=obs.history_local_xy,
                    ade_m=ade_m,
                    fde_m=fde_m,
                    policy_name=policy_name,
                )
                video_writer.write(render)

    if sampled_reader is not None:
        sampled_reader.close()
    if video_writer is not None:
        video_writer.release()

    if not per_frame_rows:
        raise SystemExit("No evaluable samples remained after applying horizon and video overlap.")

    with metrics_path.open("w", newline="", encoding="utf-8") as metrics_fh:
        fieldnames = list(per_frame_rows[0].keys())
        writer = csv.DictWriter(metrics_fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(per_frame_rows)

    session_metadata = {}
    if session_json_path.exists():
        try:
            session_metadata = json.loads(session_json_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            session_metadata = {}

    ade_values = [float(row["ade_m"]) for row in per_frame_rows]
    fde_values = [float(row["fde_m"]) for row in per_frame_rows]
    summary = {
        "session_dir": str(session_dir),
        "video_path": str(video_path),
        "policy": policy_name,
        "policy_hook": args.policy_hook,
        "config": asdict(config),
        "telemetry_rows_total": len(rows),
        "telemetry_rows_usable": len(usable_samples),
        "evaluated_samples": len(per_frame_rows),
        "video_fps": fps,
        "video_duration_s": duration_s,
        "video_size": {"width": video_w, "height": video_h},
        "stride": args.stride,
        "sample_dt_s": sample_dt,
        "mean_ade_m": mean(ade_values),
        "median_ade_m": median(ade_values),
        "mean_fde_m": mean(fde_values),
        "median_fde_m": median(fde_values),
        "session_metadata": {
            "id": session_metadata.get("id"),
            "started_at_iso": session_metadata.get("started_at_iso"),
            "ended_at_iso": session_metadata.get("ended_at_iso"),
            "duration_s": session_metadata.get("duration_s"),
        },
        "artifacts": {
            "summary_json": str(summary_path),
            "per_frame_metrics_csv": str(metrics_path),
            "predictions_jsonl": str(predictions_path),
            "overlay_mp4": str(output_dir / "overlay.mp4") if args.render_video else None,
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"[offline_eval] session      : {session_dir}")
    print(f"[offline_eval] video        : {video_path.name} ({duration_s:.2f}s @ {fps:.2f} fps)")
    print(f"[offline_eval] policy       : {policy_name}")
    print(f"[offline_eval] samples      : {len(per_frame_rows)}")
    print(f"[offline_eval] mean ADE/FDE : {summary['mean_ade_m']:.3f} m / {summary['mean_fde_m']:.3f} m")
    print(f"[offline_eval] output       : {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
