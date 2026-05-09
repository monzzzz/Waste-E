#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

VIDEO_EXTS = {".mp4", ".mkv", ".mov", ".webm"}
CAMERA_RE = re.compile(r"^(orangepi|rasppi)-(?:video)?(\d+)$")
DEFAULT_RECORDINGS_DIR = Path(__file__).resolve().parents[1] / "recordings"


@dataclass(frozen=True)
class VideoInput:
    device: str
    camera_id: int
    path: Path
    width: int
    height: int
    fps: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Combine per-camera Waste-E recordings into one OrangePi composite "
            "and one RasPi composite."
        )
    )
    parser.add_argument(
        "session",
        nargs="*",
        help=(
            "Session folder name(s) under hardware/recordings, or absolute/relative "
            "paths to session directories. Defaults to the latest session."
        ),
    )
    parser.add_argument(
        "--recordings-dir",
        type=Path,
        default=DEFAULT_RECORDINGS_DIR,
        help=f"Root recordings directory (default: {DEFAULT_RECORDINGS_DIR})",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Process every session under the recordings directory.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing composite outputs.",
    )
    parser.add_argument(
        "--preset",
        default="veryfast",
        help="ffmpeg x264 preset for the combined output (default: veryfast).",
    )
    parser.add_argument(
        "--crf",
        type=int,
        default=20,
        help="ffmpeg x264 CRF for the combined output (default: 20).",
    )
    return parser.parse_args()


def require_binary(name: str) -> None:
    if shutil.which(name) is None:
        raise SystemExit(f"Required binary not found on PATH: {name}")


def list_sessions(recordings_dir: Path) -> list[Path]:
    if not recordings_dir.exists():
        return []
    return sorted(
        path
        for path in recordings_dir.iterdir()
        if path.is_dir() and (path / "videos").is_dir()
    )


def resolve_sessions(args: argparse.Namespace) -> list[Path]:
    recordings_dir = args.recordings_dir.resolve()
    sessions: list[Path] = []

    if args.all:
        sessions = list_sessions(recordings_dir)
    elif args.session:
        for raw in args.session:
            candidate = Path(raw)
            if not candidate.exists():
                candidate = recordings_dir / raw
            candidate = candidate.resolve()
            if not (candidate / "videos").is_dir():
                raise SystemExit(f"Session directory missing videos/: {candidate}")
            sessions.append(candidate)
    else:
        all_sessions = list_sessions(recordings_dir)
        if not all_sessions:
            raise SystemExit(f"No recording sessions found in {recordings_dir}")
        sessions = [all_sessions[-1]]

    if not sessions:
        raise SystemExit(f"No recording sessions found in {recordings_dir}")
    return sessions


def probe_video(path: Path) -> tuple[int, int, float]:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,r_frame_rate",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed for {path}: {result.stderr.strip() or result.stdout.strip()}")

    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    try:
        width = int(lines[0])
        height = int(lines[1])
        fps_raw = lines[2] if len(lines) > 2 else "30/1"
        fps = float(Fraction(fps_raw)) if fps_raw != "0/0" else 30.0
        return width, height, fps
    except Exception as exc:
        raise RuntimeError(f"Unexpected ffprobe output for {path}: {lines!r}") from exc


def collect_device_videos(session_dir: Path, device: str) -> list[VideoInput]:
    videos_dir = session_dir / "videos"
    found: list[VideoInput] = []

    for path in sorted(videos_dir.iterdir()):
        if not path.is_file() or path.suffix.lower() not in VIDEO_EXTS:
            continue

        match = CAMERA_RE.match(path.stem)
        if not match or match.group(1) != device:
            continue

        width, height, fps = probe_video(path)
        found.append(
            VideoInput(
                device=device,
                camera_id=int(match.group(2)),
                path=path,
                width=width,
                height=height,
                fps=fps,
            )
        )

    return sorted(found, key=lambda item: item.camera_id)


def grid_shape(video_count: int) -> tuple[int, int]:
    if video_count <= 1:
        return 1, 1
    if video_count <= 2:
        return 2, 1
    return 2, 2


def build_filter(video_count: int, cols: int, rows: int, tile_w: int, tile_h: int) -> str:
    total_tiles = cols * rows
    parts: list[str] = []

    for index in range(total_tiles):
        if index < video_count:
            parts.append(
                f"[{index}:v]"
                f"scale={tile_w}:{tile_h}:force_original_aspect_ratio=decrease,"
                f"pad={tile_w}:{tile_h}:(ow-iw)/2:(oh-ih)/2:black,"
                f"setsar=1,setpts=PTS-STARTPTS[v{index}]"
            )
        else:
            parts.append(
                f"[{index}:v]"
                f"format=yuv420p,setsar=1,setpts=PTS-STARTPTS[v{index}]"
            )

    layout_items: list[str] = []
    for index in range(total_tiles):
        col = index % cols
        row = index // cols
        layout_items.append(f"{col * tile_w}_{row * tile_h}")

    joined_inputs = "".join(f"[v{index}]" for index in range(total_tiles))
    parts.append(
        f"{joined_inputs}"
        f"xstack=inputs={total_tiles}:layout={'|'.join(layout_items)}:fill=black:shortest=1[vout]"
    )
    return ";".join(parts)


def combine_device_videos(
    session_dir: Path,
    device: str,
    videos: list[VideoInput],
    *,
    overwrite: bool,
    preset: str,
    crf: int,
) -> Path | None:
    if not videos:
        return None

    cols, rows = grid_shape(len(videos))
    tile_w = max(video.width for video in videos)
    tile_h = max(video.height for video in videos)
    output_fps = max(video.fps for video in videos)
    total_tiles = cols * rows

    composites_dir = session_dir / "composites"
    composites_dir.mkdir(exist_ok=True)
    output_path = composites_dir / f"{device}_combined.mp4"

    if output_path.exists() and not overwrite:
        print(f"[skip] {output_path} already exists")
        return output_path

    filter_complex = build_filter(len(videos), cols, rows, tile_w, tile_h)
    cmd = ["ffmpeg", "-y" if overwrite or output_path.exists() else "-n"]

    for video in videos:
        cmd.extend(["-i", str(video.path)])

    for _ in range(total_tiles - len(videos)):
        cmd.extend(["-f", "lavfi", "-i", f"color=c=black:s={tile_w}x{tile_h}:r=30"])

    cmd.extend(
        [
            "-filter_complex",
            filter_complex,
            "-map",
            "[vout]",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            preset,
            "-crf",
            str(crf),
            "-pix_fmt",
            "yuv420p",
            "-r",
            format_fps(output_fps),
            "-movflags",
            "+faststart",
            str(output_path),
        ]
    )

    print(f"[build] {device}: {len(videos)} input videos -> {output_path}")
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed while building {output_path}")
    return output_path


def format_fps(value: float) -> str:
    rounded = round(value)
    if abs(value - rounded) < 0.01:
        return str(int(rounded))
    return f"{value:.3f}".rstrip("0").rstrip(".")


def process_session(
    session_dir: Path,
    *,
    overwrite: bool,
    preset: str,
    crf: int,
) -> int:
    print(f"\n[session] {session_dir}")
    built = 0

    for device in ("orangepi", "rasppi"):
        videos = collect_device_videos(session_dir, device)
        if not videos:
            print(f"[skip] {device}: no matching videos found")
            continue

        output_path = combine_device_videos(
            session_dir,
            device,
            videos,
            overwrite=overwrite,
            preset=preset,
            crf=crf,
        )
        if output_path is not None:
            built += 1

    return built


def main() -> int:
    args = parse_args()
    require_binary("ffmpeg")
    require_binary("ffprobe")

    sessions = resolve_sessions(args)
    built = 0
    for session_dir in sessions:
        built += process_session(
            session_dir,
            overwrite=args.overwrite,
            preset=args.preset,
            crf=args.crf,
        )

    print(f"\n[done] built {built} composite video set(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
