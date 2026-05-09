#!/usr/bin/env python3
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_RECORDINGS_DIR = Path(__file__).resolve().parents[2] / "hardware" / "recordings"
VIDEO_EXTENSIONS = {".mp4", ".mkv", ".mov", ".avi", ".webm"}
KNOWN_TOP_LEVEL_ENTRIES = {
    "session.json",
    "telemetry.csv",
    "gps_track.geojson",
    "videos",
    "composites",
}


@dataclass(frozen=True)
class RecordingSession:
    session_id: str
    path: Path
    started_at_iso: str | None
    ended_at_iso: str | None
    duration_s: float | None
    telemetry_rows: int | None
    gps_points: int | None
    video_count: int
    composite_count: int
    size_bytes: int
    has_session_json: bool
    has_telemetry: bool
    has_gps_track: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "path": str(self.path),
            "started_at_iso": self.started_at_iso,
            "ended_at_iso": self.ended_at_iso,
            "duration_s": self.duration_s,
            "telemetry_rows": self.telemetry_rows,
            "gps_points": self.gps_points,
            "video_count": self.video_count,
            "composite_count": self.composite_count,
            "size_bytes": self.size_bytes,
            "has_session_json": self.has_session_json,
            "has_telemetry": self.has_telemetry,
            "has_gps_track": self.has_gps_track,
        }


def is_recording_session(path: Path) -> bool:
    if not path.is_dir():
        return False
    return any(
        candidate.exists()
        for candidate in (
            path / "session.json",
            path / "telemetry.csv",
            path / "gps_track.geojson",
            path / "videos",
            path / "composites",
        )
    )


def list_recording_dirs(recordings_dir: Path) -> list[Path]:
    if not recordings_dir.exists():
        return []
    return sorted(
        path
        for path in recordings_dir.iterdir()
        if is_recording_session(path)
    )


def resolve_session_dirs(
    raw_sessions: list[str],
    recordings_dir: Path,
    *,
    select_all: bool = False,
    latest: int | None = None,
    default_latest: int | None = None,
) -> list[Path]:
    recordings_dir = recordings_dir.resolve()
    if select_all and raw_sessions:
        raise ValueError("Cannot use --all together with explicit session names.")
    if latest is not None and raw_sessions:
        raise ValueError("Cannot use --latest together with explicit session names.")
    if select_all and latest is not None:
        raise ValueError("Cannot use --all together with --latest.")

    available = list_recording_dirs(recordings_dir)
    if not available:
        raise ValueError(f"No recording sessions found in {recordings_dir}")

    if select_all:
        selected = available
    elif raw_sessions:
        selected = [resolve_session_dir(raw, recordings_dir) for raw in raw_sessions]
    else:
        count = latest if latest is not None else default_latest
        if count is None:
            selected = available
        else:
            if count < 1:
                raise ValueError("--latest must be at least 1.")
            selected = available[-count:]

    return _dedupe_paths(selected)


def resolve_session_dir(raw_session: str, recordings_dir: Path) -> Path:
    candidate = Path(raw_session)
    if not candidate.exists():
        candidate = recordings_dir / raw_session
    candidate = candidate.resolve()
    if not is_recording_session(candidate):
        raise ValueError(f"Recording session not found or incomplete: {candidate}")
    return candidate


def build_recording_session(session_dir: Path, *, compute_size: bool = True) -> RecordingSession:
    metadata = _load_json(session_dir / "session.json")
    videos_dir = session_dir / "videos"
    composites_dir = session_dir / "composites"

    return RecordingSession(
        session_id=session_dir.name,
        path=session_dir,
        started_at_iso=_coerce_str(metadata.get("started_at_iso")),
        ended_at_iso=_coerce_str(metadata.get("ended_at_iso")),
        duration_s=_coerce_float(metadata.get("duration_s")),
        telemetry_rows=_coerce_int(metadata.get("telemetry_rows")),
        gps_points=_coerce_int(metadata.get("gps_points")),
        video_count=_count_files(videos_dir, VIDEO_EXTENSIONS),
        composite_count=_count_files(composites_dir, VIDEO_EXTENSIONS),
        size_bytes=_path_size(session_dir) if compute_size else 0,
        has_session_json=(session_dir / "session.json").is_file(),
        has_telemetry=(session_dir / "telemetry.csv").is_file(),
        has_gps_track=(session_dir / "gps_track.geojson").is_file(),
    )


def format_bytes(size_bytes: int) -> str:
    if size_bytes <= 0:
        return "0 B"
    units = ("B", "KB", "MB", "GB", "TB")
    value = float(size_bytes)
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            if unit == "B":
                return f"{int(value)} {unit}"
            return f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{size_bytes} B"


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _count_files(directory: Path, suffixes: set[str]) -> int:
    if not directory.is_dir():
        return 0
    return sum(1 for path in directory.iterdir() if path.is_file() and path.suffix.lower() in suffixes)


def _path_size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    if not path.is_dir():
        return 0

    total = 0
    for child in path.rglob("*"):
        if child.is_file():
            try:
                total += child.stat().st_size
            except OSError:
                continue
    return total


def _coerce_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _coerce_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _dedupe_paths(paths: list[Path]) -> list[Path]:
    seen: set[Path] = set()
    result: list[Path] = []
    for path in paths:
        if path in seen:
            continue
        seen.add(path)
        result.append(path)
    return result
