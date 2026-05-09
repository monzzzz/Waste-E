#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from recording_catalog import (
    DEFAULT_RECORDINGS_DIR,
    build_recording_session,
    format_bytes,
    resolve_session_dirs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="List Waste-E recording sessions under hardware/recordings."
    )
    parser.add_argument(
        "session",
        nargs="*",
        help="Optional session name(s) or path(s) to inspect.",
    )
    parser.add_argument(
        "--recordings-dir",
        type=Path,
        default=DEFAULT_RECORDINGS_DIR,
        help=f"Recordings root directory (default: {DEFAULT_RECORDINGS_DIR})",
    )
    parser.add_argument(
        "--latest",
        type=int,
        metavar="N",
        help="Show only the latest N sessions.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print session details as JSON.",
    )
    parser.add_argument(
        "--no-size",
        action="store_true",
        help="Skip directory size calculation.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        sessions = resolve_session_dirs(
            args.session,
            args.recordings_dir,
            latest=args.latest,
            default_latest=None,
        )
    except ValueError as exc:
        raise SystemExit(str(exc))

    details = [
        build_recording_session(path, compute_size=not args.no_size)
        for path in sessions
    ]

    if args.json:
        print(json.dumps([detail.to_dict() for detail in details], indent=2))
        return 0

    _print_table(details, show_size=not args.no_size)
    return 0


def _print_table(details: list, *, show_size: bool) -> None:
    headers = [
        "Session",
        "Start",
        "Dur(s)",
        "Videos",
        "Comp",
        "Telemetry",
        "GPS",
    ]
    if show_size:
        headers.append("Size")

    rows: list[list[str]] = []
    for detail in details:
        row = [
            detail.session_id,
            _short_timestamp(detail.started_at_iso),
            _fmt_number(detail.duration_s, precision=1),
            str(detail.video_count),
            str(detail.composite_count),
            _fmt_int(detail.telemetry_rows),
            _fmt_int(detail.gps_points),
        ]
        if show_size:
            row.append(format_bytes(detail.size_bytes))
        rows.append(row)

    widths = [len(header) for header in headers]
    for row in rows:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(cell))

    header_line = "  ".join(header.ljust(widths[index]) for index, header in enumerate(headers))
    separator = "  ".join("-" * widths[index] for index in range(len(headers)))
    print(header_line)
    print(separator)
    for row in rows:
        print("  ".join(cell.ljust(widths[index]) for index, cell in enumerate(row)))


def _short_timestamp(value: str | None) -> str:
    if not value:
        return "-"
    return value.replace("T", " ")[:19]


def _fmt_number(value: float | None, *, precision: int) -> str:
    if value is None:
        return "-"
    return f"{value:.{precision}f}"


def _fmt_int(value: int | None) -> str:
    if value is None:
        return "-"
    return str(value)


if __name__ == "__main__":
    sys.exit(main())
